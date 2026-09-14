#!/usr/bin/env python
"""End-to-end runner for the reference-anchored methylation mixture model.

Implements the staged workflow of the spec (§18, §23):

    [STEP 0] Input validation / alignment      (or synthetic benchmark)
    [STEP 1] Reference distribution + OTHER init
    [STEP 2] Control mixture fitting  -> QC gate
    [STEP 3] Disease delta fitting    (DMR-only, logit space, before mixing)
    [STEP 4] Attribution (composition vs cell-intrinsic)
    [STEP 5] QC + outputs

Each stage writes a checkpoint and a `stage{N}_done.json` marker; `--resume`
continues from the latest completed stage.

Examples
--------
    # self-contained synthetic benchmark (no input files needed)
    python run_methylmix.py --synthetic --out-dir output/methylmix_demo

    # real data (see spec §3 for the required columns)
    python run_methylmix.py \
        --reference ref_tracks.tsv.gz --ctrl bulk_ctrl.tsv.gz --dis bulk_dis.tsv.gz \
        --prior sc_prior.tsv --dmr dmr.tsv.gz --out-dir output/methylmix_run
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from methylmix_data import (
    AlignedData,
    build_aligned_data,
    make_model,
    make_synthetic_data,
    write_outputs,
)
from methylmix_model import (
    MixtureBatch,
    MixtureConfig,
    ReferenceAnchoredMethylationMixture,
    fit_stage,
    stage_metrics,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stage_marker(out_dir: Path, stage: int) -> Path:
    return out_dir / f"stage{stage}_done.json"


def stage_checkpoint(out_dir: Path, stage: int) -> Path:
    return out_dir / f"stage{stage}_checkpoint.pt"


def save_stage(
    out_dir: Path,
    stage: int,
    model: ReferenceAnchoredMethylationMixture,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        stage_checkpoint(out_dir, stage),
    )
    stage_marker(out_dir, stage).write_text(json.dumps({"stage": stage, "metrics": metrics}, indent=2, default=float))
    print(f"[STEP {stage}] checkpoint -> {stage_checkpoint(out_dir, stage).name}")


def load_stage(out_dir: Path, stage: int, model: ReferenceAnchoredMethylationMixture) -> dict:
    payload = torch.load(stage_checkpoint(out_dir, stage), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    print(f"[resume] loaded stage {stage} checkpoint ({stage_checkpoint(out_dir, stage).name})")
    return payload.get("metrics", {})


def top_attribution(model: ReferenceAnchoredMethylationMixture, n: int = 5) -> list[dict]:
    """Largest per-cell-type intrinsic contributions (spec §14)."""
    att = model.attribution()
    intrinsic = att["celltype_intrinsic"]
    mag = intrinsic.abs().mean(dim=1)
    order = torch.argsort(mag, descending=True)[:n]
    return [
        {
            "cell_type": model.cell_type_names[int(j)],
            "mean_abs_intrinsic": float(mag[int(j)]),
            "mean_signed_intrinsic": float(intrinsic[int(j)].mean()),
        }
        for j in order
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reference-anchored cell-type methylation mixture model (deterministic v1).")
    # inputs
    p.add_argument("--reference", default=None, help="Reference cell-type tracks TSV/CSV (spec §3.1).")
    p.add_argument("--ctrl", default=None, help="Bulk CTRL tracks (spec §3.2).")
    p.add_argument("--dis", default=None, help="Bulk DIS tracks (spec §3.2).")
    p.add_argument("--prior", default=None, help="Single-cell composition prior (spec §3.3).")
    p.add_argument("--dmr", default=None, help="DMR annotation (spec §3.4).")
    p.add_argument("--min-coverage", type=float, default=None)
    p.add_argument("--out-dir", default="output/methylmix")
    # model
    p.add_argument("--composition-mode", choices=["dirichlet_prior", "fixed"], default="dirichlet_prior")
    p.add_argument("--no-other", action="store_true", help="Disable the OTHER component (spec §16.4 sensitivity run).")
    p.add_argument("--delta-scope", choices=["dmr_only", "all_loci"], default="dmr_only")
    p.add_argument("--prior-strength", type=float, default=100.0)
    # loss weights
    p.add_argument("--lambda-ctrl-recon", type=float, default=10.0)
    p.add_argument("--lambda-dis-recon", type=float, default=10.0)
    p.add_argument("--lambda-delta-l1", type=float, default=0.01)
    p.add_argument("--lambda-delta-group", type=float, default=0.0)
    p.add_argument("--lambda-other-anchor", type=float, default=1.0)
    p.add_argument("--lambda-smooth", type=float, default=0.0)
    p.add_argument("--huber-delta", type=float, default=0.05)
    # training
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-5)
    p.add_argument("--max-epochs-stage2", type=int, default=2000)
    p.add_argument("--max-epochs-stage3", type=int, default=3000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--target-mae", type=float, default=0.03)
    p.add_argument("--target-rmse", type=float, default=0.05)
    p.add_argument("--force-continue", action="store_true", help="Do not stop when control QC fails.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cpu", action="store_true")
    # synthetic benchmark
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic-loci", type=int, default=300)
    p.add_argument("--synthetic-dmr", type=int, default=60)
    p.add_argument("--synthetic-delta-logit", type=float, default=1.2)
    p.add_argument("--synthetic-delta-celltype", default="Astro")
    return p


def _make_config(args: argparse.Namespace) -> MixtureConfig:
    return MixtureConfig(
        composition_mode=args.composition_mode,
        composition_prior_strength=args.prior_strength,
        use_other=not args.no_other,
        delta_scope=args.delta_scope,
        lambda_ctrl_recon=args.lambda_ctrl_recon,
        lambda_dis_recon=args.lambda_dis_recon,
        lambda_delta_l1=args.lambda_delta_l1,
        lambda_delta_group=args.lambda_delta_group,
        lambda_other_anchor=args.lambda_other_anchor,
        lambda_smooth=args.lambda_smooth,
        huber_delta=args.huber_delta,
        target_mae=args.target_mae,
        target_rmse=args.target_rmse,
    )


def main() -> int:
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = None

    # ---------------- STEP 0 ----------------
    print("[STEP 0] Input validation")
    if args.synthetic:
        data, truth = make_synthetic_data(
            n_loci=args.synthetic_loci,
            n_dmr=args.synthetic_dmr,
            delta_logit=args.synthetic_delta_logit,
            delta_cell_type=args.synthetic_delta_celltype,
            use_other=not args.no_other,
            seed=args.seed,
        )
        print(f"[STEP 0] synthetic benchmark: {data.n_loci} loci, "
              f"{len(data.ctrl_sample_ids)} CTRL / {len(data.dis_sample_ids)} DIS samples, "
              f"components={data.cell_types_all}")
    else:
        for name in ("reference", "ctrl", "dis", "prior"):
            if getattr(args, name) is None:
                raise SystemExit(f"--{name} is required (or use --synthetic)")
        data = build_aligned_data(
            args.reference, args.ctrl, args.dis, args.prior, args.dmr,
            min_coverage=args.min_coverage, use_other=not args.no_other,
        )
        print(f"[STEP 0] aligned {data.n_loci} loci, {len(data.ctrl_sample_ids)} CTRL / "
              f"{len(data.dis_sample_ids)} DIS samples, components={data.cell_types_all}")

    cfg = _make_config(args)
    model = make_model(data, cfg=cfg, composition_mode=args.composition_mode).to(device)
    batch: MixtureBatch = data.batch.to(device)

    # ---------------- STEP 1 ----------------
    print("[STEP 1] Reference fitting")
    ref_counts = torch.tensor([1.0 if c in data.ref_cell_types else 0.0 for c in model.cell_type_names])
    print(f"[STEP 1] {len(data.ref_cell_types)} reference cell types, "
          f"OTHER={'on' if cfg.use_other else 'off'}, delta_scope={cfg.delta_scope}")
    if cfg.use_other:
        other = torch.sigmoid(model.other_logits)
        print(f"[STEP 1] OTHER init: mean={float(other.mean()):.4f} "
              f"(from control residual: {'yes' if float(other.std()) > 0 else 'fallback'})")

    # ---------------- STEP 2 ----------------
    resumed_stage = 0
    if args.resume:
        for stage in (3, 2):
            if stage_marker(out_dir, stage).exists():
                load_stage(out_dir, stage, model)
                resumed_stage = stage
                break
    control_metrics: dict = {}
    if resumed_stage < 2:
        print("[STEP 2] Control mixture fitting")
        fit_stage(model, batch, "ctrl", lr=args.lr, weight_decay=args.weight_decay,
                  max_epochs=args.max_epochs_stage2, patience=args.patience,
                  log_every=args.log_every)
        control_metrics = stage_metrics(model, batch)["control"]
        print(f"[STEP 2] CTRL RMSE={control_metrics['rmse']:.4f} MAE={control_metrics['mae']:.4f} "
              f"pearson={control_metrics['pearson']:.4f} within_tol={control_metrics['within_tolerance']:.3f}")
        save_stage(out_dir, 2, model, control_metrics, args)
        if control_metrics["mae"] > args.target_mae and not args.force_continue:
            print(f"[STEP 2] control MAE {control_metrics['mae']:.4f} > target {args.target_mae} — "
                  f"stopping before disease attribution (spec §16.1). "
                  f"Use --force-continue to override.")
            return 2
    else:
        control_metrics = stage_metrics(model, batch)["control"]
        print(f"[STEP 2] skipped (resumed); CTRL MAE={control_metrics['mae']:.4f}")

    # ---------------- STEP 3 ----------------
    if resumed_stage < 3:
        print("[STEP 3] Disease delta fitting")
        fit_stage(model, batch, "dis", lr=args.lr, weight_decay=args.weight_decay,
                  max_epochs=args.max_epochs_stage3, patience=args.patience,
                  log_every=args.log_every)
    metrics = stage_metrics(model, batch)
    print(f"[STEP 3] DIS RMSE={metrics['disease']['rmse']:.4f} MAE={metrics['disease']['mae']:.4f} "
          f"pearson={metrics['disease']['pearson']:.4f} within_tol={metrics['disease']['within_tolerance']:.3f}")
    save_stage(out_dir, 3, model, metrics, args)

    # ---------------- STEP 4 ----------------
    print("[STEP 4] Attribution")
    top = top_attribution(model)
    for entry in top:
        print(f"[STEP 4]   {entry['cell_type']:<12} mean|intrinsic|={entry['mean_abs_intrinsic']:.5f} "
              f"signed={entry['mean_signed_intrinsic']:+.5f}")
    if truth is not None:
        delta = model.delta_effective().detach()
        mag = delta.abs()
        # Only DMR loci are identifiable; rank cell types on the injected loci.
        loc = truth["dmr_loci"]
        ranked = torch.argsort(mag[:, loc].mean(dim=1), descending=True).tolist()
        inferred = model.cell_type_names[ranked[0]]
        signed = float(delta[ranked[0]][loc].mean())
        print(f"[STEP 4] recovery: truth={truth['delta_cell_type']} inferred={inferred} "
              f"sign={'OK' if signed > 0 else 'WRONG'}  top3={[model.cell_type_names[j] for j in ranked[:3]]}")

    # ---------------- STEP 5 ----------------
    print("[STEP 5] QC")
    qc = {
        "control": metrics["control"],
        "disease": metrics["disease"],
        "delta_sparsity": metrics["delta_sparsity"],
        "composition_prior_deviation": metrics["composition_prior_deviation"],
        "config": {
            "composition_mode": cfg.composition_mode,
            "use_other": cfg.use_other,
            "delta_scope": cfg.delta_scope,
            "lambda_other_anchor": cfg.lambda_other_anchor,
            "lambda_delta_l1": cfg.lambda_delta_l1,
            "target_mae": cfg.target_mae,
        },
        "attribution_top": top,
    }
    if truth is not None:
        delta = model.delta_effective().detach()
        loc = truth["dmr_loci"]
        ranked = torch.argsort(delta.abs()[:, loc].mean(dim=1), descending=True).tolist()
        qc["synthetic_recovery"] = {
            "truth_cell_type": truth["delta_cell_type"],
            "inferred_top_cell_type": model.cell_type_names[ranked[0]],
            "top3": [model.cell_type_names[j] for j in ranked[:3]],
            "inferred_delta_mean_on_dmr": float(delta[ranked[0]][loc].mean()),
            "true_delta": float(args.synthetic_delta_logit),
        }
    files = write_outputs(out_dir, model, data, qc)
    print("[STEP 5] wrote: " + ", ".join(Path(p).name for p in files.values()))
    print(f"[STEP 5] control_rmse={metrics['control']['rmse']:.4f} "
          f"disease_rmse={metrics['disease']['rmse']:.4f} "
          f"delta_nonzero={metrics['delta_sparsity']['fraction_nonzero']:.3f}")
    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
