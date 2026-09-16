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
    mask_summary,
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
    # ``ref_mask`` is a property of the data, not of the fit: a checkpoint written
    # before the mask existed must not silently restore an all-observed mask.
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    stale = [k for k in missing if not k.endswith("ref_mask")]
    if stale or unexpected:
        print(f"[resume] WARNING state-dict mismatch: missing={stale} unexpected={list(unexpected)}")
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
    p.add_argument("--min-coverage", type=float, default=None,
                   help="Treat coverage below this as no signal. Applies to reference AND bulk "
                        "tracks; coverage <= 0 always counts as no signal.")
    p.add_argument("--out-dir", default="output/methylmix")
    p.add_argument("--dump-per-locus", choices=["auto", "always", "never"], default="auto",
                   help="Per-locus output tables cost one row per locus per component/sample; "
                        "'auto' skips them above 200k loci (spec §15).")
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
    p.add_argument("--recon-loss", choices=["huber", "mse"], default="huber",
                   help="Reconstruction error; the mask/weight gating is identical either way.")
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
    p.add_argument("--synthetic-ref-masked", default=None, metavar="CELL:i,j;CELL:k",
                   help="Synthetic only: drop individual reference observations, e.g. 'Astro:0,1'.")
    p.add_argument("--synthetic-all-ref-masked-loci", default=None, metavar="i,j,k",
                   help="Synthetic only: loci no reference observes (dropped from the loss).")
    # minimal in-memory toy (100 loci x 4 cell types); see toy_case() at the bottom
    p.add_argument("--toy", action="store_true", help="Run the minimal toy case (100 loci, 4 cell types) and exit.")
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
        recon_loss=args.recon_loss,
        target_mae=args.target_mae,
        target_rmse=args.target_rmse,
    )


def parse_ref_masked(spec: str | None) -> dict[str, list[int]] | None:
    """Parse ``'Astro:0,1;GABA:5'`` into ``{'Astro': [0, 1], 'GABA': [5]}``."""
    if not spec:
        return None
    out: dict[str, list[int]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(f"--synthetic-ref-masked: expected 'CELL:i,j', got {chunk!r}")
        cell, loci = chunk.split(":", 1)
        out[cell.strip()] = [int(v) for v in loci.split(",") if v.strip()]
    return out


def parse_loci(spec: str | None) -> list[int] | None:
    """Parse ``'10,11,12'`` into ``[10, 11, 12]`` (None when empty)."""
    if not spec:
        return None
    return [int(v) for v in spec.split(",") if v.strip()]


def print_mask_qc(prefix: str, data: AlignedData) -> None:
    """Report how much reference signal was actually available (spec §3.1)."""
    if data.ref_mask is None:
        return
    locus_valid = data.locus_valid
    n = data.n_loci
    print(f"{prefix} reference mask: {int(locus_valid.sum())}/{n} loci observed by >=1 reference, "
          f"{int((~locus_valid).sum())} by none (excluded from the loss)")
    for j, cell in enumerate(data.ref_cell_types):
        observed = int(data.ref_mask[j].sum())
        print(f"{prefix}   {cell:<12} observed {observed}/{n} ({observed / n:.1%})")
    if data.dropped_dmr_ids:
        shown = ", ".join(data.dropped_dmr_ids[:5])
        more = f" ... (+{len(data.dropped_dmr_ids) - 5})" if len(data.dropped_dmr_ids) > 5 else ""
        print(f"{prefix} dropped {len(data.dropped_dmr_ids)} DMR(s) no reference can inform: {shown}{more}")


def main() -> int:
    args = build_parser().parse_args()
    if args.toy:
        toy_case()
        return 0
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
            ref_masked_loci=parse_ref_masked(args.synthetic_ref_masked),
            all_ref_masked_loci=parse_loci(args.synthetic_all_ref_masked_loci),
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
    print_mask_qc("[STEP 0]", data)

    cfg = _make_config(args)
    model = make_model(data, cfg=cfg, composition_mode=args.composition_mode).to(device)
    batch: MixtureBatch = data.batch.to(device)

    # ---------------- STEP 1 ----------------
    print("[STEP 1] Reference fitting")
    ref_counts = torch.tensor([1.0 if c in data.ref_cell_types else 0.0 for c in model.cell_type_names])
    print(f"[STEP 1] {len(data.ref_cell_types)} reference cell types, "
          f"OTHER={'on' if cfg.use_other else 'off'}, delta_scope={cfg.delta_scope}")
    if cfg.use_other:
        other = torch.sigmoid(model.other_logits).detach()
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
        "mask": mask_summary(model, batch),
        "dropped_dmr_ids": data.dropped_dmr_ids,
        "config": {
            "composition_mode": cfg.composition_mode,
            "use_other": cfg.use_other,
            "delta_scope": cfg.delta_scope,
            "recon_loss": cfg.recon_loss,
            "lambda_other_anchor": cfg.lambda_other_anchor,
            "lambda_delta_l1": cfg.lambda_delta_l1,
            "target_mae": cfg.target_mae,
            "min_coverage": args.min_coverage,
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
    files = write_outputs(out_dir, model, data, qc, dump_per_locus=args.dump_per_locus)
    print("[STEP 5] wrote: " + ", ".join(Path(p).name for p in files.values()))
    print(f"[STEP 5] control_rmse={metrics['control']['rmse']:.4f} "
          f"disease_rmse={metrics['disease']['rmse']:.4f} "
          f"delta_nonzero={metrics['delta_sparsity']['fraction_nonzero']:.3f} "
          f"loci_without_reference={qc['mask']['n_loci_all_refs_masked']}/{data.n_loci} "
          f"dropped_dmrs={len(data.dropped_dmr_ids)}")
    print("[DONE]")
    return 0


def toy_case(
    n_loci: int = 100,
    cell_types: tuple[str, ...] = ("Glut", "GABA", "Astro", "Oligo"),
    n_ctrl: int = 3,
    n_dis: int = 3,
    n_dmr: int = 100,
    delta_spec: dict[str, float] | None = None,
    use_other: bool = False,
    composition_mode: str = "dirichlet_prior",
    seed: int = 0,
    mask_demo: bool = True,
) -> dict:
    """Minimal in-memory toy: 100 loci x 4 cell types, no files, a few seconds.

    Shows what the model actually computes:
        y_hat_CTRL = sum_c pi_c * theta0_c          (raw reference values, no transform)
        y_hat_DIS  = sum_c pi_c * sigmoid(logit(theta0_c) + delta_c)
    with a known delta injected into one or more cell types, so the attribution
    is checkable. Default injection: Astro +1.5 and Oligo +0.5.

    ``use_other=False`` keeps the components exactly the 4 named cell types;
    pass ``use_other=True`` to add the OTHER component as a 5th.

    With ``mask_demo`` a few reference observations are masked out and a DMR is
    made missing in every reference, then the run shows that neither reaches the
    loss and that the unreachable DMR is dropped.
    """
    if delta_spec is None:
        delta_spec = {"Astro": 1.5, "Oligo": 0.5}
    if not delta_spec:
        raise ValueError("delta_spec must contain at least one cell type")

    print("=" * 72)
    print(f"TOY CASE: {n_loci} loci x {len(cell_types)} cell types"
          f"{' + OTHER' if use_other else ''}, {n_ctrl} CTRL / {n_dis} DIS samples")
    print("=" * 72)

    torch.manual_seed(seed)
    masked_cell = "Astro" if "Astro" in cell_types else cell_types[0]
    # Keep locus 0 fully observed: the worked example below reads it.
    ref_masked_loci = {masked_cell: [2, 3]} if mask_demo else None
    all_ref_masked_loci: list[int] | None = None
    if mask_demo:
        # Only a locus that IS a DMR can be dropped as a DMR, so discover which
        # loci the generator marks as DMR under this seed first (it is seeded, so
        # the second call reproduces the same locus set).
        probe, probe_truth = make_synthetic_data(
            n_loci=n_loci, cell_types=cell_types, n_ctrl=n_ctrl, n_dis=n_dis,
            n_dmr=n_dmr, delta_spec=delta_spec, seed=seed, noise=0.005, use_other=use_other,
        )
        dmr_loci = [r for r in probe_truth["dmr_loci"].nonzero().flatten().tolist() if r > 0]
        all_ref_masked_loci = dmr_loci[:2] or None
        del probe, probe_truth

    data, truth = make_synthetic_data(
        n_loci=n_loci, cell_types=cell_types, n_ctrl=n_ctrl, n_dis=n_dis,
        n_dmr=n_dmr, delta_spec=delta_spec,
        seed=seed, noise=0.005, use_other=use_other,
        ref_masked_loci=ref_masked_loci, all_ref_masked_loci=all_ref_masked_loci,
    )
    model = make_model(data, composition_mode=composition_mode)
    batch = data.batch

    comps = model.cell_type_names
    print(f"shapes      : theta_ref {tuple(model.theta_ref.shape)}  "
          f"pi_ctrl {tuple(batch.pi0_ctrl.shape)}  delta {tuple(model.delta.shape)}")
    print(f"components  : {comps}   (+{len(data.ref_cell_types)} references, "
          f"OTHER={'on' if model.cfg.use_other else 'off'})")

    # ---- identity 1: the mixture is literally sum_c pi_c * theta0_c ----------
    manual = torch.einsum("sc,cr->sr", model.composition("CTRL"), model.theta0())
    identity_mix = torch.allclose(manual, model.predict_control(), atol=1e-6)
    # ---- identity 2: delta == 0  =>  perturbed tracks == baseline tracks -----
    identity_delta0 = torch.allclose(model.theta_disease(), model.theta0(), atol=1e-7)
    pi = model.composition("CTRL")
    simplex_ok = bool(torch.allclose(pi.sum(dim=-1), torch.ones(pi.shape[0]), atol=1e-6))
    print(f"identity    : sum_c pi_c*theta0_c == predict_control()  -> {'OK' if identity_mix else 'FAIL'}")
    print(f"              delta=0 => theta_disease == theta0        -> {'OK' if identity_delta0 else 'FAIL'}")
    print(f"              pi rows sum to 1 (CTRL)                   -> {'OK' if simplex_ok else 'FAIL'}")

    # ---- masking: an unobserved entry and an unreachable DMR ---------------
    mask_report: dict = {}
    if mask_demo:
        masked_entries = int((~data.ref_mask).sum())
        print("-" * 72)
        print(f"mask demo   : {masked_cell} unobserved at loci {ref_masked_loci[masked_cell]}; "
              f"loci {all_ref_masked_loci} unobserved by EVERY reference")
        print(f"              masked (cell, locus) entries: {masked_entries}; "
              f"loci still usable: {int(model.locus_valid().sum())}/{model.n_loci}; "
              f"DMRs dropped: {truth['dropped_dmr_ids']}")

        # (a) the masked entry is inert: delta is forced to 0 there, so perturbing
        #     it cannot change the loss.
        masked_row = data.ref_cell_types.index(masked_cell)
        masked_locus = ref_masked_loci[masked_cell][0]
        loss_before = float(model.loss(batch, stage="dis")[0].detach())
        with torch.no_grad():
            model.delta[masked_row, masked_locus] = 5.0
            delta_at_masked = float(model.delta_effective()[masked_row, masked_locus])
            model.delta[masked_row, masked_locus] = 0.0
        loss_after = float(model.loss(batch, stage="dis")[0].detach())
        delta_move = abs(loss_after - loss_before)
        print(f"              delta[{masked_row},{masked_locus}] ({masked_cell}) set to +5.0: "
              f"delta_eff={delta_at_masked:.1f} (zeroed), loss moved by "
              f"{delta_move:.2e}")

        # (b) a locus no reference observes is out of the loss too: rewriting the
        #     bulk there changes nothing.
        probe_locus = int(all_ref_masked_loci[0])
        loss_before = float(model.loss(batch, stage="dis")[0].detach())
        with torch.no_grad():
            saved_col = batch.y_ctrl[:, probe_locus].clone()
            batch.y_ctrl[:, probe_locus] = 0.123456
        loss_after = float(model.loss(batch, stage="dis")[0].detach())
        with torch.no_grad():
            batch.y_ctrl[:, probe_locus] = saved_col
        locus_move = abs(loss_after - loss_before)
        print(f"              bulk at unobserved locus {probe_locus} rewritten: "
              f"loss moved by {locus_move:.2e} (masked entries never reach the "
              f"loss, so both moves must be 0)")
        mask_report = {
            "masked_cell": masked_cell,
            "ref_masked_loci": {masked_cell: ref_masked_loci[masked_cell]},
            "all_ref_masked_loci": all_ref_masked_loci,
            "dropped_dmr_ids": truth["dropped_dmr_ids"],
            "n_masked_entries": masked_entries,
            "n_loci_usable": int(model.locus_valid().sum()),
            "delta_at_masked_entry": delta_at_masked,
            "loss_move_from_delta": delta_move,
            "loss_move_from_bulk_rewrite": locus_move,
        }

    # ---- worked example at one locus (raw values, weighted by composition) ---
    r = 0
    pi_mean = pi.detach().mean(dim=0)
    theta0_det = model.theta0().detach()
    terms = [f"{float(pi_mean[c]):.3f}*{float(theta0_det[c, r]):.3f}" for c in range(len(comps))]
    print(f"locus {data.locus_df['locus_id'][r]}: " + " + ".join(terms)
          + f" = {float(model.predict_control().detach()[0, r]):.4f}"
          f"   (observed bulk: {float(batch.y_ctrl[0, r]):.4f})")

    def _fit_and_report(stage: str, epochs: int) -> dict:
        fit_stage(model, batch, stage, lr=5e-3, max_epochs=epochs, patience=epochs // 4, verbose=False)
        metrics = stage_metrics(model, batch)
        pred = (model.predict_control() if stage == "ctrl" else model.predict_disease()).detach()
        key = "control" if stage == "ctrl" else "disease"
        m = metrics[key]
        in_range = bool(float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0)
        print(f"[{stage:5s}] RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  pearson={m['pearson']:.4f}  "
              f"pred in [0,1]: {'OK' if in_range else 'FAIL'}")
        return m

    print("-" * 72)
    ctrl_metrics = _fit_and_report("ctrl", 600)
    dis_metrics = _fit_and_report("dis", 800)

    # ---- recovered delta, restricted to the DMR loci ------------------------
    delta = model.delta_effective().detach()
    loc = truth["dmr_loci"]
    per_cell = delta[:, loc].mean(dim=1)                       # [C] signed, averaged over DMR loci
    mag = delta[:, loc].abs().mean(dim=1)                      # [C] magnitude ranking
    ranked = torch.argsort(mag, descending=True).tolist()
    name_to_idx = {c: i for i, c in enumerate(model.cell_type_names)}
    injected = dict(truth["delta_spec"])

    print("-" * 72)
    print(f"injected delta: " + ", ".join(f"{c}={v:+.2f}" for c, v in injected.items())
          + f"   on {int(loc.sum())} DMR loci")
    for j, cell in enumerate(model.cell_type_names):
        marker = f"   <- injected {injected[cell]:+.2f}" if cell in injected else ""
        print(f"  recovered delta {cell:<6} = {float(per_cell[j]):+.4f}{marker}")

    k = len(injected)
    top_k = [model.cell_type_names[j] for j in ranked[:k]]
    set_ok = set(top_k) == set(injected)
    sign_ok = all(float(per_cell[name_to_idx[c]]) * v > 0 for c, v in injected.items() if v != 0.0)
    truth_order = sorted(injected, key=lambda c: -abs(injected[c]))
    recovered_order = sorted(injected, key=lambda c: -float(per_cell[name_to_idx[c]]))
    order_ok = recovered_order == truth_order
    truth_vec = torch.zeros(len(model.cell_type_names))
    for c, v in injected.items():
        truth_vec[name_to_idx[c]] = v
    corr = float("nan")
    if float(per_cell.std()) > 0:
        corr = float(torch.corrcoef(torch.stack([per_cell, truth_vec]))[0, 1])

    print(f"recovery   : top-{k} by |delta| = {top_k}  (injected: {list(injected)})  "
          f"-> {'OK' if set_ok else 'FAIL'}")
    print(f"             signs {'OK' if sign_ok else 'FAIL'}   "
          f"magnitude order {'OK' if order_ok else 'FAIL'} "
          f"(recovered {[round(float(per_cell[name_to_idx[c]]), 3) for c in truth_order]}"
          f" vs injected {[injected[c] for c in truth_order]})   per-cell corr={corr:.3f}")
    print("=" * 72)

    recovered = bool(set_ok and sign_ok and order_ok)
    return {
        "shape_theta_ref": tuple(model.theta_ref.shape),
        "shape_delta": tuple(model.delta.shape),
        "components": comps,
        "identity_mixture": bool(identity_mix),
        "identity_delta_zero": bool(identity_delta0),
        "simplex_ok": simplex_ok,
        "ctrl_mae": ctrl_metrics["mae"],
        "dis_mae": dis_metrics["mae"],
        "dis_pearson": dis_metrics["pearson"],
        "delta_spec_injected": injected,
        "recovered_delta": {c: float(per_cell[name_to_idx[c]]) for c in model.cell_type_names},
        "top_k_by_magnitude": top_k,
        "set_recovered": bool(set_ok),
        "sign_ok": bool(sign_ok),
        "magnitude_order_ok": bool(order_ok),
        "per_cell_corr": corr,
        "recovered": recovered,
        "mask": mask_report,
    }


if __name__ == "__main__":
    raise SystemExit(main())
