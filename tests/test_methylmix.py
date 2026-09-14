"""Mandatory unit tests for the reference-anchored methylation mixture model.

Implements spec §21 Test 1-6. Run with pytest or directly:

    python tests/test_methylmix.py

No external data is needed — everything is built in memory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methylmix_data import make_model, make_synthetic_data  # noqa: E402
from methylmix_model import (  # noqa: E402
    MixtureBatch,
    MixtureConfig,
    ReferenceAnchoredMethylationMixture,
    fit_stage,
    stage_metrics,
)


def _fixed_model(theta_ref, pi_ctrl, pi_dis=None, use_other=False, is_dmr=None):
    """Helper: build a model with frozen composition (spec §5 composition_mode=fixed)."""
    n_loci = theta_ref.shape[1]
    cfg = MixtureConfig(composition_mode="fixed", use_other=use_other)
    pi_ctrl_t = torch.tensor([pi_ctrl], dtype=torch.float32)
    pi_dis_t = torch.tensor([pi_dis if pi_dis is not None else pi_ctrl], dtype=torch.float32)
    return ReferenceAnchoredMethylationMixture(
        theta_ref=theta_ref.float(),
        pi0_ctrl=pi_ctrl_t,
        pi0_dis=pi_dis_t,
        is_dmr=torch.ones(n_loci, dtype=torch.bool) if is_dmr is None else is_dmr,
        cfg=cfg,
        cell_type_names=[f"cell{i}" for i in range(theta_ref.shape[0])],
    )


# ---------------------------------------------------------------------- #
# Test 1: mixture identity (spec §21)
# ---------------------------------------------------------------------- #
def test_mixture_identity():
    theta = torch.tensor([[0.8, 0.2], [0.2, 0.6]])
    model = _fixed_model(theta, [0.75, 0.25])
    pred = model.predict_control()
    expected = torch.tensor([[0.65, 0.30]])
    assert torch.allclose(pred, expected, atol=1e-6), f"got {pred}, expected {expected}"
    print("  Test 1 mixture identity            OK  ->", pred.tolist())


# ---------------------------------------------------------------------- #
# Test 2: simplex constraints (spec §21)
# ---------------------------------------------------------------------- #
def test_simplex():
    data, _ = make_synthetic_data(n_loci=40, n_ctrl=3, n_dis=3, seed=1)
    model = make_model(data)
    for condition in ("CTRL", "DIS"):
        pi = model.composition(condition)
        assert torch.all(pi >= 0), f"{condition}: negative fraction"
        assert torch.allclose(pi.sum(dim=-1), torch.ones(pi.shape[0]), atol=1e-6), \
            f"{condition}: rows do not sum to 1 (max err {float((pi.sum(-1)-1).abs().max())})"
    # delta=0 must also hold the disease track inside [0, 1]
    print(f"  Test 2 simplex                     OK  -> sum={float(model.composition('DIS').sum(-1).mean()):.6f}")


# ---------------------------------------------------------------------- #
# Test 3: methylation bounds (spec §21)
# ---------------------------------------------------------------------- #
def test_bounds():
    data, _ = make_synthetic_data(n_loci=50, n_ctrl=2, n_dis=2, seed=2)
    model = make_model(data)
    model.delta.data.normal_(0, 3.0)          # deliberately extreme perturbation
    for name, t in (
        ("theta0", model.theta0()),
        ("theta_dis", model.theta_disease()),
        ("pred_ctrl", model.predict_control()),
        ("pred_dis", model.predict_disease()),
    ):
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0, f"{name} out of [0,1]: {float(t.min())}, {float(t.max())}"
    print(f"  Test 3 methylation bounds          OK  -> theta_dis in "
          f"[{float(model.theta_disease().min()):.4f}, {float(model.theta_disease().max()):.4f}]")


# ---------------------------------------------------------------------- #
# Test 4: delta = 0 and equal composition  =>  pred_dis == pred_ctrl (spec §21)
# ---------------------------------------------------------------------- #
def test_zero_delta_identity():
    theta = torch.tensor([[0.7, 0.3, 0.5], [0.3, 0.6, 0.2]])
    model = _fixed_model(theta, [0.6, 0.4], pi_dis=[0.6, 0.4], use_other=False)
    assert torch.allclose(model.delta, torch.zeros_like(model.delta))
    assert torch.allclose(model.predict_disease(), model.predict_control(), atol=1e-7), \
        f"pred_dis {model.predict_disease()} != pred_ctrl {model.predict_control()}"
    print("  Test 4 delta=0 => pred_dis==pred_ctrl OK")


# ---------------------------------------------------------------------- #
# Test 5: missing reference absorbed by OTHER (spec §21 / §16.6)
# ---------------------------------------------------------------------- #
def test_missing_reference_runs():
    data, truth = make_synthetic_data(
        n_loci=80, n_ctrl=3, n_dis=3, missing_reference="Astro", seed=3, use_other=True
    )
    assert "Astro" not in data.ref_cell_types, "hidden reference still present"
    assert data.cell_types_all[-1] == "OTHER"
    model = make_model(data)
    batch = data.batch
    fit_stage(model, batch, "ctrl", max_epochs=150, patience=40)
    metrics = stage_metrics(model, batch)["control"]
    assert metrics["mae"] < 0.15, f"control MAE after hiding a reference too high: {metrics['mae']:.4f}"
    print(f"  Test 5 missing reference -> OTHER  OK  -> CTRL MAE={metrics['mae']:.4f}")


# ---------------------------------------------------------------------- #
# Test 6: synthetic delta recovery (spec §21 / §16.7)
# ---------------------------------------------------------------------- #
def test_synthetic_delta_recovery(verbose: bool = True):
    data, truth = make_synthetic_data(
        n_loci=250, n_ctrl=5, n_dis=5, delta_cell_type="Astro",
        n_dmr=60, delta_logit=1.5, seed=4, noise=0.005,
    )
    model = make_model(data)
    cfg = model.cfg
    fit_stage(model, data.batch, "ctrl", lr=5e-3, max_epochs=600, patience=150,
              log_every=0, verbose=False)
    ctrl = stage_metrics(model, data.batch)["control"]
    fit_stage(model, data.batch, "dis", lr=5e-3, max_epochs=800, patience=200,
              log_every=0, verbose=False)
    dis = stage_metrics(model, data.batch)["disease"]

    delta = model.delta_effective().detach()
    loc = truth["dmr_loci"]
    ranked = torch.argsort(delta.abs()[:, loc].mean(dim=1), descending=True).tolist()
    inferred = model.cell_type_names[ranked[0]]
    signed = float(delta[ranked[0]][loc].mean())

    assert inferred == "Astro", f"top cell type {inferred} != truth Astro (rank {ranked[:3]})"
    assert signed > 0, f"recovered delta sign wrong: {signed:.3f}"
    assert dis["mae"] < 0.06, f"disease MAE too high: {dis['mae']:.4f}"
    # Non-DMR loci must stay unperturbed (delta_scope=dmr_only invariant).
    off = delta[:, ~loc]
    assert float(off.abs().max()) < 1e-6, f"delta leaked off-DMR: {float(off.abs().max())}"
    if verbose:
        print(f"  Test 6 synthetic recovery          OK  -> inferred={inferred} signed={signed:+.3f} "
              f"| ctrl MAE={ctrl['mae']:.4f} dis MAE={dis['mae']:.4f}")
    return {"inferred": inferred, "signed_delta": signed, "ctrl_mae": ctrl["mae"], "dis_mae": dis["mae"]}


def main() -> int:
    torch.manual_seed(0)
    tests = [
        test_mixture_identity,
        test_simplex,
        test_bounds,
        test_zero_delta_identity,
        test_missing_reference_runs,
        test_synthetic_delta_recovery,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"  {test.__name__:<34} FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"  {test.__name__:<34} ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{'all tests passed' if failed == 0 else f'{failed} test(s) failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
