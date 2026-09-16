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

from methylmix_data import (  # noqa: E402
    drop_fully_masked_dmrs,
    make_model,
    make_synthetic_data,
)
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


# ---------------------------------------------------------------------- #
# Test 7: a masked bulk observation cannot reach the loss (spec §3.1, §11)
# ---------------------------------------------------------------------- #
def test_masked_bulk_entry_does_not_enter_loss():
    data, _ = make_synthetic_data(n_loci=30, n_ctrl=2, n_dis=2, n_dmr=10, seed=5)
    model = make_model(data)
    batch = data.batch

    # While the entry is observed its value drives the loss ...
    batch.y_ctrl[0, 0] = 0.95
    observed = float(model.loss(batch, stage="ctrl")[0].detach())

    # ... and masking it makes the value irrelevant: a non-finite placeholder and
    # a wild finite one both leave the loss exactly where the mask left it.
    # (NaN * 0 is NaN, so this also checks the loss cannot be poisoned.)
    batch.mask_ctrl[0, 0] = False
    batch.y_ctrl[0, 0] = float("nan")
    masked_nan = float(model.loss(batch, stage="ctrl")[0].detach())
    batch.y_ctrl[0, 0] = 123.0
    masked_wild = float(model.loss(batch, stage="ctrl")[0].detach())

    assert torch.isfinite(torch.tensor(masked_nan)), f"NaN at a masked entry poisoned the loss: {masked_nan}"
    assert masked_wild == masked_nan, \
        f"value at a masked entry changed the loss: {masked_nan} -> {masked_wild}"
    assert abs(masked_nan - observed) > 1e-6, "the entry should have moved the loss while observed"
    print(f"  Test 7 masked bulk entry skipped    OK  -> observed {observed:.6f} vs masked "
          f"{masked_nan:.6f}; NaN and 123.0 both ignored")


# ---------------------------------------------------------------------- #
# Test 8: a masked reference entry carries no delta (spec §3.1)
# ---------------------------------------------------------------------- #
def test_ref_mask_zeroes_delta():
    data, _ = make_synthetic_data(n_loci=20, n_ctrl=2, n_dis=2, n_dmr=20, seed=6)
    data.ref_mask[0, 0] = False
    model = make_model(data)
    assert not bool(model.component_mask()[0, 0]), "masked reference entry reported as observed"
    if model.cfg.use_other:
        assert bool(model.component_mask()[-1].all()), "OTHER must always count as available"

    model.delta.data.normal_(0, 2.0)
    assert float(model.delta_effective()[0, 0]) == 0.0, \
        f"delta survived on a masked entry: {float(model.delta_effective()[0, 0])}"

    before = float(model.loss(data.batch, stage="dis")[0].detach())
    with torch.no_grad():
        model.delta[0, 0] = 7.0
    after = float(model.loss(data.batch, stage="dis")[0].detach())
    assert abs(after - before) < 1e-7, f"delta on a masked entry changed the loss: {before} -> {after}"
    print(f"  Test 8 ref mask zeroes delta        OK  -> delta_eff=0.0, loss {before:.6f} unchanged")


# ---------------------------------------------------------------------- #
# Test 9: a locus no reference observes is out of the loss (spec §3.1)
# ---------------------------------------------------------------------- #
def test_all_ref_masked_locus_excluded():
    data, truth = make_synthetic_data(
        n_loci=25, n_ctrl=2, n_dis=2, n_dmr=0, seed=7, all_ref_masked_loci=[0]
    )
    assert truth["dropped_dmr_ids"] == [], "a non-DMR locus should not drop a DMR"
    model = make_model(data)
    assert not bool(model.locus_valid()[0]), "locus missing in every reference reported usable"
    assert int((~model.ref_mask[:, 0]).sum()) == model.n_ref, "not every reference was masked"
    # the placeholder must not be a silent zero (spec §24 criterion 7)
    assert float(data.theta_ref[:, 0].min()) > 0.0, \
        f"missing reference value was replaced by 0.0: {data.theta_ref[:, 0].tolist()}"

    before = float(model.loss(data.batch, stage="ctrl")[0].detach())
    with torch.no_grad():
        saved = data.batch.y_ctrl[:, 0].clone()
        data.batch.y_ctrl[:, 0] = 0.654321
    after = float(model.loss(data.batch, stage="ctrl")[0].detach())
    with torch.no_grad():
        data.batch.y_ctrl[:, 0] = saved
    assert abs(after - before) < 1e-7, f"bulk at an unobserved locus changed the loss: {before} -> {after}"
    print(f"  Test 9 all-ref-masked locus skipped OK  -> loss {before:.6f} unchanged, "
          f"placeholder={float(data.theta_ref[0, 0]):.4f} (not 0)")


# ---------------------------------------------------------------------- #
# Test 10: a DMR no reference can inform is dropped as a unit (spec §3.1)
# ---------------------------------------------------------------------- #
def test_all_ref_masked_dmr_dropped():
    # unit: dropped only when EVERY locus of the DMR is unreachable
    is_dmr = torch.tensor([True] * 3 + [True] * 3)
    dmr_ids = ["Dfull"] * 3 + ["Dpart"] * 3
    ref_mask = torch.ones(2, 6, dtype=torch.bool)
    ref_mask[:, :3] = False       # Dfull: no reference sees any of its loci
    ref_mask[0, 3] = False        # Dpart: one entry masked, the region stays reachable
    kept, dropped = drop_fully_masked_dmrs(is_dmr, dmr_ids, ref_mask)
    assert dropped == ["Dfull"], f"expected only Dfull to be dropped, got {dropped}"
    assert kept.tolist() == [False] * 3 + [True] * 3, f"is_dmr wrongly rewritten: {kept.tolist()}"

    # end-to-end: pick a locus that IS a DMR, then make it invisible to every reference
    _, probe_truth = make_synthetic_data(n_loci=30, n_ctrl=2, n_dis=2, n_dmr=10, seed=8)
    target = int(probe_truth["dmr_loci"].nonzero().flatten()[0])
    data, truth = make_synthetic_data(
        n_loci=30, n_ctrl=2, n_dis=2, n_dmr=10, seed=8, all_ref_masked_loci=[target]
    )
    assert truth["dropped_dmr_ids"] == [f"D{target}"], \
        f"dropped DMRs {truth['dropped_dmr_ids']} != ['D{target}']"
    assert not bool(data.locus_df["is_dmr"][target]), "dropped DMR kept is_dmr=True"
    assert not bool(truth["dmr_loci"][target]), "dropped DMR still counted as a truth DMR"

    model = make_model(data)
    model.delta.data.normal_(0, 2.0)
    assert float(model.delta_effective()[:, target].abs().max()) == 0.0, \
        "delta survived inside a dropped DMR"
    assert float(model.delta_effective().abs().max()) > 0.0, "no delta left on the surviving DMRs"
    print(f"  Test 10 unreachable DMR dropped     OK  -> D{target} dropped, "
          f"delta zero inside it, {len(data.dropped_dmr_ids)} DMR(s) dropped in total")


# ---------------------------------------------------------------------- #
# Test 11: the reconstruction error function is switchable (huber | mse)
# ---------------------------------------------------------------------- #
def test_recon_loss_mse_switch():
    theta = torch.tensor([[0.8, 0.2], [0.2, 0.6]])
    model = _fixed_model(theta, [0.75, 0.25])
    observed = torch.tensor([[0.60, 0.40]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    # Full coverage weight, so the reducer is a plain mean and the expected value
    # is analytic (the coverage weight is normalised by cfg.coverage_cap).
    weight = torch.full((1, 2), model.cfg.coverage_cap)
    pred = model.predict_control().detach()

    model.cfg.recon_loss = "huber"
    huber = float(model._recon(model.predict_control(), observed, mask, weight).detach())
    model.cfg.recon_loss = "mse"
    mse = float(model._recon(model.predict_control(), observed, mask, weight).detach())
    assert abs(mse - float(((pred - observed) ** 2).mean())) < 1e-7, f"mse switch: got {mse}"
    assert abs(huber - mse) > 0.0, "huber and mse should differ on these residuals"

    # masking an entry excludes it under either function
    mask[0, 1] = False
    mse_masked = float(model._recon(model.predict_control(), observed, mask, weight).detach())
    assert abs(mse_masked - float((pred[0, 0] - observed[0, 0]) ** 2)) < 1e-7, \
        f"masked mse got {mse_masked}, expected the first entry only"
    assert MixtureConfig().recon_loss == "huber", "huber must remain the default"
    print(f"  Test 11 huber/mse switch            OK  -> huber={huber:.6f} mse={mse:.6f}, "
          f"masked mse={mse_masked:.6f} (entry 1 excluded)")


def main() -> int:
    torch.manual_seed(0)
    tests = [
        test_mixture_identity,
        test_simplex,
        test_bounds,
        test_zero_delta_identity,
        test_missing_reference_runs,
        test_synthetic_delta_recovery,
        test_masked_bulk_entry_does_not_enter_loss,
        test_ref_mask_zeroes_delta,
        test_all_ref_masked_locus_excluded,
        test_all_ref_masked_dmr_dropped,
        test_recon_loss_mse_switch,
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
