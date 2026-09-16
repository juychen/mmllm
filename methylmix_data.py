"""Data alignment and IO for the reference-anchored methylation mixture model.

Implements spec §3 (input formats), §18 stage 0 (alignment) and §15 (outputs).

Conventions enforced here:
  * percentages are converted to [0, 1] (`meth_value` > 1.5 is assumed to be a
    percentage and divided by 100);
  * missing observations are encoded by ``valid = 0`` — they are NEVER turned
    into a 0.0 methylation value (spec §3.1, acceptance criterion 7);
  * a locus that no reference observes is kept as a row but carries no
    information: it is excluded from the reconstruction loss, and a DMR whose
    every locus is in that state is dropped as a unit (spec §3.5);
  * unreferenced single-cell cell types are summed into a single ``OTHER``
    column of the composition prior (spec §2, §6).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylmix_model import OTHER, MixtureBatch, ReferenceAnchoredMethylationMixture

PERCENT_THRESHOLD = 1.5  # above this we assume the column is in percent


# ---------------------------------------------------------------------- #
# reading helpers
# ---------------------------------------------------------------------- #
def read_table(path: str | Path) -> pd.DataFrame:
    """Read a TSV/CSV (optionally gzipped) into a DataFrame."""
    path = str(path)
    sep = "," if path.replace(".gz", "").lower().endswith(".csv") else "\t"
    return pd.read_csv(path, sep=sep, compression="infer")


def _to_fraction(values: pd.Series) -> pd.Series:
    """Convert a methylation column to [0, 1] if it looks like percentages."""
    finite = values.dropna()
    if len(finite) and float(finite.max()) > PERCENT_THRESHOLD:
        return values / 100.0
    return values


def _require_columns(frame: pd.DataFrame, required: list[str], what: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{what}: missing required column(s) {missing}; got {list(frame.columns)}")


# ---------------------------------------------------------------------- #
# stage 0: alignment
# ---------------------------------------------------------------------- #
@dataclass
class AlignedData:
    locus_df: pd.DataFrame                 # locus_id, chrom, start, end, is_dmr, dmr_id
    theta_ref: torch.Tensor                # [C_ref, R]
    ref_cell_types: list[str]
    theta_ref_conc: torch.Tensor           # [C_ref, R]
    batch: MixtureBatch                    # aligned CTRL/DIS tensors + priors
    ctrl_sample_ids: list[str]
    dis_sample_ids: list[str]
    cell_types_all: list[str] = field(default_factory=list)   # components incl. OTHER
    group_priors: dict = field(default_factory=dict)          # bookkeeping
    ref_mask: torch.Tensor | None = None   # [C_ref, R] bool, True = reference observed
    dropped_dmr_ids: list[str] = field(default_factory=list)  # DMRs no reference can inform

    @property
    def n_loci(self) -> int:
        return len(self.locus_df)

    @property
    def locus_valid(self) -> torch.Tensor | None:
        """[R] bool, True where at least one reference observes the locus."""
        if self.ref_mask is None:
            return None
        return self.ref_mask.any(dim=0)


def load_reference_tracks(
    path: str | Path,
) -> tuple[pd.DataFrame, torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """Load reference cell-type methylation (spec §3.1).

    Returns ``(locus_frame, theta_ref [C_ref, R], cell_types, concentration,
    ref_mask [C_ref, R])``.

    A locus is kept as soon as ONE reference observes it; loci no reference sees
    at all are dropped. Entries the mask flags False carry an imputed value (the
    mean of the references observed at that locus) purely so ``theta_ref`` stays
    finite — they carry no information and the model gates them out.

    ``valid = 0`` marks a missing observation and never becomes a 0.0 value, so
    "no signal" stays distinguishable from a true methylation of 0.
    """
    frame = read_table(path)
    _require_columns(
        frame, ["locus_id", "cell_type", "meth_value", "valid"], "reference tracks"
    )
    frame = frame[frame["valid"].astype(bool)].copy()
    frame["meth_value"] = _to_fraction(frame["meth_value"]).clip(0.0, 1.0)

    # Beta concentration when counts are available, else the fixed default.
    if {"modified_reads", "total_reads"} <= set(frame.columns):
        a0, b0 = 0.5, 0.5
        frame["_conc"] = frame["total_reads"].astype(float) + a0 + b0
    elif "coverage" in frame.columns:
        frame["_conc"] = frame["coverage"].astype(float) + 1.0
    else:
        frame["_conc"] = np.nan

    cell_types = sorted(frame["cell_type"].unique().tolist())
    pivot = frame.pivot_table(index="locus_id", columns="cell_type", values="meth_value", aggfunc="mean")
    conc = frame.pivot_table(index="locus_id", columns="cell_type", values="_conc", aggfunc="mean")
    locus_cols = [c for c in ["chrom", "start", "end"] if c in frame.columns]
    locus_frame = (
        frame.drop_duplicates("locus_id").set_index("locus_id")[locus_cols]
        if locus_cols else pd.DataFrame(index=pivot.index)
    )
    observed = pivot[cell_types].notna()
    complete = observed.any(axis=1)
    if not bool(complete.any()):
        raise ValueError(f"reference tracks: no locus is observed by any cell type in {path}")
    pivot = pivot.loc[complete, cell_types]
    observed = observed.loc[complete]
    locus_frame = locus_frame.loc[pivot.index]

    values = pivot.to_numpy(dtype=np.float64)
    mask_np = observed.to_numpy(dtype=bool)
    with np.errstate(invalid="ignore"):
        observed_values = np.where(mask_np, values, np.nan)
        locus_mean = np.nanmean(observed_values, axis=1, keepdims=True)
    global_mean = float(np.nanmean(observed_values))
    fill = np.where(np.isfinite(locus_mean), locus_mean, global_mean)
    values = np.where(mask_np, values, fill)

    theta_ref = torch.tensor(values.T.astype(np.float32))       # [C_ref, R]
    ref_mask = torch.tensor(mask_np.T)                          # [C_ref, R]
    conc_mat = conc.loc[pivot.index, cell_types].to_numpy(dtype=np.float32).T
    conc_t = torch.tensor(np.where(np.isnan(conc_mat), 0.0, conc_mat))
    locus_frame = locus_frame.reset_index()
    return locus_frame, theta_ref, cell_types, conc_t, ref_mask


def load_bulk_tracks(
    path: str | Path,
    locus_ids: pd.Index,
    condition: str,
    min_coverage: float | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one condition's bulk samples aligned to `locus_ids` (spec §3.2).

    Returns ``(sample_ids, y [S, R], mask [S, R] bool, weight [S, R])``. Missing
    rows and invalid entries become mask=0 (never value 0).
    """
    frame = read_table(path)
    _require_columns(frame, ["sample_id", "condition", "locus_id", "meth_value", "valid"], "bulk tracks")
    frame = frame[frame["condition"].astype(str).str.upper() == condition.upper()]
    if frame.empty:
        raise ValueError(f"no bulk samples with condition={condition} in {path}")
    frame = frame.copy()
    frame["meth_value"] = _to_fraction(frame["meth_value"]).clip(0.0, 1.0)
    valid = frame["valid"].astype(bool)
    if "coverage" in frame.columns:
        # Zero coverage is no signal even when the row says valid: keeping such an
        # entry would let a placeholder value into the QC metrics (the loss already
        # dropped it via the weight).
        valid &= frame["coverage"].astype(float) > 0.0
    if min_coverage is not None:
        if "coverage" not in frame.columns:
            raise ValueError("min_coverage requested but bulk tracks have no 'coverage' column")
        valid &= frame["coverage"].astype(float) >= float(min_coverage)
    frame["_valid"] = valid
    frame.loc[~frame["_valid"], "meth_value"] = np.nan

    # Pivot on a constant as well as on the values. The constant pivot gives the
    # full (sample, locus) universe — an invalidated entry still occupies a row —
    # while the values pivot carries NaN exactly where an observation is missing
    # or was invalidated. Reindexing the latter onto the former means ``y.notna()``
    # is the validity mask: an absent entry, a `valid = 0` entry and an entry below
    # min_coverage are all NaN here, so none of them can become a 0.0 value
    # (spec §3.1, acceptance criterion 7). Using the constant pivot as the mask
    # itself would silently accept every invalid entry.
    universe = frame.assign(_one=1.0).pivot_table(
        index="sample_id", columns="locus_id", values="_one", aggfunc="max"
    )
    y = frame.pivot_table(index="sample_id", columns="locus_id", values="meth_value", aggfunc="mean")
    y = y.reindex(index=universe.index, columns=universe.columns)
    weight = None
    if "coverage" in frame.columns:
        weight = frame.pivot_table(index="sample_id", columns="locus_id", values="coverage", aggfunc="mean")
        weight = weight.reindex(index=universe.index, columns=universe.columns)

    sample_ids = universe.index.tolist()
    y = y.reindex(columns=locus_ids)
    mask = y.notna()
    if weight is not None:
        weight = weight.reindex(columns=locus_ids)
    y = y.fillna(0.0)  # value for masked-out entries is irrelevant; mask gates the loss

    y_t = torch.tensor(y.to_numpy(dtype=np.float32))
    mask_t = torch.tensor(mask.to_numpy(dtype=bool))
    weight_t = (
        torch.tensor(weight.fillna(0.0).to_numpy(dtype=np.float32))
        if weight is not None
        else torch.ones_like(y_t)
    )
    return sample_ids, y_t, mask_t, weight_t


def load_composition_prior(
    path: str | Path,
    sample_ids: list[str],
    ref_cell_types: list[str],
    use_other: bool = True,
    sample_groups: dict[str, str] | None = None,
) -> tuple[torch.Tensor, list[str], dict]:
    """Build the composition prior matrix [S, C] from single-cell fractions.

    Unreferenced cell types are aggregated into OTHER (spec §2). Priors may be
    given per sample (`sample_id, cell_type, prior_fraction`) or per group
    (`group_id, ..., cell_type, prior_fraction`), in which case `sample_groups`
    maps each bulk sample to its group.
    """
    frame = read_table(path)
    _require_columns(frame, ["cell_type", "prior_fraction"], "composition prior")
    group_mode = "sample_id" not in frame.columns

    if group_mode:
        if "group_id" not in frame.columns:
            raise ValueError("composition prior needs either 'sample_id' or 'group_id'")
        if sample_groups is None:
            raise ValueError("group-level composition prior requires sample_groups mapping")
        frame = frame.assign(sample_id=frame["group_id"].map(
            {g: s for s, g in sample_groups.items()}  # group -> first sample using it
        ))

    components = list(ref_cell_types) + ([OTHER] if use_other else [])
    rows = []
    for sample in sample_ids:
        sub = frame[frame["sample_id"] == sample]
        values = {c: 0.0 for c in components}
        other_mass = 0.0
        for _, row in sub.iterrows():
            cell = str(row["cell_type"])
            frac = float(row["prior_fraction"])
            if cell in ref_cell_types:
                values[cell] += frac
            else:
                other_mass += frac
        if use_other:
            values[OTHER] = other_mass
        elif other_mass > 0:
            # OTHER disabled: those cells cannot be represented; record the mass.
            pass
        rows.append([values[c] for c in components])

    prior = torch.tensor(np.asarray(rows, dtype=np.float32))
    totals = prior.sum(dim=-1, keepdim=True)
    if bool((totals <= 0).any()):
        bad = [sample_ids[i] for i, t in enumerate(totals.flatten().tolist()) if t <= 0]
        raise ValueError(f"empty composition prior for sample(s) {bad}")
    prior = prior / totals
    info = {"group_mode": bool(group_mode), "unreferenced_mass_mean": float(
        np.mean([r[-1] for r in rows]) if use_other and rows else 0.0
    )}
    return prior, components, info


def load_dmr_annotation(path: str | Path, locus_ids: pd.Index) -> tuple[torch.Tensor, list[str]]:
    """Load `locus_id, is_dmr, dmr_id`; loci absent from the file are non-DMR."""
    frame = read_table(path)
    _require_columns(frame, ["locus_id", "is_dmr"], "DMR annotation")
    frame = frame.drop_duplicates("locus_id").set_index("locus_id")
    is_dmr = frame["is_dmr"].astype(bool).reindex(locus_ids).fillna(False).to_numpy()
    dmr_ids = (
        frame["dmr_id"].reindex(locus_ids).fillna("").astype(str).tolist()
        if "dmr_id" in frame.columns else [""] * len(locus_ids)
    )
    return torch.tensor(is_dmr.astype(bool)), dmr_ids


def drop_fully_masked_dmrs(
    is_dmr: torch.Tensor, dmr_ids: list[str], ref_mask: torch.Tensor
) -> tuple[torch.Tensor, list[str]]:
    """Drop DMRs that no reference can inform (spec §3.1).

    A DMR is dropped only when EVERY one of its loci is missing in EVERY
    reference: the region then carries no reference signal at all, so its delta
    has nothing to fit and must stay out of the penalties and the attribution.
    A locus that is missing in every reference but sits in a DMR with at least one
    observed locus keeps its row and is excluded from the loss individually.

    Returns ``(is_dmr with dropped regions cleared, sorted dropped dmr_ids)``.
    """
    locus_ok_np = ref_mask.any(dim=0).numpy()
    ids = pd.Series(list(dmr_ids), dtype="object")
    selected = (ids.str.len() > 0).to_numpy()
    if not bool(selected.any()):
        return is_dmr, []
    grouped = pd.DataFrame({"dmr_id": ids.to_numpy()[selected], "ok": locus_ok_np[selected]})
    any_ok = grouped.groupby("dmr_id", sort=False)["ok"].any()
    dropped = sorted(any_ok.index[~any_ok].tolist())
    if not dropped:
        return is_dmr, []
    in_dropped = torch.tensor(ids.isin(dropped).to_numpy(), dtype=torch.bool)
    return is_dmr & ~in_dropped, dropped


def build_aligned_data(
    reference_path: str | Path,
    ctrl_path: str | Path,
    dis_path: str | Path,
    prior_path: str | Path,
    dmr_path: str | Path | None = None,
    min_coverage: float | None = None,
    use_other: bool = True,
    sample_groups: dict[str, str] | None = None,
) -> AlignedData:
    """Stage 0: intersect loci across references and both bulk conditions."""
    locus_frame, theta_ref, ref_cell_types, conc, ref_mask = load_reference_tracks(reference_path)
    locus_ids = pd.Index(locus_frame["locus_id"])
    if locus_ids.has_duplicates:
        raise ValueError("duplicate locus_id in reference tracks")

    # The DMR annotation is needed before the reference mask is evaluated: a DMR
    # whose every locus is missing in every reference is dropped as a unit.
    if dmr_path is not None:
        is_dmr, dmr_ids = load_dmr_annotation(dmr_path, locus_ids)
    else:
        is_dmr, dmr_ids = torch.ones(len(locus_frame), dtype=torch.bool), [""] * len(locus_frame)
    is_dmr, dropped_dmr_ids = drop_fully_masked_dmrs(is_dmr, dmr_ids, ref_mask)

    ctrl_ids, y_c, m_c, w_c = load_bulk_tracks(ctrl_path, locus_ids, "CTRL", min_coverage)
    dis_ids, y_d, m_d, w_d = load_bulk_tracks(dis_path, locus_ids, "DIS", min_coverage)

    # Keep loci that at least one sample of each condition can actually see. Loci
    # without reference signal are NOT removed here: they keep their row so the
    # output tables stay aligned, and the model drops them from the loss.
    keep = (m_c.any(dim=0)) & (m_d.any(dim=0))
    if int(keep.sum()) == 0:
        raise ValueError("no loci are observed in both CTRL and DIS tracks")
    locus_frame = locus_frame.loc[keep.numpy()].reset_index(drop=True)
    theta_ref, conc, ref_mask = theta_ref[:, keep], conc[:, keep], ref_mask[:, keep]
    is_dmr = is_dmr[keep]
    dmr_ids = [d for d, k in zip(dmr_ids, keep.tolist()) if k]
    locus_frame["is_dmr"] = is_dmr.numpy()
    locus_frame["dmr_id"] = dmr_ids

    pi0_ctrl, components, info = load_composition_prior(
        prior_path, ctrl_ids, ref_cell_types, use_other, sample_groups
    )
    pi0_dis, _, _ = load_composition_prior(
        prior_path, dis_ids, ref_cell_types, use_other, sample_groups
    )

    batch = MixtureBatch(
        y_ctrl=y_c[:, keep], mask_ctrl=m_c[:, keep], weight_ctrl=w_c[:, keep], pi0_ctrl=pi0_ctrl,
        y_dis=y_d[:, keep], mask_dis=m_d[:, keep], weight_dis=w_d[:, keep], pi0_dis=pi0_dis,
    )
    return AlignedData(
        locus_df=locus_frame,
        theta_ref=theta_ref,
        ref_cell_types=ref_cell_types,
        theta_ref_conc=conc,
        batch=batch,
        ctrl_sample_ids=ctrl_ids,
        dis_sample_ids=dis_ids,
        cell_types_all=components,
        group_priors=info,
        ref_mask=ref_mask,
        dropped_dmr_ids=dropped_dmr_ids,
    )


def init_other_from_control_residual(
    data: AlignedData, eps: float = 1.0e-4
) -> torch.Tensor | None:
    """spec §6: OTHER init = control residual / OTHER fraction, clipped.

    Only entries that are observed (valid, non-zero coverage, and on a locus some
    reference actually sees) enter the control mean. At loci where the control
    says nothing there is no residual to estimate, so OTHER falls back to the
    reference mean weighted by the control prior — the same default the model
    would use if no init were supplied.
    """
    batch = data.batch
    if batch.pi0_ctrl is None or not data.cell_types_all or data.cell_types_all[-1] != OTHER:
        return None
    pi_ctrl = batch.pi0_ctrl
    other_frac = float(pi_ctrl[:, -1].mean())
    if other_frac <= 0:
        return None

    pi_ref_mean = pi_ctrl[:, :-1].mean(dim=0)
    ref_mean = torch.einsum("c,cr->r", pi_ref_mean, data.theta_ref)
    weights = pi_ref_mean / pi_ref_mean.sum().clamp_min(eps)
    default_other = torch.einsum("c,cr->r", weights, data.theta_ref)

    observed = batch.mask_ctrl.bool()
    if batch.weight_ctrl is not None:
        observed = observed & (batch.weight_ctrl > 0)
    if data.ref_mask is not None:
        observed = observed & data.ref_mask.any(dim=0)[None, :]
    covered = observed.sum(dim=0)
    if int(covered.sum()) == 0:
        return None

    y_bar = torch.where(
        covered > 0,
        (batch.y_ctrl * observed).sum(dim=0) / covered.clamp_min(1),
        ref_mean,
    )
    residual = y_bar - ref_mean
    other = (residual / other_frac).clamp(eps, 1.0 - eps)
    return torch.where(covered > 0, other, default_other)


def make_model(data: AlignedData, cfg=None, composition_mode: str = "dirichlet_prior"):
    """Build the model with OTHER initialised from the control residual (spec §6)."""
    from methylmix_model import MixtureConfig

    cfg = cfg or MixtureConfig()
    cfg.composition_mode = composition_mode
    cfg.use_other = bool(data.cell_types_all[-1] == OTHER)
    other_init = init_other_from_control_residual(data, cfg.eps)
    return ReferenceAnchoredMethylationMixture(
        theta_ref=data.theta_ref,
        pi0_ctrl=data.batch.pi0_ctrl,
        pi0_dis=data.batch.pi0_dis,
        other_init=other_init,
        is_dmr=torch.tensor(data.locus_df["is_dmr"].to_numpy()),
        theta_ref_conc=data.theta_ref_conc,
        cfg=cfg,
        cell_type_names=data.ref_cell_types,
        ref_mask=data.ref_mask,
    )


# ---------------------------------------------------------------------- #
# synthetic data (spec §16.7) and unit-test fixtures
# ---------------------------------------------------------------------- #
def make_synthetic_data(
    n_loci: int = 200,
    cell_types: tuple[str, ...] = ("Glut", "GABA", "Astro"),
    n_ctrl: int = 4,
    n_dis: int = 4,
    delta_cell_type: str = "Astro",
    n_dmr: int = 40,
    delta_logit: float = 1.2,
    delta_spec: dict[str, float] | None = None,
    missing_reference: str | None = None,
    seed: int = 0,
    noise: float = 0.01,
    use_other: bool = True,
    ref_masked_loci: dict[str, list[int]] | None = None,
    all_ref_masked_loci: list[int] | None = None,
) -> tuple[AlignedData, dict]:
    """Generate a synthetic benchmark with a KNOWN responsible cell type.

    ``delta_spec`` injects into several cell types at once (e.g.
    ``{"Astro": 1.5, "Oligo": 0.5}``); when it is given it takes precedence over
    ``delta_cell_type`` / ``delta_logit``. All injected deltas are placed on the
    same DMR loci, positive or negative as specified.

    ``ref_masked_loci`` marks individual references as unobserved at given loci
    (``{"Astro": [3, 7]}``) and ``all_ref_masked_loci`` marks loci every reference
    is missing — the latter are dropped from the loss, and a DMR made up entirely
    of them is dropped as a unit. Both default to None (fully observed).

    Returns ``(data, truth)`` where ``truth`` holds the injected delta, the true
    compositions and the responsible cell type(s).
    """
    rng = np.random.default_rng(seed)
    theta = rng.beta(2.0, 2.0, size=(len(cell_types), n_loci)).astype(np.float32)

    if delta_spec is None:
        delta_spec = {delta_cell_type: delta_logit}
    delta_spec = {str(k): float(v) for k, v in delta_spec.items()}
    unknown = [c for c in delta_spec if c not in cell_types]
    if unknown:
        raise ValueError(f"delta_spec cell type(s) {unknown} not in {list(cell_types)}")

    def prior_for(n: int) -> np.ndarray:
        p = rng.dirichlet(np.ones(len(cell_types)) * 3.0, size=n)
        if not use_other:
            return p
        other = rng.uniform(0.02, 0.12, size=(n, 1))
        return np.concatenate([p * (1 - other), other], axis=1)

    pi_ctrl = prior_for(n_ctrl).astype(np.float32)
    pi_dis = prior_for(n_dis).astype(np.float32)
    # Give disease samples a small composition shift so the composition effect is identified.
    pi_dis[:, :-1] = pi_dis[:, :-1] * 0.9
    pi_dis = pi_dis / pi_dis.sum(axis=1, keepdims=True)
    pi_dis = pi_dis.astype(np.float32)

    # OTHER baseline: made harsher/different from the references on purpose.
    theta_other = np.clip(rng.beta(2.0, 5.0, size=n_loci).astype(np.float32), 1e-3, 1 - 1e-3)
    theta_all = np.concatenate([theta, theta_other[None, :]], axis=0) if use_other else theta

    loc = np.zeros(n_loci, dtype=bool)
    loc[: min(n_dmr, n_loci)] = True
    rng.shuffle(loc)
    delta_true = np.zeros_like(theta_all)
    for cell, value in delta_spec.items():
        delta_true[list(cell_types).index(cell), loc] = value

    def mix(pi, theta_use):
        return np.einsum("sc,cr->sr", pi, theta_use)

    y_ctrl = mix(pi_ctrl, theta_all)
    theta_dis = 1.0 / (1.0 + np.exp(-(np.log(np.clip(theta_all, 1e-4, 1 - 1e-4) /
                                                  np.clip(1 - theta_all, 1e-4, 1 - 1e-4)).astype(np.float64)
                                       + delta_true.astype(np.float64))))
    y_dis = mix(pi_dis, theta_dis)
    y_ctrl = np.clip(y_ctrl + rng.normal(0, noise, y_ctrl.shape), 0, 1).astype(np.float32)
    y_dis = np.clip(y_dis + rng.normal(0, noise, y_dis.shape), 0, 1).astype(np.float32)

    # Observations above were generated from the FULL component set, i.e. the
    # bulk still contains the hidden cell type's contribution. Hiding a
    # reference (spec §16.6) therefore means: drop its column/track, move its
    # single-cell fraction into OTHER, and let OTHER absorb its methylome.
    ref_types = list(cell_types)
    theta_ref_rows = [theta[i].copy() for i in range(len(cell_types))]
    theta_other_row = theta_other.copy()
    if missing_reference is not None:
        if missing_reference not in ref_types:
            raise ValueError(f"missing_reference {missing_reference!r} not in {ref_types}")
        idx = ref_types.index(missing_reference)
        hidden = theta_ref_rows.pop(idx)
        ref_types.pop(idx)
        theta_other_row = 0.5 * (theta_other_row + hidden)
        moved = pi_ctrl[:, idx] + pi_dis[:, idx]
        pi_ctrl = np.delete(pi_ctrl, idx, axis=1)
        pi_dis = np.delete(pi_dis, idx, axis=1)
        if use_other:
            pi_ctrl[:, -1] += moved
            pi_dis[:, -1] += moved
        pi_ctrl = pi_ctrl / pi_ctrl.sum(axis=1, keepdims=True)
        pi_dis = pi_dis / pi_dis.sum(axis=1, keepdims=True)
    theta_ref = (
        np.stack(theta_ref_rows, axis=0) if theta_ref_rows
        else np.zeros((0, n_loci), dtype=np.float32)
    )

    # ---- reference observation mask (spec §3.1) ---------------------------
    n_ref = len(ref_types)
    ref_mask_np = np.ones((n_ref, n_loci), dtype=bool)
    for cell, loci in (ref_masked_loci or {}).items():
        if cell not in ref_types:
            raise ValueError(f"ref_masked_loci cell type {cell!r} not in {ref_types}")
        ref_mask_np[ref_types.index(cell), np.asarray(loci, dtype=int)] = False
    if all_ref_masked_loci is not None:
        ref_mask_np[:, np.asarray(all_ref_masked_loci, dtype=int)] = False
    if not ref_mask_np.all():
        # Mirror the loader: masked entries keep a finite placeholder (the locus
        # mean of the observed references) and carry no information. A locus no
        # reference observes falls back to the global mean rather than to 0.0, so
        # a missing value is never mistaken for a true methylation of zero.
        observed_w = ref_mask_np.astype(np.float32)
        covered = observed_w.sum(axis=0)
        locus_mean = (theta_ref * observed_w).sum(axis=0) / np.maximum(covered, 1.0)
        global_mean = float((theta_ref * observed_w).sum() / max(float(observed_w.sum()), 1.0))
        fill = np.where(covered > 0, locus_mean, global_mean)
        theta_ref = np.where(ref_mask_np, theta_ref, fill[None, :]).astype(np.float32)

    locus_df = pd.DataFrame({
        "locus_id": [f"L{i}" for i in range(n_loci)],
        "chrom": "chr1",
        "start": np.arange(n_loci),
        "end": np.arange(n_loci) + 1,
        "is_dmr": loc,
        "dmr_id": [f"D{i}" if loc[i] else "" for i in range(n_loci)],
    })
    ref_mask_t = torch.tensor(ref_mask_np)
    is_dmr_t, dropped_dmr_ids = drop_fully_masked_dmrs(
        torch.tensor(loc), locus_df["dmr_id"].tolist(), ref_mask_t
    )
    locus_df["is_dmr"] = is_dmr_t.numpy()

    n_comp = len(ref_types) + (1 if use_other else 0)
    batch = MixtureBatch(
        y_ctrl=torch.tensor(y_ctrl), mask_ctrl=torch.ones(n_ctrl, n_loci, dtype=torch.bool),
        weight_ctrl=torch.ones(n_ctrl, n_loci),
        pi0_ctrl=torch.tensor(pi_ctrl.astype(np.float32)),
        y_dis=torch.tensor(y_dis), mask_dis=torch.ones(n_dis, n_loci, dtype=torch.bool),
        weight_dis=torch.ones(n_dis, n_loci),
        pi0_dis=torch.tensor(pi_dis.astype(np.float32)),
    )
    data = AlignedData(
        locus_df=locus_df,
        theta_ref=torch.tensor(theta_ref.astype(np.float32)),
        ref_cell_types=ref_types,
        theta_ref_conc=torch.zeros_like(torch.tensor(theta_ref)),
        batch=batch,
        ctrl_sample_ids=[f"ctrl{i}" for i in range(n_ctrl)],
        dis_sample_ids=[f"dis{i}" for i in range(n_dis)],
        cell_types_all=list(ref_types) + ([OTHER] if use_other else []),
        ref_mask=ref_mask_t,
        dropped_dmr_ids=dropped_dmr_ids,
    )
    truth = {
        "delta_true": torch.tensor(delta_true.astype(np.float32)),
        # delta_cell_type keeps the single-injection meaning for older callers;
        # with multiple injections it reports the largest-magnitude one.
        "delta_cell_type": max(delta_spec, key=lambda c: abs(delta_spec[c])),
        "delta_spec": dict(delta_spec),
        "dmr_loci": is_dmr_t,
        "pi_ctrl_true": torch.tensor(pi_ctrl.astype(np.float32)),
        "pi_dis_true": torch.tensor(pi_dis.astype(np.float32)),
        "theta_all_true": torch.tensor(theta_all.astype(np.float32)),
        "ref_mask": ref_mask_t,
        "dropped_dmr_ids": dropped_dmr_ids,
        "all_ref_masked_loci": sorted(set(all_ref_masked_loci or [])),
    }
    return data, truth


def dump_synthetic_files(
    data: AlignedData,
    out_dir: str | Path,
    unreferenced_cell_types: tuple[str, ...] = ("Oligo",),
) -> dict[str, str]:
    """Export a synthetic AlignedData to the spec §3 file formats.

    Used to exercise the loaders (`build_aligned_data`) and to provide example
    input files. Reference values are written as percentages (0-100) on purpose
    so the percentage -> [0,1] conversion is covered; bulk values are written as
    fractions with a coverage column. The OTHER prior mass is written out as one
    or more *unreferenced* cell-type rows, so the loader has to re-aggregate them
    into OTHER (spec §2).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    locus = data.locus_df
    files: dict[str, str] = {}

    # --- reference tracks (percent, valid flag, a few forced-invalid rows) ----
    rows = []
    ref_frac = data.theta_ref.numpy()
    ref_observed = (
        data.ref_mask.numpy() if data.ref_mask is not None
        else np.ones_like(ref_frac, dtype=bool)
    )
    for j, cell in enumerate(data.ref_cell_types):
        for r, lid in enumerate(locus["locus_id"]):
            # honour the mask, and keep dropping ~1% to exercise the valid=0 path
            valid = int(bool(ref_observed[j, r])) and (0 if r % 97 == 3 else 1)
            rows.append({
                "locus_id": lid, "chrom": locus["chrom"][r], "start": int(locus["start"][r]),
                "end": int(locus["end"][r]), "cell_type": cell,
                "meth_value": float(ref_frac[j, r]) * 100.0,
                "valid": valid,
                "coverage": 12 + (r % 7),
            })
    path = out_dir / "reference_tracks.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    files["reference"] = str(path)

    # --- bulk tracks (fractions + coverage) ---------------------------------
    rows = []
    for condition, sample_ids, y, mask, w in (
        ("CTRL", data.ctrl_sample_ids, data.batch.y_ctrl, data.batch.mask_ctrl, data.batch.weight_ctrl),
        ("DIS", data.dis_sample_ids, data.batch.y_dis, data.batch.mask_dis, data.batch.weight_dis),
    ):
        y_np, m_np, w_np = y.numpy(), mask.numpy(), w.numpy()
        for i, sample in enumerate(sample_ids):
            for r, lid in enumerate(locus["locus_id"]):
                rows.append({
                    "sample_id": sample, "condition": condition, "locus_id": lid,
                    "meth_value": float(y_np[i, r]), "valid": int(bool(m_np[i, r])),
                    "coverage": float(w_np[i, r]), "brain_region": "SYN",
                })
    path = out_dir / "bulk_tracks.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    files["bulk"] = str(path)

    # --- composition prior (referenced cells + unreferenced spelled out) -----
    rows = []
    for sample, prior_row in zip(
        data.ctrl_sample_ids + data.dis_sample_ids,
        torch.cat([data.batch.pi0_ctrl, data.batch.pi0_dis], dim=0),
    ):
        p = prior_row.numpy()
        for j, cell in enumerate(data.ref_cell_types):
            rows.append({"sample_id": sample, "cell_type": cell, "prior_fraction": float(p[j])})
        other_mass = float(p[-1]) if data.cell_types_all[-1] == OTHER else 0.0
        if unreferenced_cell_types and other_mass > 0:
            share = other_mass / len(unreferenced_cell_types)
            for cell in unreferenced_cell_types:
                rows.append({"sample_id": sample, "cell_type": cell, "prior_fraction": share})
    path = out_dir / "composition_prior.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    files["prior"] = str(path)

    # --- DMR annotation -----------------------------------------------------
    path = out_dir / "dmr_annotation.tsv"
    locus[["locus_id", "is_dmr", "dmr_id"]].to_csv(path, sep="\t", index=False)
    files["dmr"] = str(path)
    return files


# ---------------------------------------------------------------------- #
# outputs (spec §15)
# ---------------------------------------------------------------------- #
def _write_tsv(frame: pd.DataFrame, path: str | Path, gzip_out: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip" if gzip_out else None)


# The per-locus tables cost one row per locus per component (or per sample), so a
# genome-wide run with tens of millions of loci cannot materialise them. Above
# this many loci ``auto`` writes only the aggregated tables and says why.
AUTO_PER_LOCUS_MAX_LOCI = 200_000


def should_dump_per_locus(policy: str, n_loci: int) -> bool:
    """Decide whether the per-locus output tables are affordable (see above)."""
    if policy == "always":
        return True
    if policy == "never":
        return False
    if policy == "auto":
        return n_loci <= AUTO_PER_LOCUS_MAX_LOCI
    raise ValueError(f"unknown dump_per_locus policy: {policy!r} (auto|always|never)")


def _skipped(name: str, n_loci: int) -> None:
    print(
        f"  [skip] {name}: {n_loci} loci is above the per-locus dump limit "
        f"({AUTO_PER_LOCUS_MAX_LOCI}); rerun with --dump-per-locus always to force it"
    )


def write_outputs(
    out_dir: str | Path,
    model: ReferenceAnchoredMethylationMixture,
    data: AlignedData,
    qc: dict,
    dump_per_locus: str = "auto",
) -> dict[str, str]:
    """Write the spec §15 output files and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    locus_ids = data.locus_df["locus_id"].tolist()
    comps = model.cell_type_names
    dmr_ids = data.locus_df["dmr_id"].tolist()
    region = pd.Series(dmr_ids, dtype="object").replace("", "(non-DMR)")
    per_locus = should_dump_per_locus(dump_per_locus, len(locus_ids))
    files: dict[str, str] = {}

    # reference_missingness.tsv.gz -- per reference x region mask summary (§3.1)
    ref_mask_np = (
        data.ref_mask.numpy()
        if data.ref_mask is not None
        else np.ones((model.n_ref, model.n_loci), dtype=bool)
    )
    recs = []
    for j, cell in enumerate(comps[: model.n_ref]):
        grouped = pd.DataFrame({
            "region_id": region.to_numpy(),
            "observed": ref_mask_np[j],
        })
        agg = grouped.groupby("region_id", sort=False)["observed"].agg(["size", "sum"])
        for region_id, row in agg.iterrows():
            n = int(row["size"])
            n_observed = int(row["sum"])
            recs.append({
                "cell_type": cell, "region_id": region_id,
                "n_loci": n, "n_observed": n_observed, "n_masked": n - n_observed,
                "frac_masked": (n - n_observed) / n if n else 0.0,
            })
    path = out_dir / "reference_missingness.tsv.gz"
    _write_tsv(pd.DataFrame(recs), path)
    files["reference_missingness"] = str(path)

    # dropped_dmrs.tsv -- regions no reference could inform
    region_sizes = region.value_counts()
    recs = [
        {
            "dmr_id": dmr_id,
            "n_loci": int(region_sizes.get(dmr_id, 0)),
            "reason": "all_loci_missing_in_all_references",
        }
        for dmr_id in data.dropped_dmr_ids
    ]
    path = out_dir / "dropped_dmrs.tsv"
    _write_tsv(pd.DataFrame(recs, columns=["dmr_id", "n_loci", "reason"]), path, gzip_out=False)
    files["dropped_dmrs"] = str(path)

    # sample_composition_posterior.tsv
    rows = []
    for condition, sample_ids, prior in (
        ("CTRL", data.ctrl_sample_ids, model.pi0_ctrl),
        ("DIS", data.dis_sample_ids, model.pi0_dis),
    ):
        post = model.composition(condition).detach()
        for i, sample in enumerate(sample_ids):
            for j, cell in enumerate(comps):
                rows.append({
                    "sample_id": sample, "condition": condition, "cell_type": cell,
                    "prior_fraction": float(prior[i, j]),
                    "posterior_fraction": float(post[i, j]),
                })
    path = out_dir / "sample_composition_posterior.tsv"
    _write_tsv(pd.DataFrame(rows), path, gzip_out=False)
    files["sample_composition_posterior"] = str(path)

    # celltype_baseline_methylation.tsv.gz
    if per_locus:
        theta0 = model.theta0().detach().cpu().numpy()
        recs = []
        for j, cell in enumerate(comps):
            is_ref = 1 if cell in data.ref_cell_types else 0
            sd = float(np.nanstd(theta0[j])) if is_ref == 0 else 0.0
            recs += [{"locus_id": lid, "cell_type": cell, "theta0_mean": float(theta0[j, r]),
                      "theta0_sd": sd, "is_reference": is_ref} for r, lid in enumerate(locus_ids)]
        path = out_dir / "celltype_baseline_methylation.tsv.gz"
        _write_tsv(pd.DataFrame(recs), path)
        files["celltype_baseline_methylation"] = str(path)
    else:
        _skipped("celltype_baseline_methylation.tsv.gz", len(locus_ids))

    # celltype_disease_delta.tsv.gz
    att = model.attribution()
    delta = att["delta_effective"].cpu().numpy()
    theta0_np = att["theta0"].cpu().numpy()
    theta_dis_np = att["theta_dis"].cpu().numpy()
    if per_locus:
        recs = []
        for j, cell in enumerate(comps):
            for r, lid in enumerate(locus_ids):
                recs.append({
                    "locus_id": lid, "dmr_id": dmr_ids[r], "cell_type": cell,
                    "delta_logit": float(delta[j, r]),
                    "theta_ctrl": float(theta0_np[j, r]),
                    "theta_disease": float(theta_dis_np[j, r]),
                    "delta_probability": float(theta_dis_np[j, r] - theta0_np[j, r]),
                })
        path = out_dir / "celltype_disease_delta.tsv.gz"
        _write_tsv(pd.DataFrame(recs), path)
        files["celltype_disease_delta"] = str(path)
    else:
        _skipped("celltype_disease_delta.tsv.gz", len(locus_ids))

    # bulk_reconstruction.tsv.gz
    if per_locus:
        n_refs_observed = ref_mask_np.sum(axis=0)
        recs = []
        with torch.no_grad():
            for condition, sample_ids, y, m, pred in (
                ("CTRL", data.ctrl_sample_ids, data.batch.y_ctrl, data.batch.mask_ctrl, model.predict_control()),
                ("DIS", data.dis_sample_ids, data.batch.y_dis, data.batch.mask_dis, model.predict_disease()),
            ):
                for i, sample in enumerate(sample_ids):
                    for r, lid in enumerate(locus_ids):
                        recs.append({
                            "sample_id": sample, "condition": condition, "locus_id": lid,
                            "observed": float(y[i, r]), "predicted": float(pred[i, r]),
                            "residual": float(pred[i, r] - y[i, r]), "valid": bool(m[i, r]),
                            "n_refs_observed": int(n_refs_observed[r]),
                        })
        path = out_dir / "bulk_reconstruction.tsv.gz"
        _write_tsv(pd.DataFrame(recs), path)
        files["bulk_reconstruction"] = str(path)
    else:
        _skipped("bulk_reconstruction.tsv.gz", len(locus_ids))

    # dmr_attribution.tsv.gz -- one row per DMR per component, so it stays affordable
    intrinsic = att["celltype_intrinsic"].cpu().numpy()
    comp_eff = att["composition_effect"].cpu().numpy()
    recs = []
    seen: dict[str, list[int]] = {}
    for r, dmr_id in enumerate(dmr_ids):
        if dmr_id:
            seen.setdefault(dmr_id, []).append(r)
    for dmr_id, loci in seen.items():
        for j, cell in enumerate(comps):
            intrinsic_effect = float(intrinsic[j, loci].sum())
            composition_effect = float(comp_eff[loci].sum())
            total_abs = abs(intrinsic_effect) + abs(composition_effect)
            recs.append({
                "dmr_id": dmr_id, "cell_type": cell,
                "composition_effect": composition_effect,
                "intrinsic_effect": intrinsic_effect,
                "celltype_intrinsic_contribution": intrinsic_effect,
                "abs_contribution_fraction": (abs(intrinsic_effect) / total_abs) if total_abs > 0 else 0.0,
            })
    path = out_dir / "dmr_attribution.tsv.gz"
    _write_tsv(pd.DataFrame(recs), path)
    files["dmr_attribution"] = str(path)

    # model_qc.json
    path = out_dir / "model_qc.json"
    path.write_text(json.dumps(qc, indent=2, default=float))
    files["model_qc"] = str(path)
    return files
