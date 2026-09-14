"""Data alignment and IO for the reference-anchored methylation mixture model.

Implements spec §3 (input formats), §18 stage 0 (alignment) and §15 (outputs).

Conventions enforced here:
  * percentages are converted to [0, 1] (`meth_value` > 1.5 is assumed to be a
    percentage and divided by 100);
  * missing observations are encoded by ``valid = 0`` — they are NEVER turned
    into a 0.0 methylation value (spec §3.1, acceptance criterion 7);
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

    @property
    def n_loci(self) -> int:
        return len(self.locus_df)


def load_reference_tracks(path: str | Path) -> tuple[pd.DataFrame, torch.Tensor, list[str], torch.Tensor]:
    """Load reference cell-type methylation (spec §3.1).

    Returns ``(locus_frame, theta_ref [C_ref, R], cell_types, concentration)`` on
    the loci where EVERY reference has a valid observation.
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
    complete = pivot[cell_types].notna().all(axis=1)
    pivot = pivot.loc[complete, cell_types]
    locus_frame = locus_frame.loc[pivot.index]

    theta_ref = torch.tensor(pivot.to_numpy(dtype=np.float32).T)          # [C_ref, R]
    conc_mat = conc.loc[pivot.index, cell_types].to_numpy(dtype=np.float32).T
    conc_t = torch.tensor(np.where(np.isnan(conc_mat), 0.0, conc_mat))
    locus_frame = locus_frame.reset_index()
    return locus_frame, theta_ref, cell_types, conc_t


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
    if min_coverage is not None:
        if "coverage" not in frame.columns:
            raise ValueError("min_coverage requested but bulk tracks have no 'coverage' column")
        valid &= frame["coverage"].astype(float) >= float(min_coverage)
    frame["_valid"] = valid
    frame.loc[~frame["_valid"], "meth_value"] = np.nan

    y = frame.pivot_table(index="sample_id", columns="locus_id", values="meth_value", aggfunc="mean")
    mask = frame.assign(_one=1.0).pivot_table(
        index="sample_id", columns="locus_id", values="_one", aggfunc="max"
    )
    weight = None
    if "coverage" in frame.columns:
        weight = frame.pivot_table(index="sample_id", columns="locus_id", values="coverage", aggfunc="mean")

    sample_ids = y.index.tolist()
    y = y.reindex(columns=locus_ids)
    mask = mask.reindex(columns=locus_ids).notna() if mask is not None else y.notna()
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
        bad = [sample_ids[i] for i in range(len(sample_ids)) if float(totals[i]) <= 0]
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
    locus_frame, theta_ref, ref_cell_types, conc = load_reference_tracks(reference_path)
    locus_ids = pd.Index(locus_frame["locus_id"])
    if locus_ids.has_duplicates:
        raise ValueError("duplicate locus_id in reference tracks")

    ctrl_ids, y_c, m_c, w_c = load_bulk_tracks(ctrl_path, locus_ids, "CTRL", min_coverage)
    dis_ids, y_d, m_d, w_d = load_bulk_tracks(dis_path, locus_ids, "DIS", min_coverage)

    # Keep loci that at least one sample of each condition can actually see.
    keep = (m_c.any(dim=0)) & (m_d.any(dim=0))
    if int(keep.sum()) == 0:
        raise ValueError("no loci are observed in both CTRL and DIS tracks")
    locus_frame = locus_frame.loc[keep.numpy()].reset_index(drop=True)
    theta_ref, conc = theta_ref[:, keep], conc[:, keep]

    if dmr_path is not None:
        is_dmr, dmr_ids = load_dmr_annotation(dmr_path, pd.Index(locus_frame["locus_id"]))
    else:
        is_dmr, dmr_ids = torch.ones(len(locus_frame), dtype=torch.bool), [""] * len(locus_frame)
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
    )


def init_other_from_control_residual(
    data: AlignedData, eps: float = 1.0e-4
) -> torch.Tensor | None:
    """spec §6: OTHER init = control residual / OTHER fraction, clipped."""
    batch = data.batch
    if batch.pi0_ctrl is None or not data.cell_types_all or data.cell_types_all[-1] != OTHER:
        return None
    pi_ctrl = batch.pi0_ctrl
    other_frac = float(pi_ctrl[:, -1].mean())
    if other_frac <= 0:
        return None
    m = batch.mask_ctrl
    if m.sum() == 0:
        return None
    y_bar = (batch.y_ctrl * m).sum(dim=0) / m.sum(dim=0).clamp_min(1)
    ref_mean = torch.einsum("c,cr->r", pi_ctrl[:, :-1].mean(dim=0), data.theta_ref)
    residual = y_bar - ref_mean
    return (residual / other_frac).clamp(eps, 1.0 - eps)


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
    missing_reference: str | None = None,
    seed: int = 0,
    noise: float = 0.01,
    use_other: bool = True,
) -> tuple[AlignedData, dict]:
    """Generate a synthetic benchmark with a KNOWN responsible cell type.

    Returns ``(data, truth)`` where ``truth`` holds the injected delta, the true
    compositions and the responsible cell type.
    """
    rng = np.random.default_rng(seed)
    theta = rng.beta(2.0, 2.0, size=(len(cell_types), n_loci)).astype(np.float32)

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
    target = list(cell_types).index(delta_cell_type)
    delta_true[target, loc] = delta_logit

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

    locus_df = pd.DataFrame({
        "locus_id": [f"L{i}" for i in range(n_loci)],
        "chrom": "chr1",
        "start": np.arange(n_loci),
        "end": np.arange(n_loci) + 1,
        "is_dmr": loc,
        "dmr_id": [f"D{i}" if loc[i] else "" for i in range(n_loci)],
    })

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
    )
    truth = {
        "delta_true": torch.tensor(delta_true.astype(np.float32)),
        "delta_cell_type": delta_cell_type,
        "dmr_loci": torch.tensor(loc),
        "pi_ctrl_true": torch.tensor(pi_ctrl.astype(np.float32)),
        "pi_dis_true": torch.tensor(pi_dis.astype(np.float32)),
        "theta_all_true": torch.tensor(theta_all.astype(np.float32)),
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
    for j, cell in enumerate(data.ref_cell_types):
        for r, lid in enumerate(locus["locus_id"]):
            valid = 0 if r % 97 == 3 else 1   # drop ~1% to exercise the valid=0 path
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


def write_outputs(
    out_dir: str | Path,
    model: ReferenceAnchoredMethylationMixture,
    data: AlignedData,
    qc: dict,
) -> dict[str, str]:
    """Write the spec §15 output files and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    locus_ids = data.locus_df["locus_id"].tolist()
    comps = model.cell_type_names
    files: dict[str, str] = {}

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

    # celltype_disease_delta.tsv.gz
    att = model.attribution()
    delta = att["delta_effective"].cpu().numpy()
    theta0_np = att["theta0"].cpu().numpy()
    theta_dis_np = att["theta_dis"].cpu().numpy()
    dmr_ids = data.locus_df["dmr_id"].tolist()
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

    # bulk_reconstruction.tsv.gz
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
                    })
    path = out_dir / "bulk_reconstruction.tsv.gz"
    _write_tsv(pd.DataFrame(recs), path)
    files["bulk_reconstruction"] = str(path)

    # dmr_attribution.tsv.gz
    intrinsic = att["celltype_intrinsic"].cpu().numpy()
    comp_eff = att["composition_effect"].cpu().numpy()
    recs = []
    seen = {}
    for r, dmr_id in enumerate(dmr_ids):
        if dmr_id:
            seen.setdefault(dmr_id, []).append(r)
        if not dmr_id:
            continue
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
