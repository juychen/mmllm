import random
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def resolve_sample_sizes(sample_sizes, args):
    """Resolve ``all`` in sample_sizes to the line count of the input file.

    If any element of *sample_sizes* is the string ``"all"``, it is replaced
    by the number of data rows in the primary input file.  Priority:
    ``--dmr-csv`` > first ``--m5c-bedgraph`` > first ``--hm5c-bedgraph``.
    For ``.csv`` files the header line is subtracted; for BED/bedGraph files
    (which have no header and use ``#`` for comments) the raw line count is
    used.
    """
    resolved = []
    if "all" not in sample_sizes:
        return [int(s) for s in sample_sizes]

    # Pick the best available input file
    input_file = None
    if hasattr(args, "dmr_csv") and args.dmr_csv:
        input_file = args.dmr_csv
    elif hasattr(args, "m5c_bedgraph") and args.m5c_bedgraph:
        paths = args.m5c_bedgraph if isinstance(args.m5c_bedgraph, list) else [args.m5c_bedgraph]
        input_file = paths[0]
    elif hasattr(args, "hm5c_bedgraph") and args.hm5c_bedgraph:
        paths = args.hm5c_bedgraph if isinstance(args.hm5c_bedgraph, list) else [args.hm5c_bedgraph]
        input_file = paths[0]

    if input_file is None:
        raise ValueError(
            "Cannot resolve 'all' sample size: no input file found. "
            "Provide --dmr-csv, --m5c-bedgraph, or --hm5c-bedgraph."
        )

    result = subprocess.run(["wc", "-l", input_file], capture_output=True, text=True, check=True)
    total_lines = int(result.stdout.split()[0])

    # CSV files have a header line; BED/bedGraph files do not
    is_csv = input_file.lower().endswith(".csv") or input_file.lower().endswith(".csv.gz")
    if is_csv:
        total_lines -= 1

    # Multi-dataset runs: `--use-all-input-groups` replicates the DMR table once
    # per input group, so "all" must resolve to (rows-per-group) × (num_groups),
    # otherwise prepare_*() would truncate back to the first group's rows.
    num_groups = 1
    if getattr(args, "use_all_input_groups", False):
        for attr in ("m5c_bedgraph", "hm5c_bedgraph", "atac_bw"):
            v = getattr(args, attr, None)
            if v:
                paths = v if isinstance(v, list) else [v]
                num_groups = max(num_groups, len(paths))
                break

    for s in sample_sizes:
        if s == "all":
            resolved.append(total_lines * num_groups)
        else:
            resolved.append(int(s))
    return resolved


def get_freest_gpu() -> int:
    """Return the index of the GPU with the most free memory (MiB).

    Falls back to GPU 0 if nvidia-smi is unavailable or fails.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        free_mems = [int(line.strip()) for line in output.strip().split("\n") if line.strip()]
        if not free_mems:
            return 0
        return free_mems.index(max(free_mems))
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 0


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def export_prediction_signals(
    output_csv: str,
    region_metadata: pd.DataFrame,
    predictions,
    targets,
    masks,
) -> pd.DataFrame:
    ensure_parent_dir(output_csv)

    prediction_array = np.asarray(predictions)
    target_array = np.asarray(targets)
    mask_array = np.asarray(masks)
    if prediction_array.ndim == 3 and prediction_array.shape[-1] == 1:
        prediction_array = prediction_array[..., 0]
    if target_array.ndim == 3 and target_array.shape[-1] == 1:
        target_array = target_array[..., 0]
    if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
        mask_array = mask_array[..., 0]

    rows = []
    metadata_records = region_metadata.reset_index(drop=True).to_dict("records")
    for region_idx, metadata in enumerate(metadata_records):
        sequence = str(metadata.get("sequence") or "").upper()
        region_start = metadata.get("start_expanded")
        for position_idx in range(prediction_array.shape[1]):
            base = sequence[position_idx] if position_idx < len(sequence) else ""
            genomic_position = int(region_start) + position_idx if pd.notna(region_start) else None
            rows.append(
                {
                    "region_idx": region_idx,
                    "original_idx": metadata.get("original_idx"),
                    "strand_view": metadata.get("strand_view", "+"),
                    "chr": metadata.get("chr"),
                    "start_expanded": metadata.get("start_expanded"),
                    "end_expanded": metadata.get("end_expanded"),
                    "position_idx": position_idx,
                    "genomic_position": genomic_position,
                    "base": base,
                    "predicted_signal": float(prediction_array[region_idx, position_idx]),
                    "true_signal": float(target_array[region_idx, position_idx]),
                    "mask": float(mask_array[region_idx, position_idx]),
                }
            )

    prediction_frame = pd.DataFrame(rows)
    prediction_frame.to_csv(output_csv, index=False)
    return prediction_frame


def export_prediction_signals_h5ad(
    output_path: str,
    region_metadata: pd.DataFrame,
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> None:
    """Export predictions, targets, and masks to sparse .h5ad (AnnData).

    Only C positions (where methylation can occur) are stored as CSC sparse
    matrices; non-C positions are implicitly 0, achieving ~75-80% compression
    vs a dense matrix.

    .X                = predicted signal (CSC float32, non-C → 0)
    .layers['targets'] = ground-truth signal (CSC float32, non-C → 0)
    .layers['masks']   = loss mask (CSC bool)
    .obs              = region-level metadata
    .var              = position-level metadata: position_idx, base (A/C/G/T),
                        motif (CG/CH if base=C, otherwise "")
    """
    import anndata as ad
    from scipy.sparse import csc_matrix

    output_path = str(output_path)
    if not output_path.endswith(".h5ad"):
        output_path += ".h5ad"

    ensure_parent_dir(output_path)

    N, L = predictions.shape
    seqs = region_metadata["sequence"].values

    # Build boolean mask of C positions only (non-C → implicit 0 in sparse)
    seq_chars = np.array([list(str(s)[:L]) for s in seqs])   # (N, L), dtype='<U1'
    c_mask = seq_chars == "C"                                 # (N, L)
    del seq_chars

    rows, cols = np.where(c_mask)   # indices of C positions
    nnz = len(rows)

    if nnz == 0:
        print(f"[export_prediction_signals_h5ad] WARNING: no C positions found → "
              f"{output_path.replace('.h5ad', '.empty.info')}")
        Path(output_path.replace(".h5ad", ".empty.info")).touch()
        return

    pred_vals = np.asarray(predictions)[rows, cols].astype(np.float32)
    tgt_vals = np.asarray(targets)[rows, cols].astype(np.float32)
    msk_vals = np.asarray(masks)[rows, cols].astype(np.int8)

    shape = (N, L)

    # Build sparse matrices (CSC) — non-C positions are implicitly 0
    X_sp = csc_matrix((pred_vals, (rows, cols)), shape=shape, dtype=np.float32)
    t_sp = csc_matrix((tgt_vals, (rows, cols)), shape=shape, dtype=np.float32)
    m_sp = csc_matrix((msk_vals, (rows, cols)), shape=shape, dtype=np.int8)

    # .var: position-level metadata (global index only, no per-region info)
    var_df = pd.DataFrame({"position_idx": range(L)})
    var_df.index = [f"pos_{j}" for j in range(L)]

    # .obs: region metadata (select relevant columns)
    obs_cols = ["chr", "start_expanded", "end_expanded", "strand_view"]
    available = [c for c in obs_cols if c in region_metadata.columns]
    obs_df = region_metadata[available].reset_index(drop=True).copy()
    if "original_idx" in region_metadata.columns:
        obs_df["original_idx"] = region_metadata["original_idx"].values
    # .obs_names = "chr:start-end"
    obs_df.index = [
        f"{row['chr']}:{int(row['start_expanded'])}-{int(row['end_expanded'])}"
        for _, row in obs_df.iterrows()
    ]

    # .varm / .layers: per-region-per-position metadata (region-dependent)
    #   base_encoded: (N, L) sparse int8   → 1=A, 2=C, 3=G, 4=T, 0=other
    #   motif:        (N, L) sparse int8   → 1=CG, 2=CH, 0=non-C
    base_map = {"A": 1, "C": 2, "G": 3, "T": 4}
    base_vals = np.zeros(nnz, dtype=np.int8)
    motif_vals = np.zeros(nnz, dtype=np.int8)
    for i, (r, c) in enumerate(zip(rows, cols)):
        seq = str(seqs[r])[:L].upper()
        base_map_val = base_map.get(seq[c], 0)
        base_vals[i] = base_map_val
        if base_map_val == 2:  # C
            if c + 1 < L and seq[c + 1] == "G":
                motif_vals[i] = 1  # CG
            else:
                motif_vals[i] = 2  # CH

    adata = ad.AnnData(X=X_sp, obs=obs_df, var=var_df)
    adata.layers["targets"] = t_sp
    adata.layers["masks"] = m_sp
    adata.layers["base_encoded"] = csc_matrix(
        (base_vals, (rows, cols)), shape=shape, dtype=np.int8
    )
    adata.layers["motif"] = csc_matrix(
        (motif_vals, (rows, cols)), shape=shape, dtype=np.int8
    )

    # Gzipped compression — still fast to read back
    adata.write(output_path, compression="gzip")
    density_pct = 100.0 * nnz / max(N * L, 1)
    print(f"[export_prediction_signals_h5ad] {N:,} regions × {L:,} positions, "
          f"{nnz:,} C positions ({density_pct:.1f}% density, ~{100/density_pct:.0f}× sparse vs dense) "
          f"→ {output_path}")


def derive_test_h5ad_path(checkpoint_path: str) -> str:
    """Default .h5ad path for test predictions next to a checkpoint.

    ``..._best_80137.pt`` → ``..._test_pred_80137.h5ad``; any other name gets a
    ``_test_pred`` suffix instead of clobbering the checkpoint or a sibling file.
    """
    path = str(checkpoint_path)
    if "_best_" in path:
        return path.replace("_best_", "_test_pred_").replace(".pt", ".h5ad")
    if path.endswith(".pt"):
        return path[: -len(".pt")] + "_test_pred.h5ad"
    return path + "_test_pred.h5ad"


def export_multiquery_predictions_h5ad(
    output_path: str,
    region_metadata: pd.DataFrame,
    base_ids: np.ndarray,
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    track_names: list[str] | None = None,
    x_track: str | None = None,
    positions: str = "cpg",
    uns_extra: dict | None = None,
) -> str:
    """Export multi-query predictions to a single sparse .h5ad (AnnData).

    One row per genomic window, one column per position in the window; only
    predicted/reconstructed positions are stored as CSC sparse matrices
    (everything else is implicitly 0).

    Layout (aligned with :func:`export_prediction_signals_h5ad`):
      .X                          predicted signal of ``x_track`` (default: first query)
      .layers['pred_<track>']     predicted signal, one layer per query track
      .layers['target_<track>']   ground-truth signal, one layer per query track
      .layers['mask']             shared position mask (1 = masked / scored position)
      .layers['base_encoded']     1=A 2=C 3=G 4=T (N is encoded as A, as in the dataset)
      .layers['motif']            1=CG, 2=CH, 0=not C
      .obs                        chr, start_expanded, end_expanded, [strand_view, original_idx]
      .var                        position_idx (0-based offset inside the window)
      .uns                        query_tracks, x_track, positions, mask_fraction, ...

    Parameters
    ----------
    base_ids:
        ``(N, L)`` integer array, A=0 C=1 G=2 T=3. Used to locate the C
        positions (and CpG/CH motif).
    positions:
        Which positions to store:

        * ``"cpg"`` (default) — the C **and** G of every CpG. This is exactly the
          set that ``generate_pretraining_cpg_mask`` samples the shared position
          mask from, so the export reproduces the reported masked metrics.
          Note the forward signal tracks only carry a value on the C, i.e. the
          target at G positions is 0; filter ``base_encoded == 2`` (C) for
          biology-facing analyses.
        * ``"cpg_forward"`` — only the C of forward-strand CpGs (the positions
          the CG bedGraph tracks are actually defined on).
        * ``"c"`` — every C.
    """
    import anndata as ad
    from scipy.sparse import csc_matrix

    output_path = str(output_path)
    if not output_path.endswith(".h5ad"):
        output_path += ".h5ad"
    ensure_parent_dir(output_path)

    def _2d(arr) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"expected a (N, L) or (N, L, 1) array, got shape {arr.shape}")
        return arr

    base_ids = _2d(base_ids)
    N, L = base_ids.shape

    names = list(track_names) if track_names else list(predictions.keys())
    if not names:
        raise ValueError("no query tracks to export")
    missing = [n for n in names if n not in predictions or n not in targets]
    if missing:
        raise ValueError(f"missing predictions/targets for track(s): {missing}")

    pred = {n: _2d(predictions[n]).astype(np.float32) for n in names}
    tgt = {n: _2d(targets[n]).astype(np.float32) for n in names}
    msk = (_2d(mask) > 0.5).astype(np.int8)

    for n in names:
        if pred[n].shape != (N, L) or tgt[n].shape != (N, L):
            raise ValueError(f"track {n!r}: expected ({N}, {L}), got pred {pred[n].shape}, target {tgt[n].shape}")
    if msk.shape != (N, L):
        raise ValueError(f"mask: expected ({N}, {L}), got {msk.shape}")

    # ----- which positions to store -----------------------------------------
    is_c = base_ids == 1
    is_g = base_ids == 2
    right_is_g = np.zeros_like(is_c)
    right_is_g[:, :-1] = is_g[:, 1:]
    left_is_c = np.zeros_like(is_c)
    left_is_c[:, 1:] = is_c[:, :-1]

    if positions == "cpg":
        # C and G of every CpG — exactly the set generate_pretraining_cpg_mask
        # samples from, so the export reproduces the reported masked metrics.
        selected = (is_c & right_is_g) | (is_g & left_is_c)
    elif positions == "cpg_forward":
        selected = is_c & right_is_g
    elif positions == "c":
        selected = is_c
    else:
        raise ValueError(f"unknown positions={positions!r} (expected 'cpg', 'cpg_forward' or 'c')")

    rows, cols = np.nonzero(selected)
    nnz = len(rows)
    if nnz == 0:
        print(f"[export_multiquery_predictions_h5ad] WARNING: no {positions} positions found → "
              f"{output_path.replace('.h5ad', '.empty.info')}")
        Path(output_path.replace(".h5ad", ".empty.info")).touch()
        return output_path

    shape = (N, L)
    x_name = x_track if x_track in pred else names[0]

    # ----- .obs: region-level metadata --------------------------------------
    obs_cols = ["chr", "start_expanded", "end_expanded", "strand_view"]
    available = [c for c in obs_cols if c in region_metadata.columns]
    if len(region_metadata) != N:
        raise ValueError(f"region_metadata has {len(region_metadata)} rows, expected {N}")
    obs_df = region_metadata[available].reset_index(drop=True).copy()
    if "original_idx" in region_metadata.columns:
        obs_df["original_idx"] = region_metadata["original_idx"].values
    obs_df.index = [
        f"{row['chr']}:{int(row['start_expanded'])}-{int(row['end_expanded'])}"
        + (f":{int(row['original_idx'])}" if "original_idx" in obs_df.columns else "")
        for _, row in obs_df.iterrows()
    ]
    if obs_df.index.duplicated().any():
        obs_df.index = pd.Index([f"{name}-{i}" for i, name in enumerate(obs_df.index)])

    # ----- .var: position offset inside the window --------------------------
    var_df = pd.DataFrame({"position_idx": np.arange(L, dtype=np.int32)})
    var_df.index = [f"pos_{j}" for j in range(L)]

    # ----- sparse layers ----------------------------------------------------
    def _csc(values: np.ndarray, dtype) -> "csc_matrix":
        return csc_matrix((values, (rows, cols)), shape=shape, dtype=dtype)

    adata = ad.AnnData(X=_csc(pred[x_name][rows, cols], np.float32), obs=obs_df, var=var_df)
    for n in names:
        adata.layers[f"pred_{n}"] = _csc(pred[n][rows, cols], np.float32)
        adata.layers[f"target_{n}"] = _csc(tgt[n][rows, cols], np.float32)
    adata.layers["mask"] = _csc(msk[rows, cols], np.int8)
    adata.layers["base_encoded"] = _csc((base_ids[rows, cols] + 1).astype(np.int8), np.int8)

    if positions == "c":
        motif_vals = np.full(nnz, 2, dtype=np.int8)   # CH by default
        next_is_g = (cols + 1 < L) & (base_ids[rows, np.minimum(cols + 1, L - 1)] == 2)
        motif_vals[next_is_g] = 1                     # CG
    else:
        motif_vals = np.ones(nnz, dtype=np.int8)      # every stored position is CG
    adata.layers["motif"] = _csc(motif_vals, np.int8)

    # ----- .uns -------------------------------------------------------------
    adata.uns["query_tracks"] = names
    adata.uns["x_track"] = x_name
    adata.uns["positions"] = positions
    for key, value in (uns_extra or {}).items():
        if value is None:
            continue
        if isinstance(value, dict):
            value = {str(k): float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                     for k, v in value.items()}
        elif isinstance(value, (np.floating, np.integer)):
            value = value.item()
        adata.uns[str(key)] = value

    adata.write(output_path, compression="gzip")
    density_pct = 100.0 * nnz / max(N * L, 1)
    print(f"[export_multiquery_predictions_h5ad] {N:,} regions × {L:,} positions, "
          f"{nnz:,} {positions} positions ({density_pct:.2f}% density) | tracks={names} | "
          f"X={x_name} → {output_path}")
    return output_path


def plot_regression_predictions(
    output_path: str,
    predictions,
    targets,
    masks,
    title: str = "Ground Truth vs Prediction",
) -> None:
    ensure_parent_dir(output_path)

    prediction_array = np.asarray(predictions).reshape(-1)
    target_array = np.asarray(targets).reshape(-1)
    mask_array = np.asarray(masks).reshape(-1).astype(bool)

    masked_predictions = prediction_array[mask_array]
    masked_targets = target_array[mask_array]

    if masked_predictions.size == 0:
        raise ValueError("No masked prediction points available for plotting.")

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(masked_targets, masked_predictions, s=8, alpha=0.25, edgecolors="none")

    min_value = float(min(masked_targets.min(), masked_predictions.min()))
    max_value = float(max(masked_targets.max(), masked_predictions.max()))
    axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", linewidth=1.0, color="black")

    if masked_predictions.size > 1:
        slope, intercept = np.polyfit(masked_targets, masked_predictions, 1)
        fit_x = np.linspace(min_value, max_value, 100)
        fit_y = slope * fit_x + intercept
        axis.plot(fit_x, fit_y, linewidth=1.2, color="tab:red")

    axis.set_xlabel("Ground truth 5hmC")
    axis.set_ylabel("Predicted 5hmC")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)