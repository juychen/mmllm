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

    for s in sample_sizes:
        if s == "all":
            resolved.append(total_lines)
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

    # .var: position-level metadata
    # Use the first region's sequence as reference base for each column.
    ref_seq = str(seqs[0])[:L].upper()
    bases = list(ref_seq)
    motifs = []
    for j in range(L):
        if bases[j] == "C":
            if j + 1 < L and ref_seq[j + 1] == "G":
                motifs.append("CG")
            else:
                motifs.append("CH")
        else:
            motifs.append("")

    var_df = pd.DataFrame({
        "position_idx": range(L),
        "base": bases,
        "motif": motifs,
    })
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

    adata = ad.AnnData(X=X_sp, obs=obs_df, var=var_df)
    adata.layers["targets"] = t_sp
    adata.layers["masks"] = m_sp

    # Gzipped compression — still fast to read back
    adata.write(output_path, compression="gzip")
    density_pct = 100.0 * nnz / max(N * L, 1)
    print(f"[export_prediction_signals_h5ad] {N:,} regions × {L:,} positions, "
          f"{nnz:,} C positions ({density_pct:.1f}% density, ~{100/density_pct:.0f}× sparse vs dense) "
          f"→ {output_path}")


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