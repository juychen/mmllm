from dataclasses import dataclass
import threading

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import pysam
import torch
import torch.nn.functional as F


@dataclass
class PreparedExperimentData:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    usable_dmrs: int
    seq_len: int
    post_filter_len: int
    train_regions: int
    val_regions: int
    non_overlap_groups: int
    val_region_metadata: pd.DataFrame


TRACK_MODALITIES = {"5mc", "5hmc", "atac"}
CONTEXT_MODALITIES = {"sequence", *TRACK_MODALITIES}
BASE_COMPLEMENT_INDEX = torch.tensor([3, 2, 1, 0], dtype=torch.long)
DNA_COMPLEMENT_TABLE = str.maketrans("ACGTN", "TGCAN")


def ensure_path_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def add_clip_at_zero_argument(parser):
    parser.add_argument(
        "--clip-at-zero",
        dest="clip_at_zero",
        action="store_true",
        help="Clamp negative 5mC, 5hmC, and ATAC input values to zero.",
    )
    return parser


def get_sequence(chrom: str, start: int, end: int, genome: pyfaidx.Fasta) -> str:
    return genome[chrom][start - 1 : end].seq


def fast_tabix_to_track(tbx: pysam.TabixFile, chrom: str, start_1based: int, end_1based: int) -> np.ndarray:
    region_start_0based = int(start_1based) - 1
    region_end_0based = int(end_1based)
    resolved_chrom = resolve_tabix_chrom(tbx, chrom)
    data = [line.split("\t") for line in tbx.fetch(resolved_chrom, region_start_0based, region_end_0based)]
    track_length = region_end_0based - region_start_0based
    if not data:
        return np.zeros(track_length, dtype=np.float32)

    starts = np.array([int(x[1]) for x in data], dtype=np.int64) - region_start_0based
    ends = np.array([int(x[2]) for x in data], dtype=np.int64) - region_start_0based
    vals = np.array([float(x[3]) for x in data], dtype=np.float32)

    starts = np.clip(starts, 0, track_length)
    ends = np.clip(ends, 0, track_length)
    track = np.zeros(track_length, dtype=np.float32)
    for start, end, value in zip(starts, ends, vals):
        if start < end:
            track[start:end] = value
    return track


def _detect_track_format(path: str) -> str:
    """Detect whether a track file is bigWig or bedGraph based on extension."""
    lower = path.lower()
    if lower.endswith(".bw") or lower.endswith(".bigwig"):
        return "bigwig"
    return "bedgraph"


def _open_track_handle(path: str, fmt: str):
    """Open a track file handle. ``fmt`` is 'bigwig' or 'bedgraph'."""
    if fmt == "bigwig":
        return pyBigWig.open(path)
    return pysam.TabixFile(path)


# Per-handle cache: maps requested chrom -> chrom name actually present in the
# tabix / bigWig index.  Many bedGraphs / bigWigs use bare ``"7"`` while the
# rest of the code asks for ``"chr7"``; others do the opposite.  Rather than
# guess, we probe the index the first time we see a chromosome and remember
# the correct spelling for the lifetime of this handle.
_TABIX_CHROM_CACHE: dict[int, dict[str, str]] = {}


def _handle_id(handle) -> int:
    return id(handle)


def resolve_tabix_chrom(handle, chrom: str) -> str:
    """Return the chromosome spelling that ``handle`` actually contains.

    Tries the requested name first, then the alternative (``chr7`` ↔ ``7``).
    Caches the per-handle mapping so subsequent lookups are O(1).
    Falls back to the requested spelling if no match is found (let tabix raise
    a clear error).
    """
    cache = _TABIX_CHROM_CACHE.setdefault(_handle_id(handle), {})
    if chrom in cache:
        return cache[chrom]

    contigs = None
    if hasattr(handle, "contigs"):
        try:
            contigs_attr = handle.contigs
            # pyBigWig exposes ``contigs`` as a method that returns a dict-like
            # object; ``chroms`` (also callable) returns the same.  pysam's
            # ``TabixFile`` exposes ``contigs`` as a dict directly.
            if callable(contigs_attr):
                contigs_attr = contigs_attr()
            contigs = contigs_attr
        except Exception:
            contigs = None
    if contigs is None and hasattr(handle, "chroms"):
        try:
            contigs_attr = handle.chroms
            if callable(contigs_attr):
                contigs_attr = contigs_attr()
            contigs = contigs_attr
        except Exception:
            contigs = None

    if contigs is not None:
        candidates = [chrom]
        if chrom.lower().startswith("chr"):
            candidates.append(chrom[3:])
        else:
            candidates.append("chr" + chrom)
        for cand in candidates:
            if cand in contigs:
                cache[chrom] = cand
                return cand

    # Could not determine — let the underlying call surface the original error.
    cache[chrom] = chrom
    return chrom


def read_track_region(
    handle,
    fmt: str,
    chrom: str,
    start: int,
    end: int,
    clip_at_zero: bool = False,
) -> np.ndarray:
    """Read a track region as a 1D numpy array. ``fmt`` is 'bigwig' or 'bedgraph'."""
    resolved_chrom = resolve_tabix_chrom(handle, chrom)
    if fmt == "bigwig":
        values = np.nan_to_num(handle.values(resolved_chrom, start, end + 1), nan=0.0)
    else:
        values = fast_tabix_to_track(handle, resolved_chrom, start, end)
    if clip_at_zero:
        values = np.maximum(values, 0.0)
    return values


# def find_cpg_candidate_positions(base_ids: torch.Tensor) -> torch.Tensor:
#     is_c = base_ids == 1
#     is_g = base_ids == 2
#     right_is_g = torch.zeros_like(is_g)
#     right_is_g[:, :-1] = is_g[:, 1:]
#     right_is_c = torch.zeros_like(is_c)
#     right_is_c[:, :-1] = is_c[:, 1:]
#     return (is_c & right_is_g) | (is_g & right_is_c)
def find_cpg_candidate_positions(base_ids: torch.Tensor) -> torch.Tensor:
    is_c = base_ids == 1
    is_g = base_ids == 2
    right_is_g = torch.zeros_like(is_g)
    right_is_g[:, :-1] = is_g[:, 1:]
    left_is_c = torch.zeros_like(is_c)
    left_is_c[:, 1:] = is_c[:, :-1]
    return (is_c & right_is_g) | (is_g & left_is_c)

def find_forward_cpg_positions(base_ids: torch.Tensor) -> torch.Tensor:
    is_c = base_ids == 1
    is_g = base_ids == 2
    right_is_g = torch.zeros_like(is_g)
    right_is_g[:, :-1] = is_g[:, 1:]
    return is_c & right_is_g


def find_forward_ch_positions(base_ids: torch.Tensor) -> torch.Tensor:
    """Return C positions followed by a non-G base (CH, not CpG)."""
    is_c = base_ids == 1
    is_g = base_ids == 2
    has_right_base = torch.zeros_like(is_c)
    has_right_base[:, :-1] = True
    right_is_g = torch.zeros_like(is_g)
    right_is_g[:, :-1] = is_g[:, 1:]
    return is_c & has_right_base & ~right_is_g


def scale_atac_tensor(atac_tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return atac_tensor
    if mode == "minmax":
        atac_min = atac_tensor.amin(dim=1, keepdim=True)
        atac_max = atac_tensor.amax(dim=1, keepdim=True)
        atac_range = (atac_max - atac_min).clamp_min(1e-6)
        return (atac_tensor - atac_min) / atac_range
    raise ValueError(f"Unknown ATAC scaling mode: {mode}")


def normalize_sequence(sequence) -> str:
    if isinstance(sequence, str):
        return sequence.upper()
    if isinstance(sequence, (list, tuple, np.ndarray)):
        return "".join(sequence).upper()
    raise TypeError(f"Unsupported sequence type: {type(sequence)}")


def reverse_complement_sequence(sequence: str) -> str:
    return normalize_sequence(sequence).translate(DNA_COMPLEMENT_TABLE)[::-1]


def sequence_to_base_ids(sequence, seq_len: int, base_to_index: dict[str, int]) -> torch.Tensor:
    sequence_str = normalize_sequence(sequence)
    base_ids = torch.zeros(seq_len, dtype=torch.long)
    for pos, base in enumerate(sequence_str[:seq_len]):
        base_ids[pos] = base_to_index.get(base, 0)
    return base_ids


def assign_non_overlapping_groups(region_frame: pd.DataFrame, chrom_col: str, start_col: str, end_col: str) -> pd.DataFrame:
    ordered = region_frame.sort_values([chrom_col, start_col, end_col]).copy()
    group_ids = []
    current_group = -1
    current_chrom = None
    current_end = -1
    for row in ordered.itertuples(index=False):
        row_chrom = getattr(row, chrom_col)
        row_start = int(getattr(row, start_col))
        row_end = int(getattr(row, end_col))
        if row_chrom != current_chrom or row_start > current_end:
            current_group += 1
            current_chrom = row_chrom
            current_end = row_end
        else:
            current_end = max(current_end, row_end)
        group_ids.append(current_group)
    ordered["overlap_group"] = group_ids
    return ordered.sort_values("original_idx").reset_index(drop=True)


def write_train_val_beds(
    split_regions_df: pd.DataFrame,
    train_indices: list[int],
    val_indices: list[int],
    output_dir: "str | os.PathLike | None" = None,
    timestamp: str = "",
    tag: str = "",
) -> tuple["Path | None", "Path | None"]:
    """Write train / val region BED files to ``output_dir``.

    Writes 5 columns: chr / start_expanded / end_expanded / original_idx / overlap_group.
    Returns the (train_bed_path, val_bed_path) tuple.  If ``output_dir`` is None,
    nothing is written and (None, None) is returned.
    """
    if output_dir is None:
        return None, None
    from pathlib import Path
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bed_cols = ["chr", "start_expanded", "end_expanded", "original_idx", "overlap_group"]
    suffix = f"_{tag}" if tag else ""
    ts = f"{timestamp}_" if timestamp else ""
    train_path = out_dir / f"{ts}train_regions{suffix}.bed"
    val_path = out_dir / f"{ts}val_regions{suffix}.bed"
    split_regions_df.iloc[train_indices][bed_cols].to_csv(
        train_path, sep="\t", index=False, header=False,
    )
    split_regions_df.iloc[val_indices][bed_cols].to_csv(
        val_path, sep="\t", index=False, header=False,
    )
    print(f"  train regions : {len(train_indices)}  -> {train_path}")
    print(f"  val regions   : {len(val_indices)}  -> {val_path}")
    return train_path, val_path


def _chromosome_sort_key(chrom_value) -> tuple[int, int | str]:
    chrom_str = str(chrom_value)
    chrom_body = chrom_str[3:] if chrom_str.lower().startswith("chr") else chrom_str
    chrom_body_lower = chrom_body.lower()
    if chrom_body_lower.isdigit():
        return 0, int(chrom_body_lower)
    special_ranks = {"x": 23, "y": 24, "m": 25, "mt": 25}
    if chrom_body_lower in special_ranks:
        return 1, special_ranks[chrom_body_lower]
    return 2, chrom_body_lower


def reorder_regions_by_genomic_position(
    region_frame: pd.DataFrame,
    seqs: list,
    mcg_tracks: list,
    hmcg_tracks: list,
    atac_tracks: list,
) -> tuple[pd.DataFrame, list, list, list, list]:
    ordered = region_frame.copy()
    chrom_keys = ordered["chr"].map(_chromosome_sort_key)
    ordered["_chrom_sort_bucket"] = chrom_keys.map(lambda key: key[0])
    ordered["_chrom_sort_value"] = chrom_keys.map(lambda key: key[1])
    ordered["_original_order"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(
        ["_chrom_sort_bucket", "_chrom_sort_value", "start_expanded", "end_expanded", "input_group", "_original_order"],
        kind="mergesort",
    )
    order = ordered["_original_order"].to_numpy()
    ordered = ordered.drop(columns=["_chrom_sort_bucket", "_chrom_sort_value", "_original_order"]).reset_index(drop=True)

    def reorder_payload(values: list) -> list:
        if not values:
            return values
        return [values[idx] for idx in order]

    return ordered, reorder_payload(seqs), reorder_payload(mcg_tracks), reorder_payload(hmcg_tracks), reorder_payload(atac_tracks)


def _read_dmr_file(path: str) -> pd.DataFrame:
    """Read a DMR file in either CSV (with chr/start/end/length/center columns)
    or BED format (chr, start, end as first three columns, plus optional name/score/strand).

    For BED files, length and center are computed from start/end.
    """
    path_lower = path.lower()
    is_bed = path_lower.endswith(".bed") or path_lower.endswith(".bed.gz")

    if is_bed:
        # BED: chr, start(0-based), end(1-based), [name, score, strand, ...]
        bed_cols = ["chr", "start", "end"]
        df = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0, 1, 2], names=bed_cols)
        # BED start is 0-based, end is 1-based exclusive → convert to 1-based inclusive
        df["start"] = df["start"].astype(int) + 1
        df["end"] = df["end"].astype(int)
        df["length"] = df["end"] - df["start"] + 1
        df["center"] = (df["start"] + df["end"]) // 2
    else:
        df = pd.read_csv(path)
    return df


def load_data(args, lazy: bool = False):
    df_dmr = _read_dmr_file(args.dmr_csv)
    # Early truncation: if sample_sizes is specified, only load up to the max needed rows.
    # This avoids loading the entire DMR file (which can be hundreds of thousands of
    # regions) into memory before the sample-size filter is applied downstream.
    if hasattr(args, "sample_sizes") and args.sample_sizes:
        max_needed = max(args.sample_sizes)
        if len(df_dmr) > max_needed:
            df_dmr = df_dmr.iloc[:max_needed].copy()
    target_length = args.target_length
    half_window = target_length // 2
    df_dmr["start_expanded"] = df_dmr["start"]
    df_dmr["end_expanded"] = df_dmr["end"]
    short_mask = df_dmr["length"] < target_length
    df_dmr.loc[short_mask, "start_expanded"] = df_dmr.loc[short_mask, "center"] - half_window
    df_dmr.loc[short_mask, "end_expanded"] = df_dmr.loc[short_mask, "center"] + half_window - 1

    # In lazy mode, skip loading sequences and tracks — they will be fetched
    # on-the-fly by LazyM5cSequenceAtacDataset.  Return minimal placeholder lists
    # long enough to pass downstream length checks in prepare_*().
    if lazy:
        num_rows = len(df_dmr)
        dummy_seq = "A" * target_length
        dummy_track = np.zeros(target_length, dtype=np.float32)
        return (
            df_dmr,
            [dummy_seq] * num_rows,
            [dummy_track] * num_rows,   # mcg (5mC)
            [dummy_track] * num_rows,   # hmcg (5hmC)
            [dummy_track] * num_rows,   # atac
        )

    genome = pyfaidx.Fasta(args.genome_fasta)
    hm5c_paths = ensure_path_list(getattr(args, "hm5c_bedgraph", None))
    m5c_paths = ensure_path_list(getattr(args, "m5c_bedgraph", None))
    atac_paths = ensure_path_list(getattr(args, "atac_bw", None))

    requested_track_modalities = set()
    if hasattr(args, "input_modality"):
        requested_track_modalities.add(args.input_modality)
    if hasattr(args, "target_modality"):
        requested_track_modalities.add(args.target_modality)
    if hasattr(args, "context_modalities"):
        requested_track_modalities.update(modality for modality in args.context_modalities if modality in TRACK_MODALITIES)

    should_load_5mc = bool(m5c_paths) and (
        getattr(args, "use_m5c", False) or "5mc" in requested_track_modalities or 'multi' in requested_track_modalities
    )

    use_all_input_groups = getattr(args, "use_all_input_groups", False)
    num_groups = max(len(hm5c_paths), len(atac_paths), len(m5c_paths) if should_load_5mc else 0)
    if num_groups == 0:
        raise ValueError("No input track paths were provided. Check --hm5c-bedgraph/--m5c-bedgraph/--atac-bw.")

    if not use_all_input_groups:
        hm5c_paths = hm5c_paths[:1]
        atac_paths = atac_paths[:1]
        if should_load_5mc:
            m5c_paths = m5c_paths[:1]
        num_groups = 1
    else:
        if len(hm5c_paths) != len(atac_paths):
            raise ValueError(
                "When --use-all-input-groups is enabled, --hm5c-bedgraph and --atac-bw must have the same number of paths."
            )
        if should_load_5mc and len(m5c_paths) != len(hm5c_paths):
            raise ValueError(
                "When --use-all-input-groups is enabled, --m5c-bedgraph must have the same number of paths as --hm5c-bedgraph."
            )

    # Detect formats once (cached — no per-region overhead)
    hm5c_formats = [_detect_track_format(p) for p in hm5c_paths]
    m5c_formats = [_detect_track_format(p) for p in m5c_paths] if should_load_5mc else []
    atac_formats = [_detect_track_format(p) for p in atac_paths]
    clip_at_zero = getattr(args, "clip_at_zero", getattr(args, "clip_5hmc_at_zero", False))

    tbx_5hmc_list = [_open_track_handle(p, f) for p, f in zip(hm5c_paths, hm5c_formats)]
    tbx_5mc_list = [_open_track_handle(p, f) for p, f in zip(m5c_paths, m5c_formats)] if should_load_5mc else []
    atac_bw_list = [_open_track_handle(p, f) for p, f in zip(atac_paths, atac_formats)]

    seqs = []
    mcg_tracks = []
    hmcg_tracks = []
    atac_tracks = []
    dmr_frames = []
    for group_idx in range(num_groups):
        group_df = df_dmr.copy()
        group_df["input_group"] = group_idx
        group_df["hm5c_bedgraph_path"] = hm5c_paths[group_idx]
        group_df["atac_bw_path"] = atac_paths[group_idx]
        if should_load_5mc:
            group_df["m5c_bedgraph_path"] = m5c_paths[group_idx]
        dmr_frames.append(group_df)

        tbx_5hmc = tbx_5hmc_list[group_idx]
        tbx_5mc = tbx_5mc_list[group_idx] if should_load_5mc else None
        atac_bw = atac_bw_list[group_idx]
        hm5c_fmt = hm5c_formats[group_idx]
        m5c_fmt = m5c_formats[group_idx] if should_load_5mc else None
        atac_fmt = atac_formats[group_idx]
        for _, row in df_dmr.iterrows():
            # Normalize: strip existing "chr" prefix if present, then re-add consistently
            chrom_name = str(row["chr"]).removeprefix("chr")
            chrom = "chr" + chrom_name
            start = int(row["start_expanded"])
            end = int(row["end_expanded"])
            seqs.append(get_sequence(chrom, start, end, genome))
            if tbx_5mc is not None:
                mcg_tracks.append(
                    read_track_region(
                        tbx_5mc,
                        m5c_fmt,
                        chrom,
                        start,
                        end,
                        clip_at_zero=clip_at_zero,
                    )
                )
            hmcg_tracks.append(
                read_track_region(
                    tbx_5hmc,
                    hm5c_fmt,
                    chrom,
                    start,
                    end,
                    clip_at_zero=clip_at_zero,
                )
            )
            atac_tracks.append(
                read_track_region(
                    atac_bw,
                    atac_fmt,
                    chrom,
                    start,
                    end + 1,
                    clip_at_zero=clip_at_zero,
                )
            )

    combined_df_dmr = pd.concat(dmr_frames, ignore_index=True)
    if use_all_input_groups:
        combined_df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = reorder_regions_by_genomic_position(
            combined_df_dmr,
            seqs,
            mcg_tracks,
            hmcg_tracks,
            atac_tracks,
        )
    return combined_df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks


def get_track_arrays(args, mcg_tracks, hmcg_tracks, atac_tracks, usable_dmrs: int, seq_len: int) -> dict[str, np.ndarray]:
    track_arrays = {
        "5hmc": np.stack([np.asarray(hmcg_tracks[idx][:seq_len], dtype=np.float32) for idx in range(usable_dmrs)]),
        "atac": np.stack([np.asarray(atac_tracks[idx][:seq_len], dtype=np.float32) for idx in range(usable_dmrs)]),
    }
    if mcg_tracks:
        track_arrays["5mc"] = np.stack([np.asarray(mcg_tracks[idx][:seq_len], dtype=np.float32) for idx in range(usable_dmrs)])
    return track_arrays


def tensorize_track_modality(modality: str, track_arrays: dict[str, np.ndarray], args) -> torch.Tensor:
    if modality not in TRACK_MODALITIES:
        raise ValueError(f"Unknown track modality: {modality}")
    if modality not in track_arrays:
        raise ValueError(f"Requested modality '{modality}' is unavailable with the current inputs.")
    track_tensor = torch.tensor(track_arrays[modality], dtype=torch.float32).unsqueeze(-1)
    if modality == "atac":
        return scale_atac_tensor(track_tensor, args.atac_scaling)
    return track_tensor


def build_sequence_tensor(seqs, usable_dmrs: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
    base_ids_tensor = torch.stack([
        sequence_to_base_ids(seqs[idx], seq_len, base_to_index) for idx in range(usable_dmrs)
    ])
    sequence_onehot = F.one_hot(base_ids_tensor, num_classes=4).float()
    return base_ids_tensor, sequence_onehot


def reverse_complement_sequence_tensor(sequence_tensor: torch.Tensor) -> torch.Tensor:
    complement_index = BASE_COMPLEMENT_INDEX.to(sequence_tensor.device)
    complemented = sequence_tensor.index_select(dim=-1, index=complement_index)
    return torch.flip(complemented, dims=[1])


def generate_pretraining_cpg_mask(
    base_ids_tensor: torch.Tensor,
    mask_fraction: float = 0.15,
    seed: int | None = None,
) -> torch.Tensor:
    """
    For each sample independently, find all CpG positions and randomly select
    mask_fraction of them to be masked. Returns a binary mask of shape (N, seq_len, 1).
    """
    cpg_positions = find_cpg_candidate_positions(base_ids_tensor)  # (N, seq_len) bool
    cpg_mask = torch.zeros_like(cpg_positions, dtype=torch.float32)

    rng = np.random.RandomState(seed)
    for i in range(cpg_positions.shape[0]):
        cpg_indices = torch.nonzero(cpg_positions[i], as_tuple=False).squeeze(-1)
        if cpg_indices.numel() == 0:
            continue
        num_mask = max(1, int(cpg_indices.numel() * mask_fraction))
        selected = rng.choice(cpg_indices.numpy(), size=num_mask, replace=False)
        cpg_mask[i, selected] = 1.0

    return cpg_mask.unsqueeze(-1)  # (N, seq_len, 1)


def apply_mask_to_track(track_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
    """Zero out track values at masked positions. mask_tensor: 1.0 = masked."""
    return track_tensor * (1.0 - mask_tensor)


def build_context_tensor(context_modalities: list[str], sequence_tensor: torch.Tensor, track_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    context_parts = []
    for modality in context_modalities:
        if modality == "sequence":
            context_parts.append(sequence_tensor)
            continue
        if modality not in track_tensors:
            raise ValueError(f"Context modality '{modality}' is unavailable with the current inputs.")
        context_parts.append(track_tensors[modality])
    if not context_parts:
        raise ValueError("At least one context modality must be enabled.")
    return torch.cat(context_parts, dim=-1)


def resolve_loss_mask(mask_mode: str, base_ids_tensor: torch.Tensor) -> torch.Tensor:
    if mask_mode == "cpg_both":
        mask = find_cpg_candidate_positions(base_ids_tensor)
    elif mask_mode == "cpg_forward":
        mask = find_forward_cpg_positions(base_ids_tensor)
    elif mask_mode == "ch_only":
        mask = find_forward_ch_positions(base_ids_tensor)
    elif mask_mode == "c_only":
        # Only cytosine positions are eligible.
        mask = base_ids_tensor == 1
    elif mask_mode == "all":
        mask = torch.ones_like(base_ids_tensor, dtype=torch.bool)
    else:
        raise ValueError(f"Unknown mask mode: {mask_mode}")
    return mask.unsqueeze(-1).float()


def augment_with_reverse_complement(
    query_tensor: torch.Tensor,
    context_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    loss_mask: torch.Tensor,
    base_ids_tensor: torch.Tensor,
    sequence_tensor: torch.Tensor,
    region_metadata: pd.DataFrame,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    if not getattr(args, "augment_reverse_complement", False):
        metadata = region_metadata.copy().reset_index(drop=True)
        metadata["strand_view"] = "+"
        return query_tensor, context_tensor, target_tensor, loss_mask, metadata

    rc_query_tensor = torch.flip(query_tensor, dims=[1])
    rc_target_tensor = torch.flip(target_tensor, dims=[1])
    rc_context_tensor = torch.flip(context_tensor, dims=[1])

    if "sequence" in args.context_modalities:
        sequence_offset = 0
        for modality in args.context_modalities:
            if modality == "sequence":
                break
            sequence_offset += 1
        rc_sequence_tensor = reverse_complement_sequence_tensor(sequence_tensor)
        rc_context_tensor = rc_context_tensor.clone()
        rc_context_tensor[:, :, sequence_offset : sequence_offset + 4] = rc_sequence_tensor
    else:
        rc_sequence_tensor = torch.flip(sequence_tensor, dims=[1])

    rc_base_ids_tensor = torch.argmax(rc_sequence_tensor, dim=-1)
    rc_loss_mask = resolve_loss_mask(args.mask_mode, rc_base_ids_tensor)
    if "multi" in args.target_modality:
        rc_loss_mask = rc_loss_mask.repeat(1, 1, loss_mask.shape[-1])

    forward_metadata = region_metadata.copy().reset_index(drop=True)
    forward_metadata["strand_view"] = "+"
    rc_metadata = region_metadata.copy().reset_index(drop=True)
    rc_metadata["strand_view"] = "-"
    rc_metadata["sequence"] = rc_metadata["sequence"].map(reverse_complement_sequence)
    augmented_metadata = pd.concat([forward_metadata, rc_metadata], ignore_index=True)

    return (
        torch.cat([query_tensor, rc_query_tensor], dim=0),
        torch.cat([context_tensor, rc_context_tensor], dim=0),
        torch.cat([target_tensor, rc_target_tensor], dim=0),
        torch.cat([loss_mask, rc_loss_mask], dim=0),
        augmented_metadata,
    )


def prepare_modality_experiment_data(
    num_dmrs: int,
    args,
    df_dmr,
    seqs,
    mcg_tracks,
    hmcg_tracks,
    atac_tracks,
) -> PreparedExperimentData:
    requested_track_modalities = {args.input_modality, args.target_modality, *args.context_modalities}
    if "5mc" in requested_track_modalities and not mcg_tracks:
        raise ValueError("Requested modality '5mc' but no 5mC track was loaded. Set --m5c-bedgraph to a valid path.")

    track_lengths = [len(hmcg_tracks[0]), len(atac_tracks[0]), len(seqs[0])]
    usable_counts = [num_dmrs, len(df_dmr), len(seqs), len(hmcg_tracks), len(atac_tracks)]
    if mcg_tracks:
        track_lengths.append(len(mcg_tracks[0]))
        usable_counts.append(len(mcg_tracks))

    usable_dmrs = min(usable_counts)
    seq_len = min(track_lengths)
    post_filter_len = min(seq_len, 4)

    track_arrays = get_track_arrays(args, mcg_tracks, hmcg_tracks, atac_tracks, usable_dmrs, seq_len)
    base_ids_tensor, sequence_onehot = build_sequence_tensor(seqs, usable_dmrs, seq_len)
    track_tensors = {
        modality: tensorize_track_modality(modality, track_arrays, args) for modality in track_arrays
    }

    query_tensor = tensorize_track_modality(args.input_modality, track_arrays, args)
    target_tensor = tensorize_track_modality(args.target_modality, track_arrays, args)
    context_tensor = build_context_tensor(args.context_modalities, sequence_onehot, track_tensors)
    loss_mask = resolve_loss_mask(args.mask_mode, base_ids_tensor)

    split_regions_df = df_dmr.iloc[:usable_dmrs].copy().reset_index().rename(columns={"index": "original_idx"})
    split_regions_df["chr"] = split_regions_df["chr"].astype(str)
    split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
    split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)
    split_regions_df["sequence"] = [str(seqs[idx])[:seq_len].upper() for idx in range(usable_dmrs)]
    split_regions_df = assign_non_overlapping_groups(split_regions_df, "chr", "start_expanded", "end_expanded")

    group_ids = split_regions_df["overlap_group"].drop_duplicates().to_numpy()
    num_train_groups = max(1, int(len(group_ids) * args.train_ratio))
    train_group_ids = set(group_ids[:num_train_groups].tolist())
    train_mask = split_regions_df["overlap_group"].isin(train_group_ids).to_numpy()
    train_idx = torch.from_numpy(np.flatnonzero(train_mask)).long()
    val_idx = torch.from_numpy(np.flatnonzero(~train_mask)).long()

    train_region_metadata = split_regions_df.iloc[train_idx.numpy()].reset_index(drop=True)
    val_region_metadata = split_regions_df.iloc[val_idx.numpy()].reset_index(drop=True)

    train_query_tensor, train_context_tensor, train_target_tensor, train_loss_mask, train_region_metadata = augment_with_reverse_complement(
        query_tensor[train_idx],
        context_tensor[train_idx],
        target_tensor[train_idx],
        loss_mask[train_idx],
        base_ids_tensor[train_idx],
        sequence_onehot[train_idx],
        train_region_metadata,
        args,
    )
    val_query_tensor, val_context_tensor, val_target_tensor, val_loss_mask, val_region_metadata = augment_with_reverse_complement(
        query_tensor[val_idx],
        context_tensor[val_idx],
        target_tensor[val_idx],
        loss_mask[val_idx],
        base_ids_tensor[val_idx],
        sequence_onehot[val_idx],
        val_region_metadata,
        args,
    )

    train_dataset = torch.utils.data.TensorDataset(
        train_query_tensor,
        train_context_tensor,
        train_target_tensor,
        train_loss_mask,
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_query_tensor,
        val_context_tensor,
        val_target_tensor,
        val_loss_mask,
    )

    return PreparedExperimentData(
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        usable_dmrs=usable_dmrs,
        seq_len=seq_len,
        post_filter_len=post_filter_len,
        train_regions=len(train_dataset),
        val_regions=len(val_dataset),
        non_overlap_groups=split_regions_df["overlap_group"].nunique(),
        val_region_metadata=val_region_metadata,
    )


def prepare_multimodal_multitask_data(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> dict:
    usable_dmrs = min(num_dmrs, len(df_dmr), len(seqs), len(hmcg_tracks), len(atac_tracks), len(mcg_tracks))
    seq_len = min(len(hmcg_tracks[0]), len(atac_tracks[0]), len(seqs[0]))
    post_filter_len = min(seq_len, 4)

    if usable_dmrs <= 0:
        raise ValueError(
            "prepare_multimodal_multitask_data: no usable regions/tracks. "
            f"Lengths: df_dmr={len(df_dmr)}, seqs={len(seqs)}, hmcg_tracks={len(hmcg_tracks)}, "
            f"atac_tracks={len(atac_tracks)}, mcg_tracks={len(mcg_tracks)}. "
            "Check your --hm5c-bedgraph/--m5c-bedgraph/--atac-bw paths and DMR CSV."
        )

    track_arrays = get_track_arrays(args, mcg_tracks, hmcg_tracks, atac_tracks, usable_dmrs, seq_len)
    base_ids_tensor, sequence_onehot = build_sequence_tensor(seqs, usable_dmrs, seq_len)

    query_tensor = tensorize_track_modality("atac", track_arrays, args)
    hm5c = track_arrays.get("5hmc")
    mc5c = track_arrays.get("5mc")
    if mc5c is None:
        raise ValueError("5mC tracks required for multitask experiments but not available.")
    hm5c_tensor = torch.tensor(hm5c, dtype=torch.float32).unsqueeze(-1) /100
    mc5c_tensor = torch.tensor(mc5c, dtype=torch.float32).unsqueeze(-1) /100
    # compute complement channel `c` so that mc + hm + c = 1 per position; handle numerical issues by
    # clamping and re-normalizing per-position
    c_tensor = (1.0 - mc5c_tensor - hm5c_tensor).clamp(min=0.0)
    multitask_target = torch.cat([mc5c_tensor, hm5c_tensor, c_tensor], dim=-1)
    # normalize per-position to sum to 1 (in case input sums deviate from 1)
    sum_per_pos = multitask_target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    multitask_target = multitask_target / sum_per_pos

    loss_mask = resolve_loss_mask(getattr(args, "mask_mode", "cpg_both"), base_ids_tensor)
    loss_mask = loss_mask.repeat(1, 1, 3)

    split_regions_df = df_dmr.iloc[:usable_dmrs].copy().reset_index().rename(columns={"index": "original_idx"})
    split_regions_df["chr"] = split_regions_df["chr"].astype(str)
    split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
    split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)
    split_regions_df["sequence"] = [str(seqs[idx])[:seq_len].upper() for idx in range(usable_dmrs)]
    split_regions_df = assign_non_overlapping_groups(split_regions_df, "chr", "start_expanded", "end_expanded")

    group_ids = split_regions_df["overlap_group"].drop_duplicates().to_numpy()
    num_train_groups = max(1, int(len(group_ids) * getattr(args, "train_ratio", 0.8)))
    train_group_ids = set(group_ids[:num_train_groups].tolist())
    train_mask = split_regions_df["overlap_group"].isin(train_group_ids).to_numpy()
    train_idx = torch.from_numpy(np.flatnonzero(train_mask)).long()
    val_idx = torch.from_numpy(np.flatnonzero(~train_mask)).long()

    train_region_metadata = split_regions_df.iloc[train_idx.numpy()].reset_index(drop=True)
    val_region_metadata = split_regions_df.iloc[val_idx.numpy()].reset_index(drop=True)

    # call augmentation helper which will only augment when args.augment_reverse_complement True
    train_query_tensor, train_context_tensor, train_target_tensor, train_loss_mask, train_region_metadata = augment_with_reverse_complement(
        query_tensor[train_idx],
        sequence_onehot[train_idx],
        multitask_target[train_idx],
        loss_mask[train_idx],
        base_ids_tensor[train_idx],
        sequence_onehot[train_idx],
        train_region_metadata,
        args,
    )

    val_query_tensor, val_context_tensor, val_target_tensor, val_loss_mask, val_region_metadata = augment_with_reverse_complement(
        query_tensor[val_idx],
        sequence_onehot[val_idx],
        multitask_target[val_idx],
        loss_mask[val_idx],
        base_ids_tensor[val_idx],
        sequence_onehot[val_idx],
        val_region_metadata,
        args,
    )

    train_dataset = torch.utils.data.TensorDataset(
        train_query_tensor,
        train_context_tensor,
        train_target_tensor,
        train_loss_mask,
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_query_tensor,
        val_context_tensor,
        val_target_tensor,
        val_loss_mask,
    )

    return {
        "train_loader": torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        "val_loader": torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        "usable_dmrs": usable_dmrs,
        "seq_len": seq_len,
        "post_filter_len": post_filter_len,
        "train_regions": len(train_dataset),
        "val_regions": len(val_dataset),
        "non_overlap_groups": split_regions_df["overlap_group"].nunique(),
        "val_region_metadata": val_region_metadata,
    }


def prepare_experiment_data(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> PreparedExperimentData:
    legacy_args = args
    if not hasattr(legacy_args, "input_modality"):
        legacy_args.input_modality = "5mc"
    if not hasattr(legacy_args, "target_modality"):
        legacy_args.target_modality = "5hmc"
    if not hasattr(legacy_args, "context_modalities"):
        legacy_context_modalities = []
        if getattr(legacy_args, "use_sequence", False):
            legacy_context_modalities.append("sequence")
        if getattr(legacy_args, "use_atac", False):
            legacy_context_modalities.append("atac")
        legacy_args.context_modalities = legacy_context_modalities
    if not hasattr(legacy_args, "mask_mode"):
        legacy_args.mask_mode = "cpg_both"
    if not hasattr(legacy_args, "augment_reverse_complement"):
        legacy_args.augment_reverse_complement = False
    return prepare_modality_experiment_data(num_dmrs, legacy_args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)


def toy_test_non_overlap_split(train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    toy_regions = pd.DataFrame(
        [
            {"original_idx": 0, "chr": "1", "start_expanded": 100, "end_expanded": 180},
            {"original_idx": 1, "chr": "1", "start_expanded": 160, "end_expanded": 220},
            {"original_idx": 2, "chr": "1", "start_expanded": 230, "end_expanded": 260},
            {"original_idx": 3, "chr": "1", "start_expanded": 255, "end_expanded": 310},
            {"original_idx": 4, "chr": "1", "start_expanded": 400, "end_expanded": 450},
            {"original_idx": 5, "chr": "1", "start_expanded": 451, "end_expanded": 500},
            {"original_idx": 6, "chr": "2", "start_expanded": 100, "end_expanded": 140},
            {"original_idx": 7, "chr": "2", "start_expanded": 130, "end_expanded": 180},
            {"original_idx": 8, "chr": "2", "start_expanded": 300, "end_expanded": 360},
            {"original_idx": 9, "chr": "2", "start_expanded": 500, "end_expanded": 560},
        ]
    )

    grouped = assign_non_overlapping_groups(toy_regions, "chr", "start_expanded", "end_expanded")
    group_ids = grouped["overlap_group"].drop_duplicates().to_numpy()
    num_train_groups = max(1, int(len(group_ids) * train_ratio))
    train_group_ids = set(group_ids[:num_train_groups].tolist())

    train_regions = grouped[grouped["overlap_group"].isin(train_group_ids)].reset_index(drop=True)
    val_regions = grouped[~grouped["overlap_group"].isin(train_group_ids)].reset_index(drop=True)

    overlap_rows = []
    for train_row in train_regions.itertuples(index=False):
        for val_row in val_regions.itertuples(index=False):
            same_chr = train_row.chr == val_row.chr
            overlaps = train_row.start_expanded <= val_row.end_expanded and val_row.start_expanded <= train_row.end_expanded
            if same_chr and overlaps:
                overlap_rows.append(
                    {
                        "train_original_idx": train_row.original_idx,
                        "val_original_idx": val_row.original_idx,
                        "chr": train_row.chr,
                        "train_start": train_row.start_expanded,
                        "train_end": train_row.end_expanded,
                        "val_start": val_row.start_expanded,
                        "val_end": val_row.end_expanded,
                    }
                )

    overlap_frame = pd.DataFrame(overlap_rows)

    print("Toy regions with overlap groups:")
    print(grouped[["original_idx", "chr", "start_expanded", "end_expanded", "overlap_group"]])
    print("\nTrain groups:", sorted(train_group_ids))
    print("Train original_idx:", train_regions["original_idx"].tolist())
    print("Val original_idx:", val_regions["original_idx"].tolist())
    if overlap_frame.empty:
        print("\nResult: no train/val overlaps detected.")
    else:
        print("\nResult: found train/val overlaps.")
        print(overlap_frame)

    return grouped, train_regions, val_regions


class LazyM5cSequenceAtacDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that fetches sequence / 5mC / 5hmC / ATAC on-the-fly.

    Only the BED metadata (chr/start/end, small) is stored in memory.
    Genome FASTA, Tabix, and bigWig file handles are opened **per worker** so that
    multi-process DataLoader (num_workers > 0) works correctly with fork.

    Thread-safety: lazy handle initialization is protected by a reentrant lock
    to prevent concurrent first-call races in multi-threaded DataLoader workers.
    """

    def _open_handles(self):
        import pyfaidx
        self._genome = pyfaidx.Fasta(self.genome_fasta)
        self._tbx_5mc = _open_track_handle(self.m5c_bedgraph, self._m5c_fmt)
        self._tbx_5hmc = _open_track_handle(self.hm5c_bedgraph, self._hm5c_fmt)
        self._atac_bw = _open_track_handle(self.atac_bw_path, self._atac_fmt)

    def _close_handles(self):
        """Close all open file handles. Safe to call multiple times."""
        for attr in ("_genome", "_tbx_5mc", "_tbx_5hmc", "_atac_bw"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def __init__(
        self,
        indices: list[int],
        df_dmr: pd.DataFrame,
        genome_fasta: str,
        m5c_bedgraph: str,
        hm5c_bedgraph: str,
        atac_bw_path: str,
        target_length: int,
        mask_mode: str,
        atac_scaling: str,
        augment_rc: bool = False,
        clip_at_zero: bool = False,
    ):
        self.indices = indices
        self.df_dmr = df_dmr
        self.genome_fasta = genome_fasta
        self.m5c_bedgraph = m5c_bedgraph
        self.hm5c_bedgraph = hm5c_bedgraph
        self.atac_bw_path = atac_bw_path
        self.target_length = target_length
        self.mask_mode = mask_mode
        self.atac_scaling = atac_scaling
        self.augment_rc = augment_rc
        self.clip_at_zero = clip_at_zero
        self.N = len(indices)
        self._base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
        # Detect track formats once at init (cached — zero per-sample overhead)
        self._m5c_fmt = _detect_track_format(m5c_bedgraph)
        self._hm5c_fmt = _detect_track_format(hm5c_bedgraph)
        self._atac_fmt = _detect_track_format(atac_bw_path)
        self._genome = None
        self._tbx_5mc = None
        self._tbx_5hmc = None
        self._atac_bw = None
        self._lock = threading.RLock()

    def __len__(self):
        return self.N * (2 if self.augment_rc else 1)

    def __getstate__(self):
        """Strip file handles for pickling across worker processes."""
        state = self.__dict__.copy()
        state["_genome"] = None
        state["_tbx_5mc"] = None
        state["_tbx_5hmc"] = None
        state["_atac_bw"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._genome = None
        self._tbx_5mc = None
        self._tbx_5hmc = None
        self._atac_bw = None

    def __del__(self):
        self._close_handles()

    def __getitem__(self, idx):
        # Thread-safe lazy handle initialization (once per worker)
        with self._lock:
            if self._genome is None:
                self._close_handles()  # belt-and-suspenders: close any stale handles first
                self._open_handles()

        is_rc = self.augment_rc and idx >= self.N
        real_idx = self.indices[idx % self.N]

        row = self.df_dmr.iloc[real_idx]
        chrom_name = str(row["chr"]).removeprefix("chr")
        chrom = "chr" + chrom_name
        start = int(row["start_expanded"])
        end = int(row["end_expanded"])

        # --- fetch on the fly ---
        seq_str = get_sequence(chrom, start, end, self._genome)
        hm5c = read_track_region(
            self._tbx_5hmc,
            self._hm5c_fmt,
            chrom,
            start,
            end,
            clip_at_zero=self.clip_at_zero,
        )
        atac = read_track_region(
            self._atac_bw,
            self._atac_fmt,
            chrom,
            start,
            end + 1,
            clip_at_zero=self.clip_at_zero,
        )
        m5c = read_track_region(
            self._tbx_5mc,
            self._m5c_fmt,
            chrom,
            start,
            end,
            clip_at_zero=self.clip_at_zero,
        )

        # --- determine common length ---
        seq_len = min(self.target_length, len(seq_str), len(hm5c), len(atac), len(m5c))

        # --- build tensors ---
        base_ids = sequence_to_base_ids(seq_str, seq_len, self._base_to_index)
        sequence_onehot = F.one_hot(base_ids, num_classes=4).float()

        m5c_t = torch.tensor(m5c[:seq_len], dtype=torch.float32).unsqueeze(-1)
        hm5c_t = torch.tensor(hm5c[:seq_len], dtype=torch.float32).unsqueeze(-1)
        atac_t = torch.tensor(atac[:seq_len], dtype=torch.float32).unsqueeze(-1)

        # ATAC scaling (per-sample minmax)
        if self.atac_scaling == "minmax":
            a_min = atac_t.amin(dim=0, keepdim=True)
            a_max = atac_t.amax(dim=0, keepdim=True)
            a_range = (a_max - a_min).clamp_min(1e-6)
            atac_t = (atac_t - a_min) / a_range

        # loss mask (add batch dim, resolve, then squeeze)
        loss_mask = resolve_loss_mask(self.mask_mode, base_ids.unsqueeze(0))[0]

        if is_rc:
            m5c_t = torch.flip(m5c_t, dims=[0])
            sequence_onehot = torch.flip(sequence_onehot, dims=[0])
            complement_index = BASE_COMPLEMENT_INDEX.to(sequence_onehot.device)
            sequence_onehot = sequence_onehot.index_select(dim=-1, index=complement_index)
            atac_t = torch.flip(atac_t, dims=[0])
            hm5c_t = torch.flip(hm5c_t, dims=[0])
            base_ids_rc = torch.argmax(sequence_onehot, dim=-1)
            loss_mask = resolve_loss_mask(self.mask_mode, base_ids_rc.unsqueeze(0))[0]

        return m5c_t, sequence_onehot, atac_t, hm5c_t, loss_mask


class LazyM5cSequenceAtacRnaDataset(LazyM5cSequenceAtacDataset):
    """Lazy dataset that additionally returns an RNA-coverage track.

    Behavior mirrors :class:`LazyM5cSequenceAtacDataset` but adds:
      - an optional ``rna_bw_path`` (bigWig only) opened per worker,
      - a per-sample minmax-scaled RNA tensor,
      - an additional return value: ``rna_t`` of shape ``(L, 1)``.

    The RNA track is intended as an extra context modality stacked alongside
    ``sequence`` and ``atac``.  When ``rna_bw_path`` is ``None`` the dataset
    falls back to a zero-filled track (same convention as ATAC absence in
    :mod:`data_sequence_only`).
    """

    def _open_handles(self):
        import pyfaidx
        self._genome = pyfaidx.Fasta(self.genome_fasta)
        self._tbx_5mc = _open_track_handle(self.m5c_bedgraph, self._m5c_fmt)
        self._tbx_5hmc = _open_track_handle(self.hm5c_bedgraph, self._hm5c_fmt)
        self._atac_bw = _open_track_handle(self.atac_bw_path, self._atac_fmt)
        self._rna_bw = _open_track_handle(self.rna_bw_path, self._rna_fmt) if self.rna_bw_path else None

    def _close_handles(self):
        for attr in ("_genome", "_tbx_5mc", "_tbx_5hmc", "_atac_bw", "_rna_bw"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def __init__(
        self,
        indices: list[int],
        df_dmr: pd.DataFrame,
        genome_fasta: str,
        m5c_bedgraph: str,
        hm5c_bedgraph: str,
        atac_bw_path: str,
        target_length: int,
        mask_mode: str,
        atac_scaling: str,
        rna_bw_path: str | None = None,
        rna_scaling: str = "minmax",
        augment_rc: bool = False,
        clip_at_zero: bool = False,
    ):
        super().__init__(
            indices=indices,
            df_dmr=df_dmr,
            genome_fasta=genome_fasta,
            m5c_bedgraph=m5c_bedgraph,
            hm5c_bedgraph=hm5c_bedgraph,
            atac_bw_path=atac_bw_path,
            target_length=target_length,
            mask_mode=mask_mode,
            atac_scaling=atac_scaling,
            augment_rc=augment_rc,
            clip_at_zero=clip_at_zero,
        )
        self.rna_bw_path = rna_bw_path
        self.rna_scaling = rna_scaling
        self._rna_fmt = _detect_track_format(rna_bw_path) if rna_bw_path else "bigwig"
        self._rna_bw = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_rna_bw"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._rna_bw = None

    def __getitem__(self, idx):
        # Thread-safe lazy handle initialization (once per worker)
        with self._lock:
            if self._genome is None:
                self._close_handles()
                self._open_handles()

        is_rc = self.augment_rc and idx >= self.N
        real_idx = self.indices[idx % self.N]

        row = self.df_dmr.iloc[real_idx]
        chrom_name = str(row["chr"]).removeprefix("chr")
        chrom = "chr" + chrom_name
        start = int(row["start_expanded"])
        end = int(row["end_expanded"])

        # --- fetch on the fly ---
        seq_str = get_sequence(chrom, start, end, self._genome)
        hm5c = read_track_region(self._tbx_5hmc, self._hm5c_fmt, chrom, start, end, clip_at_zero=self.clip_at_zero)
        atac = read_track_region(self._atac_bw, self._atac_fmt, chrom, start, end + 1, clip_at_zero=self.clip_at_zero)
        m5c = read_track_region(self._tbx_5mc, self._m5c_fmt, chrom, start, end, clip_at_zero=self.clip_at_zero)
        if self._rna_bw is not None:
            rna = read_track_region(self._rna_bw, self._rna_fmt, chrom, start, end + 1, clip_at_zero=self.clip_at_zero)
        else:
            rna = np.zeros(end - start, dtype=np.float32)

        # --- determine common length ---
        seq_len = min(self.target_length, len(seq_str), len(hm5c), len(atac), len(m5c), len(rna))

        # --- build tensors ---
        base_ids = sequence_to_base_ids(seq_str, seq_len, self._base_to_index)
        sequence_onehot = F.one_hot(base_ids, num_classes=4).float()

        m5c_t = torch.tensor(m5c[:seq_len], dtype=torch.float32).unsqueeze(-1)
        hm5c_t = torch.tensor(hm5c[:seq_len], dtype=torch.float32).unsqueeze(-1)
        atac_t = torch.tensor(atac[:seq_len], dtype=torch.float32).unsqueeze(-1)
        rna_t = torch.tensor(rna[:seq_len], dtype=torch.float32).unsqueeze(-1)

        if self.atac_scaling == "minmax":
            a_min = atac_t.amin(dim=0, keepdim=True)
            a_max = atac_t.amax(dim=0, keepdim=True)
            a_range = (a_max - a_min).clamp_min(1e-6)
            atac_t = (atac_t - a_min) / a_range
        if self.rna_scaling == "minmax":
            r_min = rna_t.amin(dim=0, keepdim=True)
            r_max = rna_t.amax(dim=0, keepdim=True)
            r_range = (r_max - r_min).clamp_min(1e-6)
            rna_t = (rna_t - r_min) / r_range

        loss_mask = resolve_loss_mask(self.mask_mode, base_ids.unsqueeze(0))[0]

        if is_rc:
            m5c_t = torch.flip(m5c_t, dims=[0])
            sequence_onehot = torch.flip(sequence_onehot, dims=[0])
            complement_index = BASE_COMPLEMENT_INDEX.to(sequence_onehot.device)
            sequence_onehot = sequence_onehot.index_select(dim=-1, index=complement_index)
            atac_t = torch.flip(atac_t, dims=[0])
            hm5c_t = torch.flip(hm5c_t, dims=[0])
            rna_t = torch.flip(rna_t, dims=[0])
            base_ids_rc = torch.argmax(sequence_onehot, dim=-1)
            loss_mask = resolve_loss_mask(self.mask_mode, base_ids_rc.unsqueeze(0))[0]

        return m5c_t, sequence_onehot, atac_t, hm5c_t, rna_t, loss_mask


if __name__ == "__main__":
    toy_test_non_overlap_split()