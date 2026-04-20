from dataclasses import dataclass

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


def get_sequence(chrom: str, start: int, end: int, genome: pyfaidx.Fasta) -> str:
    return genome[chrom][start - 1 : end].seq


def fast_tabix_to_track(tbx: pysam.TabixFile, chrom: str, start_1based: int, end_1based: int) -> np.ndarray:
    region_start_0based = int(start_1based) - 1
    region_end_0based = int(end_1based)
    data = [line.split("\t") for line in tbx.fetch(chrom, region_start_0based, region_end_0based)]
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


def load_data(args):
    df_dmr = pd.read_csv(args.dmr_csv)
    target_length = args.target_length
    half_window = target_length // 2
    df_dmr["start_expanded"] = df_dmr["start"]
    df_dmr["end_expanded"] = df_dmr["end"]
    short_mask = df_dmr["length"] < target_length
    df_dmr.loc[short_mask, "start_expanded"] = df_dmr.loc[short_mask, "center"] - half_window
    df_dmr.loc[short_mask, "end_expanded"] = df_dmr.loc[short_mask, "center"] + half_window - 1

    genome = pyfaidx.Fasta(args.genome_fasta)
    tbx_5hmc = pysam.TabixFile(args.hm5c_bedgraph)
    requested_track_modalities = set()
    if hasattr(args, "input_modality"):
        requested_track_modalities.add(args.input_modality)
    if hasattr(args, "target_modality"):
        requested_track_modalities.add(args.target_modality)
    if hasattr(args, "context_modalities"):
        requested_track_modalities.update(modality for modality in args.context_modalities if modality in TRACK_MODALITIES)

    should_load_5mc = bool(getattr(args, "m5c_bedgraph", None)) and (
        getattr(args, "use_m5c", False) or "5mc" in requested_track_modalities
    )
    tbx_5mc = pysam.TabixFile(args.m5c_bedgraph) if should_load_5mc else None
    atac_bw = pyBigWig.open(args.atac_bw)

    seqs = []
    mcg_tracks = []
    hmcg_tracks = []
    atac_tracks = []
    for _, row in df_dmr.iterrows():
        chrom = "chr" + str(row["chr"])
        start = int(row["start_expanded"])
        end = int(row["end_expanded"])
        seqs.append(get_sequence(chrom, start, end, genome))
        if tbx_5mc is not None:
            mcg_tracks.append(fast_tabix_to_track(tbx_5mc, chrom.replace("chr", ""), start, end))
        hmcg_tracks.append(fast_tabix_to_track(tbx_5hmc, chrom.replace("chr", ""), start, end))
        atac_tracks.append(np.nan_to_num(atac_bw.values(chrom, start, end + 1), nan=0.0))

    return df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks


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