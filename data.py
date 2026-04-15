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


def find_cpg_candidate_positions(base_ids: torch.Tensor) -> torch.Tensor:
    is_c = base_ids == 1
    is_g = base_ids == 2
    right_is_g = torch.zeros_like(is_g)
    right_is_g[:, :-1] = is_g[:, 1:]
    left_is_c = torch.zeros_like(is_c)
    left_is_c[:, 1:] = is_c[:, :-1]
    return (is_c & right_is_g) | (is_g & left_is_c)


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
    tbx_5mc = pysam.TabixFile(args.m5c_bedgraph) if args.use_m5c and args.m5c_bedgraph else None
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


def prepare_experiment_data(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> PreparedExperimentData:
    track_lengths = [len(hmcg_tracks[0]), len(atac_tracks[0]), len(seqs[0])]
    usable_counts = [num_dmrs, len(df_dmr), len(seqs), len(hmcg_tracks), len(atac_tracks)]
    if mcg_tracks:
        track_lengths.append(len(mcg_tracks[0]))
        usable_counts.append(len(mcg_tracks))

    usable_dmrs = min(usable_counts)
    seq_len = min(track_lengths)
    post_filter_len = min(seq_len, 4)
    base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
    if mcg_tracks:
        query_tensor = torch.tensor(
            np.stack([np.asarray(mcg_tracks[idx][:seq_len], dtype=np.float32) for idx in range(usable_dmrs)]),
            dtype=torch.float32,
        ).unsqueeze(-1)
    else:
        query_tensor = torch.zeros((usable_dmrs, seq_len, 1), dtype=torch.float32)

    hm5c_target = torch.tensor(
        np.stack([np.asarray(hmcg_tracks[idx][:seq_len], dtype=np.float32) for idx in range(usable_dmrs)]),
        dtype=torch.float32,
    ).unsqueeze(-1)
    raw_atac_tensor = torch.tensor(
        np.stack([np.asarray(atac_tracks[idx][:seq_len], dtype=np.float32) for idx in range(usable_dmrs)]),
        dtype=torch.float32,
    ).unsqueeze(-1)
    atac_tensor = scale_atac_tensor(raw_atac_tensor, args.atac_scaling)
    base_ids_tensor = torch.stack([
        sequence_to_base_ids(seqs[idx], seq_len, base_to_index) for idx in range(usable_dmrs)
    ])
    sequence_onehot = F.one_hot(base_ids_tensor, num_classes=4).float()
    loss_mask = find_cpg_candidate_positions(base_ids_tensor).unsqueeze(-1).float()

    split_regions_df = df_dmr.iloc[:usable_dmrs].copy().reset_index().rename(columns={"index": "original_idx"})
    split_regions_df["chr"] = split_regions_df["chr"].astype(str)
    split_regions_df["start_expanded"] = split_regions_df["start_expanded"].astype(int)
    split_regions_df["end_expanded"] = split_regions_df["end_expanded"].astype(int)
    split_regions_df = assign_non_overlapping_groups(split_regions_df, "chr", "start_expanded", "end_expanded")

    group_ids = split_regions_df["overlap_group"].drop_duplicates().to_numpy()
    num_train_groups = max(1, int(len(group_ids) * args.train_ratio))
    train_group_ids = set(group_ids[:num_train_groups].tolist())
    train_mask = split_regions_df["overlap_group"].isin(train_group_ids).to_numpy()
    train_idx = torch.from_numpy(np.flatnonzero(train_mask)).long()
    val_idx = torch.from_numpy(np.flatnonzero(~train_mask)).long()

    train_dataset = torch.utils.data.TensorDataset(
        query_tensor[train_idx],
        sequence_onehot[train_idx],
        atac_tensor[train_idx],
        hm5c_target[train_idx],
        loss_mask[train_idx],
    )
    val_dataset = torch.utils.data.TensorDataset(
        query_tensor[val_idx],
        sequence_onehot[val_idx],
        atac_tensor[val_idx],
        hm5c_target[val_idx],
        loss_mask[val_idx],
    )

    return PreparedExperimentData(
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        usable_dmrs=usable_dmrs,
        seq_len=seq_len,
        post_filter_len=post_filter_len,
        train_regions=len(train_idx),
        val_regions=len(val_idx),
        non_overlap_groups=split_regions_df["overlap_group"].nunique(),
        val_region_metadata=split_regions_df.iloc[val_idx.numpy()].reset_index(drop=True),
    )