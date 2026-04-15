import argparse
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import pysam
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyenaFilter(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        filter_len: int | None = None,
        emb_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.filter_len = filter_len if filter_len is not None else seq_len

        t = torch.linspace(0, 1, self.filter_len).unsqueeze(-1)
        freqs = 2 * torch.pi * torch.arange(1, emb_dim // 2 + 1).float()
        pos_enc = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)
        self.register_buffer("pos_enc", pos_enc)

        layers = []
        in_dim = emb_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, emb_dim), nn.SiLU()]
            in_dim = emb_dim
        layers.append(nn.Linear(in_dim, d_model))
        self.mlp = nn.Sequential(*layers)
        self.log_decay = nn.Parameter(torch.zeros(d_model))

    def forward(self) -> torch.Tensor:
        h = self.mlp(self.pos_enc)
        decay_steps = torch.arange(self.filter_len, device=h.device).unsqueeze(-1)
        decay = torch.exp(-self.log_decay.abs()) ** decay_steps
        h = h * decay
        return h.transpose(0, 1)


class HyenaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        short_kernel: int = 3,
        filter_len: int | None = None,
        long_mixer: str = "hyena",
        long_conv_kernel: int = 65,
    ):
        super().__init__()
        self.long_mixer = long_mixer
        self.filter_len = filter_len if filter_len is not None else seq_len

        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.short_conv_k = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=short_kernel,
            padding=short_kernel - 1,
            groups=d_model,
            bias=True,
        )
        self.short_conv_v = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=short_kernel,
            padding=short_kernel - 1,
            groups=d_model,
            bias=True,
        )

        if self.long_mixer == "hyena":
            self.filter = HyenaFilter(d_model, seq_len, filter_len=self.filter_len)
            self.long_conv = None
        elif self.long_mixer == "conv":
            self.filter = None
            self.long_conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=long_conv_kernel,
                padding=long_conv_kernel // 2,
                groups=d_model,
                bias=True,
            )
        else:
            raise ValueError(f"Unknown long_mixer: {long_mixer}")

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.SiLU()

    def _fft_long_conv(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        length = u.shape[-1]
        fft_size = 2 * length
        u_f = torch.fft.rfft(u.float(), n=fft_size)
        h_f = torch.fft.rfft(h.float(), n=fft_size)
        y = torch.fft.irfft(u_f * h_f, n=fft_size)[..., :length]
        return y.to(u.dtype)

    def _apply_long_mixer(self, u: torch.Tensor) -> torch.Tensor:
        if self.long_mixer == "hyena":
            return self._fft_long_conv(u, self.filter())
        return self.long_conv(u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        k = self.short_conv_k(k.transpose(1, 2))[..., :length]
        v = self.short_conv_v(v.transpose(1, 2))[..., :length]
        y = self._apply_long_mixer(k * v)[..., :length]
        y = self.act(q.transpose(1, 2)) * y
        return self.out_proj(y.transpose(1, 2))


class CrossHyenaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        short_kernel: int = 3,
        filter_len: int | None = None,
        long_mixer: str = "hyena",
        long_conv_kernel: int = 65,
    ):
        super().__init__()
        self.long_mixer = long_mixer
        self.filter_len = filter_len if filter_len is not None else seq_len

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.short_conv_k = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=short_kernel,
            padding=short_kernel - 1,
            groups=d_model,
            bias=True,
        )
        self.short_conv_v = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=short_kernel,
            padding=short_kernel - 1,
            groups=d_model,
            bias=True,
        )

        if self.long_mixer == "hyena":
            self.filter = HyenaFilter(d_model, seq_len, filter_len=self.filter_len)
            self.long_conv = None
        elif self.long_mixer == "conv":
            self.filter = None
            self.long_conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=long_conv_kernel,
                padding=long_conv_kernel // 2,
                groups=d_model,
                bias=True,
            )
        else:
            raise ValueError(f"Unknown long_mixer: {long_mixer}")

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.SiLU()

    def _fft_long_conv(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        length = u.shape[-1]
        fft_size = 2 * length
        u_f = torch.fft.rfft(u.float(), n=fft_size)
        h_f = torch.fft.rfft(h.float(), n=fft_size)
        y = torch.fft.irfft(u_f * h_f, n=fft_size)[..., :length]
        return y.to(u.dtype)

    def _apply_long_mixer(self, u: torch.Tensor) -> torch.Tensor:
        if self.long_mixer == "hyena":
            return self._fft_long_conv(u, self.filter())
        return self.long_conv(u)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        q = self.q_proj(x)
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = self.short_conv_k(k.transpose(1, 2))[..., :length]
        v = self.short_conv_v(v.transpose(1, 2))[..., :length]
        y = self._apply_long_mixer(k * v)[..., :length]
        y = self.act(q.transpose(1, 2)) * y
        return self.out_proj(y.transpose(1, 2))


class MinimalCrossHyenaRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        query_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        post_filter_len: int | None = None,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.cross = CrossHyenaLayer(hidden_dim, seq_len, long_mixer="conv", filter_len=post_filter_len)
        self.cross_to_post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.post_hyena = HyenaLayer(hidden_dim, seq_len)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, query_track: torch.Tensor, context_track: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(query_track)
        context = self.context_proj(context_track)
        hidden = self.cross(query, context)
        hidden = hidden + self.cross_to_post(hidden)
        hidden = self.post_hyena(hidden)
        hidden = self.norm(hidden)
        return self.head(hidden)


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


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (pred - target).pow(2)
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def build_scheduler(optimizer: torch.optim.Optimizer, args, steps_per_experiment: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        total_steps = args.scheduler_t_max if args.scheduler_t_max > 0 else steps_per_experiment
        total_steps = max(1, total_steps)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.scheduler_min_lr,
        )
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=args.learning_rate)


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    targets = []
    masks = []
    with torch.no_grad():
        for query_batch, sequence_batch, atac_batch, target_batch, mask_batch in loader:
            query_batch = query_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            context_batch = torch.cat([sequence_batch, atac_batch], dim=-1)
            pred = model(query_batch, context_batch)
            loss = masked_mse_loss(pred, target_batch, mask_batch)

            batch_count = query_batch.size(0)
            total_loss += loss.item() * batch_count
            total_count += batch_count
            preds.append(pred.detach().cpu())
            targets.append(target_batch.detach().cpu())
            masks.append(mask_batch.detach().cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    masks = torch.cat(masks, dim=0)
    ss_res = (((targets - preds) ** 2) * masks).sum()
    masked_targets = targets[masks.bool()]
    masked_preds = preds[masks.bool()]
    target_mean = masked_targets.mean() if masked_targets.numel() > 0 else torch.tensor(0.0)
    ss_tot = (((targets - target_mean) ** 2) * masks).sum().clamp_min(1e-12)
    r2 = 1.0 - (ss_res / ss_tot)
    if masked_targets.numel() > 1:
        centered_targets = masked_targets - masked_targets.mean()
        centered_preds = masked_preds - masked_preds.mean()
        pearson_denom = centered_targets.pow(2).sum().sqrt() * centered_preds.pow(2).sum().sqrt()
        pearson_r = (centered_targets * centered_preds).sum() / pearson_denom.clamp_min(1e-12)
        pearson_r_value = pearson_r.item()
    else:
        pearson_r_value = float("nan")
    return total_loss / max(total_count, 1), r2.item(), pearson_r_value


@dataclass
class ExperimentResult:
    num_dmrs: int
    train_regions: int
    val_regions: int
    non_overlap_groups: int
    best_epoch: int
    final_lr: float
    best_val_loss: float
    best_val_r2: float
    best_val_pearsonr: float
    final_val_loss: float
    final_val_r2: float
    final_val_pearsonr: float


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
    tbx_5mc = pysam.TabixFile(args.m5c_bedgraph)
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
        mcg_tracks.append(fast_tabix_to_track(tbx_5mc, chrom.replace("chr", ""), start, end))
        hmcg_tracks.append(fast_tabix_to_track(tbx_5hmc, chrom.replace("chr", ""), start, end))
        atac_tracks.append(np.nan_to_num(atac_bw.values(chrom, start, end + 1), nan=0.0))

    return df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks


def run_experiment(num_dmrs: int, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks) -> ExperimentResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usable_dmrs = min(num_dmrs, len(df_dmr), len(seqs), len(mcg_tracks), len(hmcg_tracks), len(atac_tracks))
    seq_len = min(len(mcg_tracks[0]), len(hmcg_tracks[0]), len(atac_tracks[0]), len(seqs[0]))
    post_filter_len = min(seq_len, 4)
    base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}

    query_tensor = torch.tensor(
        np.stack([np.asarray(mcg_tracks[idx][:seq_len], dtype=np.float32) for idx in range(usable_dmrs)]),
        dtype=torch.float32,
    ).unsqueeze(-1)
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MinimalCrossHyenaRegressor(
        seq_len=seq_len,
        query_dim=1,
        context_dim=5,
        hidden_dim=args.hidden_dim,
        post_filter_len=post_filter_len,
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, args.num_epochs)

    best_epoch = 0
    best_val_loss = float("inf")
    best_val_r2 = float("nan")
    best_val_pearsonr = float("nan")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for query_batch, sequence_batch, atac_batch, target_batch, mask_batch in train_loader:
            query_batch = query_batch.to(device)
            sequence_batch = sequence_batch.to(device)
            atac_batch = atac_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            context_batch = torch.cat([sequence_batch, atac_batch], dim=-1)

            optimizer.zero_grad()
            pred = model(query_batch, context_batch)
            loss = masked_mse_loss(pred, target_batch, mask_batch)
            loss.backward()
            optimizer.step()

            batch_count = query_batch.size(0)
            running_loss += loss.item() * batch_count
            seen += batch_count

        train_loss = running_loss / max(seen, 1)
        val_loss, val_r2, val_pearsonr = evaluate(model, val_loader, device)
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[num_dmrs={usable_dmrs}] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_r2={val_r2:.4f} | val_pearsonr={val_pearsonr:.4f} | lr={current_lr:.6g}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            best_val_pearsonr = val_pearsonr
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_loss, final_val_r2, final_val_pearsonr = evaluate(model, val_loader, device)

    return ExperimentResult(
        num_dmrs=usable_dmrs,
        train_regions=len(train_idx),
        val_regions=len(val_idx),
        non_overlap_groups=split_regions_df["overlap_group"].nunique(),
        best_epoch=best_epoch,
        final_lr=optimizer.param_groups[0]["lr"],
        best_val_loss=best_val_loss,
        best_val_r2=best_val_r2,
        best_val_pearsonr=best_val_pearsonr,
        final_val_loss=final_val_loss,
        final_val_r2=final_val_r2,
        final_val_pearsonr=final_val_pearsonr,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run non-overlap 5mC+Sequence+ATAC to 5hmC experiments at different sample sizes.")
    parser.add_argument("--dmr-csv", default="output/dmr_with_sequences.csv")
    parser.add_argument("--genome-fasta", default="/data2st1/junyi/ref/GRCm38.p6.genome.fa")
    parser.add_argument("--m5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.m.bedGraph.gz")
    parser.add_argument("--hm5c-bedgraph", default="/data2st1/junyi/output/llm0401/processed_meth/MC_AMY.CG.h.bedGraph.gz")
    parser.add_argument("--atac-bw", default="/data2st2/junyi/output/atac1112/tobiasbam/BULK/corrected/AMY_MC_track.bw")
    parser.add_argument("--sample-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--atac-scaling", choices=["none", "minmax"], default="minmax")
    parser.add_argument("--output-csv", default="output/sample_size_results.csv")
    parser.add_argument("--output-json", default="output/sample_size_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks = load_data(args)
    results = []
    for sample_size in args.sample_sizes:
        results.append(asdict(run_experiment(sample_size, args, df_dmr, seqs, mcg_tracks, hmcg_tracks, atac_tracks)))

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(args.output_csv, index=False)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()