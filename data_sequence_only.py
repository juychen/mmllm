"""Dataset class that supports ATAC as optional input.

This mirrors the ATAC-required `LazyM5cSequenceAtacDataset` in `data.py`,
but allows `atac_bw_path=None`. When ATAC is omitted, the dataset returns a
zero-filled placeholder track so the downstream model can still consume a
`(N, seq_len, 1)` tensor.

Why a separate file?
- Keeps `data.py` (main branch) untouched.
- Avoids breaking any other script that depends on `LazyM5cSequenceAtacDataset`.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import pysam
import pyfaidx
import pyBigWig
import torch
import torch.nn.functional as F

from data import (
    BASE_COMPLEMENT_INDEX,
    fast_tabix_to_track,
    get_sequence,
    resolve_loss_mask,
    sequence_to_base_ids,
)


class LazyM5cSequenceOnlyDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that fetches 5mC / 5hmC / (optional) ATAC on-the-fly.

    Behavior matches `LazyM5cSequenceAtacDataset` in `data.py` except:
    - `atac_bw_path` is optional (None → zero placeholder).
    - All file handles are opened lazily per worker.
    """

    def _open_handles(self):
        self._genome = pyfaidx.Fasta(self.genome_fasta)
        self._tbx_5mc = pysam.TabixFile(self.m5c_bedgraph)
        self._tbx_5hmc = pysam.TabixFile(self.hm5c_bedgraph)
        self._atac_bw = pyBigWig.open(self.atac_bw_path) if self.atac_bw_path else None

    def _close_handles(self):
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
        df_dmr,
        genome_fasta: str,
        m5c_bedgraph: str,
        hm5c_bedgraph: str,
        target_length: int,
        mask_mode: str,
        atac_bw_path: Optional[str] = None,
        atac_scaling: str = "none",
        augment_rc: bool = False,
    ):
        self.indices = indices
        self.df_dmr = df_dmr
        self.genome_fasta = genome_fasta
        self.m5c_bedgraph = m5c_bedgraph
        self.hm5c_bedgraph = hm5c_bedgraph
        self.atac_bw_path = atac_bw_path  # may be None
        self.target_length = target_length
        self.mask_mode = mask_mode
        self.atac_scaling = atac_scaling
        self.augment_rc = augment_rc
        self.N = len(indices)
        self._base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
        self._genome = None
        self._tbx_5mc = None
        self._tbx_5hmc = None
        self._atac_bw = None
        self._lock = threading.RLock()

    def __len__(self):
        return self.N * (2 if self.augment_rc else 1)

    def __getstate__(self):
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

        seq_str = get_sequence(chrom, start, end, self._genome)
        hm5c = fast_tabix_to_track(self._tbx_5hmc, chrom_name, start, end)
        m5c = fast_tabix_to_track(self._tbx_5mc, chrom_name, start, end)
        if self._atac_bw is not None:
            atac = np.nan_to_num(self._atac_bw.values(chrom, start, end + 1), nan=0.0)
        else:
            atac = np.zeros(end - start, dtype=np.float32)

        seq_len = min(self.target_length, len(seq_str), len(hm5c), len(atac), len(m5c))

        base_ids = sequence_to_base_ids(seq_str, seq_len, self._base_to_index)
        sequence_onehot = F.one_hot(base_ids, num_classes=4).float()

        m5c_t = torch.tensor(m5c[:seq_len], dtype=torch.float32).unsqueeze(-1)
        hm5c_t = torch.tensor(hm5c[:seq_len], dtype=torch.float32).unsqueeze(-1)
        atac_t = torch.tensor(atac[:seq_len], dtype=torch.float32).unsqueeze(-1)

        # ATAC presence flag — downstream model can use this to gate the
        # ATAC branch entirely (rather than relying on zero-vector magic).
        atac_present = torch.tensor(1.0 if self._atac_bw is not None else 0.0)

        if self._atac_bw is not None and self.atac_scaling == "minmax":
            a_min = atac_t.amin(dim=0, keepdim=True)
            a_max = atac_t.amax(dim=0, keepdim=True)
            a_range = (a_max - a_min).clamp_min(1e-6)
            atac_t = (atac_t - a_min) / a_range

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

        return m5c_t, sequence_onehot, atac_t, hm5c_t, loss_mask, atac_present