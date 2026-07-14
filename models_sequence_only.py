"""Model class that supports ATAC as optional input.

Mirrors `M5CQuerySequenceAtacCrossHyenaRegressorModelB` in `models.py`,
but with `use_atac=False` to skip the ATAC projection entirely (rather
than feeding a zero-vector placeholder).

Why a separate file?
- Keeps `models.py` (main branch) untouched.
- Avoids the shape-mismatch issue in `transfer_pretrained_weights`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models import (
    SinusoidalPositionalEncoding,
    StripedHyenaBlock,
)


class M5CQuerySequenceOnlyCrossHyenaRegressorModelB(nn.Module):
    """Query: 5mC track. Context: DNA sequence (+ optional ATAC). Predict 5hmC.

    Constructor flag:
      use_atac=True  → identical architecture to ModelB with ATAC.
      use_atac=False → no ATAC projection; context_proj takes only seq.

    The forward signature is the same in both cases — passing `atac_present=0.0`
    (or just None) is allowed; the model branches internally.
    """

    def __init__(
        self,
        seq_len: int,
        query_dim: int = 1,
        sequence_dim: int = 4,
        atac_dim: int = 1,
        hidden_dim: int = 64,
        use_positional_encoding: bool = False,
        num_blocks: int = 2,
        fusion_type: str = "cross_hyena",
        use_atac: bool = True,
    ):
        super().__init__()
        self.use_atac = use_atac

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.sequence_proj = nn.Linear(sequence_dim, hidden_dim)
        self.sequence_norm = nn.LayerNorm(hidden_dim)

        if self.use_atac:
            self.atac_proj = nn.Linear(atac_dim, hidden_dim)
            self.atac_norm = nn.LayerNorm(hidden_dim)
            # context = cat([seq, atac]) → 2*hidden_dim
            self.context_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.atac_proj = None
            self.atac_norm = None
            # context = seq alone → hidden_dim
            self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        self.context_norm = nn.LayerNorm(hidden_dim)
        self.position_encoding = (
            SinusoidalPositionalEncoding(hidden_dim, seq_len)
            if use_positional_encoding
            else None
        )
        self.blocks = nn.ModuleList(
            [StripedHyenaBlock(hidden_dim, seq_len, fusion_type=fusion_type) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        m5c_track: torch.Tensor,
        sequence_track: torch.Tensor,
        atac_track: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.query_norm(self.query_proj(m5c_track))
        sequence_hidden = self.sequence_norm(self.sequence_proj(sequence_track))

        if self.use_atac and atac_track is not None:
            atac_hidden = self.atac_norm(self.atac_proj(atac_track))
            context_in = torch.cat([sequence_hidden, atac_hidden], dim=-1)
        else:
            context_in = sequence_hidden

        context = self.context_norm(self.context_proj(context_in))

        if self.position_encoding is not None:
            hidden = self.position_encoding(hidden)
            context = self.position_encoding(context)

        for block in self.blocks:
            hidden = block(hidden, context)

        return self.head(self.final_norm(hidden))