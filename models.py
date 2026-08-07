import torch
import torch.nn as nn
#from flashfftconv import FlashFFTConv


def _resolve_hyena_filter_lengths(seq_len: int) -> tuple[int, int, int]:
    short_len = min(seq_len, 5)
    mid_len = min(seq_len, max(16, seq_len // 4))
    long_len = seq_len
    return short_len, mid_len, long_len


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


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
            #self.flash_fft = FlashFFTConv(2 * seq_len, dtype=torch.float16)
        elif self.long_mixer == "conv":
            self.filter = None
            self.flash_fft = None
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
    # def _fft_long_conv(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    #     return self.flash_fft(u.half().contiguous(), h.half().contiguous())
    def _fft_long_conv(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        length = u.shape[-1]
        fft_size = 2 * length
        # FFT on CUDA doesn't support bf16; cast to fp32 only for the FFT op.
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
            #self.flash_fft = FlashFFTConv(2 * seq_len, dtype=torch.float16)
        elif self.long_mixer == "conv":
            self.filter = None
            #self.flash_fft = None
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

    # def _fft_long_conv(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    #     return self.flash_fft(u.half().contiguous(), h.half().contiguous())
    
    def _fft_long_conv(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        length = u.shape[-1]
        fft_size = 2 * length
        # FFT on CUDA doesn't support bf16; cast to fp32 only for the FFT op.
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
        use_positional_encoding: bool = False,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim, seq_len) if use_positional_encoding else None
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
        if self.position_encoding is not None:
            query = self.position_encoding(query)
            context = self.position_encoding(context)
        hidden = self.cross(query, context)
        hidden = hidden + self.cross_to_post(hidden)
        hidden = self.post_hyena(hidden)
        hidden = self.norm(hidden)
        out = self.head(hidden)
        # If head produces multiple task logits, convert to probabilities that sum to 1
        if out.shape[-1] > 1:
            return torch.softmax(out, dim=-1)
        return out


class M5CQuerySequenceAtacCrossHyenaRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        query_dim: int = 1,
        sequence_dim: int = 4,
        atac_dim: int = 1,
        hidden_dim: int = 64,
        post_filter_len: int | None = None,
        use_positional_encoding: bool = False,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.sequence_proj = nn.Linear(sequence_dim, hidden_dim)
        self.atac_proj = nn.Linear(atac_dim, hidden_dim)
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        self.atac_norm = nn.LayerNorm(hidden_dim)
        self.context_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim, seq_len) if use_positional_encoding else None
        self.cross = CrossHyenaLayer(hidden_dim, seq_len, long_mixer="conv", filter_len=post_filter_len)
        self.cross_to_post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.post_hyena = HyenaLayer(hidden_dim, seq_len)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, m5c_track: torch.Tensor, sequence_track: torch.Tensor, atac_track: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(m5c_track)
        sequence_hidden = self.sequence_norm(self.sequence_proj(sequence_track))
        atac_hidden = self.atac_norm(self.atac_proj(atac_track))
        context = self.context_norm(self.context_proj(torch.cat([sequence_hidden, atac_hidden], dim=-1)))
        if self.position_encoding is not None:
            query = self.position_encoding(query)
            context = self.position_encoding(context)
        hidden = self.cross(query, context)
        hidden = hidden + self.cross_to_post(hidden)
        hidden = self.post_hyena(hidden)
        hidden = self.norm(hidden)
        return self.head(hidden)


class HyenaResidualBranch(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int, filter_len: int, long_mixer: str):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.hyena = HyenaLayer(hidden_dim, seq_len, filter_len=filter_len, long_mixer=long_mixer)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        branch_input = self.norm(hidden)
        branch_output = self.hyena(branch_input)
        return hidden + self.gate(branch_output)


class MultiScaleHyenaBlock(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int, include_short: bool = True):
        super().__init__()
        short_len, mid_len, long_len = _resolve_hyena_filter_lengths(seq_len)
        self.short_branch = (
            HyenaResidualBranch(hidden_dim, seq_len, filter_len=short_len, long_mixer="conv") if include_short else None
        )
        self.mid_branch = HyenaResidualBranch(hidden_dim, seq_len, filter_len=mid_len, long_mixer="hyena")
        self.long_branch = HyenaResidualBranch(hidden_dim, seq_len, filter_len=long_len, long_mixer="hyena")
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.short_branch is not None:
            hidden = self.short_branch(hidden)
        hidden = self.mid_branch(hidden)
        hidden = self.long_branch(hidden)
        return self.output_norm(hidden)


class CrossHyenaResidualBranch(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int, filter_len: int, long_mixer: str):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross = CrossHyenaLayer(hidden_dim, seq_len, filter_len=filter_len, long_mixer=long_mixer)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        branch_output = self.cross(self.query_norm(hidden), self.context_norm(context))
        return hidden + self.gate(branch_output)


class CrossAttentionResidualBranch(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, head_dim: int | None = None):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        # Pick a (num_heads, head_dim) pair that satisfies:
        #   - head_dim >= 16 (smaller splits lose too much rank per head
        #     and the softmax over a 16k-token context collapses toward
        #     uniform on a freshly-initialized model, which is what froze
        #     training before).
        #   - hidden_dim == num_heads * head_dim (required by nn.MultiheadAttention).
        # We prefer 32 per head; if hidden_dim is not divisible by 32 we
        # fall back to 16; if even that fails we use a single head (i.e.
        # head_dim == hidden_dim) so the user can still run the ablation.
        if head_dim is None:
            if hidden_dim % 32 == 0:
                head_dim = 32
            elif hidden_dim % 16 == 0:
                head_dim = 16
            else:
                head_dim = hidden_dim
        assert hidden_dim % head_dim == 0, (
            f"hidden_dim={hidden_dim} not divisible by head_dim={head_dim}"
        )
        self.num_heads = hidden_dim // head_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Solution A: zero-init the last layer of the gate so that
        # gate(branch_output) starts at exactly 0 → the residual branch is a
        # strict identity at initialization. This prevents the freshly
        # initialized attention output (which is ≈0 because softmax over a
        # 16k-token context collapses to near-uniform on random weights)
        # from polluting the hidden state, and lets the gate learn to open
        # only where it actually helps. Without this, the cross-attention
        # path was stuck at R²≈0.04 (predicting the constant training mean).
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized_hidden = self.query_norm(hidden)
        normalized_context = self.context_norm(context)
        branch_output, _ = self.attention(normalized_hidden, normalized_context, normalized_context, need_weights=False)
        return hidden + self.gate(branch_output)


class StripedHyenaBlock(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int, fusion_type: str = "cross_hyena"):
        super().__init__()
        short_len, _, _ = _resolve_hyena_filter_lengths(seq_len)
        if fusion_type == "cross_attention":
            self.cross_short = CrossAttentionResidualBranch(hidden_dim)
        elif fusion_type == "cross_hyena":
            self.cross_short = CrossHyenaResidualBranch(hidden_dim, seq_len, filter_len=short_len, long_mixer="conv")
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        self.hyena_block_1 = MultiScaleHyenaBlock(hidden_dim, seq_len, include_short=True)
        self.hyena_block_2 = MultiScaleHyenaBlock(hidden_dim, seq_len, include_short=False)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.cross_short(hidden, context)
        hidden = self.hyena_block_1(hidden)
        hidden = self.hyena_block_2(hidden)
        return self.output_norm(hidden)


class M5CQuerySequenceAtacCrossHyenaRegressorModelB(nn.Module):
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
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.sequence_proj = nn.Linear(sequence_dim, hidden_dim)
        self.atac_proj = nn.Linear(atac_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        self.atac_norm = nn.LayerNorm(hidden_dim)
        self.context_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim, seq_len) if use_positional_encoding else None
        self.blocks = nn.ModuleList([StripedHyenaBlock(hidden_dim, seq_len, fusion_type=fusion_type) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, m5c_track: torch.Tensor, sequence_track: torch.Tensor, atac_track: torch.Tensor) -> torch.Tensor:
        hidden = self.query_norm(self.query_proj(m5c_track))
        sequence_hidden = self.sequence_norm(self.sequence_proj(sequence_track))
        atac_hidden = self.atac_norm(self.atac_proj(atac_track))
        context = self.context_norm(self.context_proj(torch.cat([sequence_hidden, atac_hidden], dim=-1)))
        if self.position_encoding is not None:
            hidden = self.position_encoding(hidden)
            context = self.position_encoding(context)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.head(self.final_norm(hidden))


class M5CQuerySequenceAtacRnaCrossHyenaRegressorModelB(nn.Module):
    """Model B variant that accepts an optional RNA-coverage context track.

    Architecture mirrors ``M5CQuerySequenceAtacCrossHyenaRegressorModelB`` but
    the context is ``[sequence | atac | (rna?)]``.  When ``rna_dim == 0`` the
    RNA projection becomes a no-op and ``context_proj`` falls back to the
    original 2*hidden_dim input dim, making the class fully backward-compatible
    with the existing ``M5CQuerySequenceAtacCrossHyenaRegressorModelB`` shape.
    """

    def __init__(
        self,
        seq_len: int,
        query_dim: int = 1,
        sequence_dim: int = 4,
        atac_dim: int = 1,
        rna_dim: int = 1,
        hidden_dim: int = 64,
        use_positional_encoding: bool = False,
        num_blocks: int = 2,
        fusion_type: str = "cross_hyena",
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.sequence_proj = nn.Linear(sequence_dim, hidden_dim)
        self.atac_proj = nn.Linear(atac_dim, hidden_dim)
        self.rna_proj = nn.Linear(rna_dim, hidden_dim) if rna_dim > 0 else None
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        self.atac_norm = nn.LayerNorm(hidden_dim)
        self.rna_norm = nn.LayerNorm(hidden_dim) if rna_dim > 0 else None
        context_in_dim = 3 * hidden_dim if rna_dim > 0 else 2 * hidden_dim
        self.context_proj = nn.Linear(context_in_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim, seq_len) if use_positional_encoding else None
        self.blocks = nn.ModuleList([StripedHyenaBlock(hidden_dim, seq_len, fusion_type=fusion_type) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, m5c_track: torch.Tensor, sequence_track: torch.Tensor, atac_track: torch.Tensor, rna_track: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.query_norm(self.query_proj(m5c_track))
        sequence_hidden = self.sequence_norm(self.sequence_proj(sequence_track))
        atac_hidden = self.atac_norm(self.atac_proj(atac_track))
        context_parts = [sequence_hidden, atac_hidden]
        if rna_track is not None and self.rna_proj is not None:
            rna_hidden = self.rna_norm(self.rna_proj(rna_track))
            context_parts.append(rna_hidden)
        context = self.context_norm(self.context_proj(torch.cat(context_parts, dim=-1)))
        if self.position_encoding is not None:
            hidden = self.position_encoding(hidden)
            context = self.position_encoding(context)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.head(self.final_norm(hidden))


class MinimalHyenaRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        context_dim: int,
        hidden_dim: int = 64,
        use_positional_encoding: bool = False,
    ):
        super().__init__()
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim, seq_len) if use_positional_encoding else None
        self.backbone = HyenaLayer(hidden_dim, seq_len)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, context_track: torch.Tensor) -> torch.Tensor:
        hidden = self.context_proj(context_track)
        if self.position_encoding is not None:
            hidden = self.position_encoding(hidden)
        hidden = self.backbone(hidden)
        hidden = hidden + self.residual(hidden)
        hidden = self.norm(hidden)
        return self.head(hidden)


class MaskedTrackPretrainingModelB(nn.Module):
    """
    Self-supervised masked pretraining model with Model B architecture.

    Architecture (modeled after M5CQuerySequenceAtacCrossHyenaRegressorModelB):
      - query = 5mC_masked  → query_proj → query_norm
      - context = 5hmC_masked + ATAC_masked  → context_track_proj (shared per-track) → concat
        with sequence → context_proj → context_norm
      - Multiple StripedHyenaBlock blocks (cross-hyena + multi-scale hyena)
      - Three independent output heads (one per track)

    During pretraining, ~15% of CpG positions in each track are independently
    masked. The model learns to reconstruct the original values using cross-track
    and sequence context.
    """

    def __init__(
        self,
        seq_len: int,
        query_dim: int = 1,
        context_track_dim: int = 1,
        sequence_dim: int = 4,
        hidden_dim: int = 64,
        use_positional_encoding: bool = False,
        num_blocks: int = 4,
        fusion_type: str = "cross_hyena",
        num_context_tracks: int = 2,
    ):
        super().__init__()
        num_heads = 1 + num_context_tracks  # query (5mC) + each context track

        # Query branch: 5mC (masked)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)

        # Context track projections — one per context track
        self.context_track_projs = nn.ModuleList([
            nn.Linear(context_track_dim, hidden_dim) for _ in range(num_context_tracks)
        ])
        self.context_track_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_context_tracks)
        ])

        # Sequence projection
        self.sequence_proj = nn.Linear(sequence_dim, hidden_dim)
        self.sequence_norm = nn.LayerNorm(hidden_dim)

        # Context fusion: seq_hidden + all context track hiddens
        context_concat_dim = hidden_dim * (1 + num_context_tracks)
        self.context_proj = nn.Linear(context_concat_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)

        # Positional encoding
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim, seq_len) if use_positional_encoding else None

        # StripedHyenaBlock blocks
        self.blocks = nn.ModuleList([
            StripedHyenaBlock(hidden_dim, seq_len, fusion_type=fusion_type) for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # Independent output heads: one per track (query + context tracks)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            ) for _ in range(num_heads)
        ])

    def forward(self, track_tensors: list[torch.Tensor], sequence_tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            track_tensors: [query_track (5mC), *context_tracks], each (N, seq_len, 1)
            sequence_tensor: (N, seq_len, 4) one-hot DNA
        Returns:
            [pred_query, pred_context_1, ...], each (N, seq_len, 1)
        """
        query_track = track_tensors[0]
        context_tracks = track_tensors[1:]

        # Query: 5mC
        hidden = self.query_norm(self.query_proj(query_track))

        # Context tracks — each through independent projection + norm
        context_track_hidden = [
            norm(proj(track))
            for norm, proj, track in zip(self.context_track_norms, self.context_track_projs, context_tracks)
        ]

        # Sequence projection
        sequence_hidden = self.sequence_norm(self.sequence_proj(sequence_tensor))

        # Context fusion: concat(sequence_hidden, *context_track_hidden) → proj → norm
        context = self.context_norm(self.context_proj(torch.cat([sequence_hidden, *context_track_hidden], dim=-1)))

        # Positional encoding
        if self.position_encoding is not None:
            hidden = self.position_encoding(hidden)
            context = self.position_encoding(context)

        # StripedHyenaBlock blocks
        for block in self.blocks:
            hidden = block(hidden, context)

        hidden = self.final_norm(hidden)

        return [head(hidden) for head in self.heads]

# ---------------------------------------------------------------------------
# Ablation models: flexible query/context for ATAC ablation experiments
# ---------------------------------------------------------------------------

class FlexibleQueryRegressorModelB(nn.Module):
    """Flexible Model-B architecture supporting any combination of query/context inputs.

    Designed for ablation experiments. Allows the user to specify:
      - which modality plays the role of "query" (the track fed into the
        cross-attention/hyena branch)
      - which modalities play the role of "context" (concatenated with sequence)

    Forward signature: forward(query_track, sequence_track, context_tracks)
    where context_tracks is a list of additional context tensors (e.g. [atac]).

    The original M5CQuerySequenceAtacCrossHyenaRegressorModelB is equivalent to
    using query_track=m5c and context_tracks=[atac].
    """
    def __init__(
        self,
        seq_len: int,
        query_dim: int = 1,
        sequence_dim: int = 4,
        context_track_dim: int = 1,
        num_context_tracks: int = 1,
        hidden_dim: int = 64,
        use_positional_encoding: bool = False,
        num_blocks: int = 2,
        fusion_type: str = "cross_hyena",
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.sequence_proj = nn.Linear(sequence_dim, hidden_dim)
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        # Independent projections per context track (like MaskedTrackPretrainingModelB)
        self.context_track_projs = nn.ModuleList([
            nn.Linear(context_track_dim, hidden_dim) for _ in range(num_context_tracks)
        ])
        self.context_track_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_context_tracks)
        ])
        # Context fusion: seq + all context tracks
        context_concat_dim = hidden_dim * (1 + num_context_tracks)
        self.context_proj = nn.Linear(context_concat_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.position_encoding = (
            SinusoidalPositionalEncoding(hidden_dim, seq_len) if use_positional_encoding else None
        )
        self.blocks = nn.ModuleList([
            StripedHyenaBlock(hidden_dim, seq_len, fusion_type=fusion_type)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_track: torch.Tensor,
        sequence_track: torch.Tensor,
        context_tracks: list[torch.Tensor],
    ) -> torch.Tensor:
        hidden = self.query_norm(self.query_proj(query_track))
        sequence_hidden = self.sequence_norm(self.sequence_proj(sequence_track))
        context_track_hidden = [
            norm(proj(t))
            for norm, proj, t in zip(self.context_track_norms, self.context_track_projs, context_tracks)
        ]
        context = self.context_norm(
            self.context_proj(torch.cat([sequence_hidden, *context_track_hidden], dim=-1))
        )
        if self.position_encoding is not None:
            hidden = self.position_encoding(hidden)
            context = self.position_encoding(context)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.head(self.final_norm(hidden))
