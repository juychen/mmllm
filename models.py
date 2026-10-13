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
        # Learnable [MASK] tokens for modality dropout (context tracks only).
        # When a track is absent, its whole hidden representation is replaced by
        # the expanded token so the model learns a "missing modality" embedding
        # instead of conflating absence with a low/zero track value.
        self.seq_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.atac_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.rna_mask_token = nn.Parameter(torch.zeros(hidden_dim)) if rna_dim > 0 else None

    def forward(
        self,
        m5c_track: torch.Tensor,
        sequence_track: torch.Tensor,
        atac_track: torch.Tensor,
        rna_track: torch.Tensor | None = None,
        sequence_present: bool = True,
        atac_present: bool = True,
        rna_present: bool = True,
    ) -> torch.Tensor:
        hidden = self.query_norm(self.query_proj(m5c_track))
        sequence_hidden = self.sequence_norm(self.sequence_proj(sequence_track))
        atac_hidden = self.atac_norm(self.atac_proj(atac_track))

        batch_size, length = atac_hidden.shape[0], atac_hidden.shape[1]
        if not sequence_present:
            sequence_hidden = self.seq_mask_token.view(1, 1, -1).expand(batch_size, length, -1)
        if not atac_present:
            atac_hidden = self.atac_mask_token.view(1, 1, -1).expand(batch_size, length, -1)

        context_parts = [sequence_hidden, atac_hidden]
        if rna_track is not None and self.rna_proj is not None:
            rna_hidden = self.rna_norm(self.rna_proj(rna_track))
            if not rna_present:
                rna_hidden = self.rna_mask_token.view(1, 1, -1).expand(batch_size, length, -1)
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
      - One independent output head per reconstructed track

    During pretraining, ~15% of CpG positions in each track are independently
    masked. The model learns to reconstruct the original values using cross-track
    and sequence context.

    Two optional extensions (both off by default → identical to the original
    class, state-dict compatible with existing checkpoints):

    * ``num_reconstructed_context_tracks``: only the first N context tracks get
      output heads; the remaining context tracks are input-only (no head, never
      in the loss). Enables e.g. 5hmC reconstructed while ATAC/RNA are just
      conditioning inputs.
    * ``use_mask_tokens=True``: learned [MASK] tokens for the two-layer masking
      scheme of ``run_masked_multimodal_pretraining.py``. Masking is done by
      SUBSTITUTING the projected hidden representation, never by zeroing the raw
      track values — 0 is a real methylation level, so zeroing would conflate
      "missing" with "unmethylated".
        - ``position_mask`` (N, L, 1), 1.0 = masked position: every signal
          track's hidden at those positions becomes that track's [POS-MASK]
          token.
        - ``track_present``: list of (N, 1, 1) multipliers (1.0 = present) in
          ``track_tensors`` order: an absent track's whole hidden becomes its
          [MODALITY-MASK] token.
        - ``sequence_present`` (N, 1, 1): whole-track [MASK] for the sequence.
      Anti-leakage: when the loss is only computed at ``position_mask``
      positions, no track's real value at those positions ever enters the
      network.
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
        num_reconstructed_context_tracks: int | None = None,
        use_mask_tokens: bool = False,
    ):
        super().__init__()
        if num_reconstructed_context_tracks is None:
            num_reconstructed_context_tracks = num_context_tracks
        num_heads = 1 + num_reconstructed_context_tracks  # query (5mC) + reconstructed context tracks
        self.num_context_tracks = num_context_tracks
        self.num_reconstructed_context_tracks = num_reconstructed_context_tracks
        self.use_mask_tokens = use_mask_tokens

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

        # Independent output heads: one per reconstructed track (query + first
        # `num_reconstructed_context_tracks` context tracks, in input order)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            ) for _ in range(num_heads)
        ])

        # Learned [MASK] tokens for the two-layer masking scheme. Created only
        # when requested so that the default state dict (and every existing
        # checkpoint) stays unchanged.
        if use_mask_tokens:
            self.query_mask_token = nn.Parameter(torch.zeros(hidden_dim))
            self.context_track_mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_context_tracks)
            ])
            self.query_pos_token = nn.Parameter(torch.zeros(hidden_dim))
            self.context_pos_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_context_tracks)
            ])
            self.sequence_mask_token = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        track_tensors: list[torch.Tensor],
        sequence_tensor: torch.Tensor,
        position_mask: torch.Tensor | None = None,
        track_present: list[torch.Tensor] | None = None,
        sequence_present: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            track_tensors: [query_track (5mC), *context_tracks], each (N, seq_len, 1)
            sequence_tensor: (N, seq_len, 4) one-hot DNA
            position_mask: optional (N, seq_len, 1), 1.0 = masked position —
                every track's hidden is replaced by its learned [POS-MASK] token
                there (requires use_mask_tokens=True).
            track_present: optional list of (N, 1, 1) presence multipliers in
                track_tensors order, 1.0 = present — an absent track's whole
                hidden is replaced by its learned [MODALITY-MASK] token.
            sequence_present: optional (N, 1, 1) presence multiplier for the
                sequence track.
        Returns:
            [pred_query, pred_context_1, ...], each (N, seq_len, 1) — one entry
            per output head (query + reconstructed context tracks).
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

        if not self.use_mask_tokens and (position_mask is not None or track_present is not None):
            raise ValueError(
                "position_mask/track_present require use_mask_tokens=True on this model."
            )
        if track_present is not None and len(track_present) != len(track_tensors):
            raise ValueError(
                f"track_present has {len(track_present)} entries but got {len(track_tensors)} track tensors."
            )

        # Layer 1 — shared position mask: substitute the per-track [POS-MASK]
        # token at masked positions (all signal tracks share the same positions).
        if position_mask is not None:
            keep = 1.0 - position_mask
            hidden = hidden * keep + self.query_pos_token.view(1, 1, -1) * position_mask
            context_track_hidden = [
                h * keep + token.view(1, 1, -1) * position_mask
                for h, token in zip(context_track_hidden, self.context_pos_tokens)
            ]

        # Layer 2 — per-track modality mask: substitute the whole track's hidden
        # with its [MODALITY-MASK] token for absent samples.
        if track_present is not None:
            q_keep = track_present[0]
            hidden = hidden * q_keep + self.query_mask_token.view(1, 1, -1) * (1.0 - q_keep)
            context_track_hidden = [
                h * keep + token.view(1, 1, -1) * (1.0 - keep)
                for h, keep, token in zip(
                    context_track_hidden, track_present[1:], self.context_track_mask_tokens
                )
            ]
        if sequence_present is not None:
            sequence_hidden = (
                sequence_hidden * sequence_present
                + self.sequence_mask_token.view(1, 1, -1) * (1.0 - sequence_present)
            )

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


class MultiQueryMaskedPretrainingModel(nn.Module):
    """Masked pretraining with N independent query streams sharing ONE context.

    Unlike :class:`MaskedTrackPretrainingModelB` (a single query trunk that all
    heads read), every query track (e.g. 5mC *and* 5hmC) gets its OWN projection
    + norm and its own pass through the shared StripedHyena blocks.

    ``context_mode`` controls what each query's fused context carries:

    * ``"shared"``     — DNA sequence + the non-query context tracks (e.g. ATAC,
      RNA), identical for every query. The query streams do NOT see each other.
    * ``"all-others"`` — DNA sequence + EVERY other signal track, i.e. the other
      query tracks *and* the context tracks. Each query sees a different context
      (everything except itself), so e.g. the 5hmC head keeps access to 5mC.

    Anti-leakage is preserved in both modes: at masked positions every signal
    track's hidden — including every query's — is replaced by that track's learned
    [POS-MASK] token, so no track's real value at a predicted position is ever
    visible, whether it is used as a trunk or as another query's context. Masking
    is always substitution in hidden space, never raw zeroing, because 0 is a real
    methylation level.

    Forward signature::

        forward(query_tracks, sequence_tensor, context_tracks,
                position_mask=None, query_present=None, track_present=None,
                sequence_present=None) -> list[Tensor]

    where each tensor in ``query_tracks`` / ``context_tracks`` is (N, seq_len, 1),
    ``sequence_tensor`` is (N, seq_len, 4) one-hot, ``position_mask`` is
    (N, seq_len, 1), and the presence tensors are (N, 1, 1). Returns one
    prediction (N, seq_len, 1) per query track, in ``query_tracks`` order.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int = 64,
        use_positional_encoding: bool = False,
        num_blocks: int = 4,
        fusion_type: str = "cross_hyena",
        num_queries: int = 2,
        num_context_tracks: int = 2,
        query_dim: int = 1,
        context_track_dim: int = 1,
        sequence_dim: int = 4,
        context_mode: str = "shared",
    ):
        super().__init__()
        if num_queries < 1:
            raise ValueError("num_queries must be >= 1")
        if context_mode not in ("shared", "all-others"):
            raise ValueError(f"unknown context_mode: {context_mode!r} (use 'shared' or 'all-others')")
        self.num_queries = num_queries
        self.num_context_tracks = num_context_tracks
        self.context_mode = context_mode
        # Number of signal tracks carried by each query's fused context.
        #   shared     : the non-query context tracks only (identical for every query)
        #   all-others : every other signal track, i.e. the other queries AND the context tracks
        if context_mode == "all-others":
            self.num_context_in_fusion = (num_queries - 1) + num_context_tracks
        else:
            self.num_context_in_fusion = num_context_tracks
        self.context_specs = self._build_context_specs()

        # One independent query stream per query track.
        self.query_projs = nn.ModuleList([
            nn.Linear(query_dim, hidden_dim) for _ in range(num_queries)
        ])
        self.query_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_queries)
        ])

        # Context tracks — one projection + norm per track.
        self.context_track_projs = nn.ModuleList([
            nn.Linear(context_track_dim, hidden_dim) for _ in range(num_context_tracks)
        ])
        self.context_track_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_context_tracks)
        ])

        # Sequence projection
        self.sequence_proj = nn.Linear(sequence_dim, hidden_dim)
        self.sequence_norm = nn.LayerNorm(hidden_dim)

        # Context fusion: seq_hidden + this query's context track hiddens.
        context_concat_dim = hidden_dim * (1 + self.num_context_in_fusion)
        self.context_proj = nn.Linear(context_concat_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)

        # Positional encoding
        self.position_encoding = SinusoidalPositionalEncoding(hidden_dim, seq_len) if use_positional_encoding else None

        # Shared StripedHyena blocks — every query stream runs through these.
        self.blocks = nn.ModuleList([
            StripedHyenaBlock(hidden_dim, seq_len, fusion_type=fusion_type)
            for _ in range(num_blocks)
        ])

        # Per-query output head (each query reconstructs itself).
        self.final_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_queries)])
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            ) for _ in range(num_queries)
        ])

        # Learned [MASK] tokens for the two-layer masking scheme.
        # Layer 1 — [POS-MASK]: per query / context track, substituted at masked positions.
        # Layer 2 — [MODALITY-MASK]: per query / context track, substituted for whole absent tracks.
        self.query_pos_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_queries)
        ])
        self.query_mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_queries)
        ])
        self.context_pos_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_context_tracks)
        ])
        self.context_track_mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_context_tracks)
        ])
        self.sequence_mask_token = nn.Parameter(torch.zeros(hidden_dim))

    def _apply_position_mask(self, hidden: torch.Tensor, token: nn.Parameter, position_mask: torch.Tensor) -> torch.Tensor:
        keep = 1.0 - position_mask
        return hidden * keep + token.view(1, 1, -1) * position_mask

    def _build_context_specs(self) -> list[list[int]]:
        """Per-query context track indices into the flat [*queries, *context] order.

        shared     -> every query uses the same non-query context tracks.
        all-others -> each query uses EVERY other signal track (other queries + context).
        """
        nq, nc = self.num_queries, self.num_context_tracks
        total = nq + nc
        if self.context_mode == "all-others":
            return [[j for j in range(total) if j != i] for i in range(nq)]
        return [[nq + k for k in range(nc)] for _ in range(nq)]

    def _fuse_context(self, sequence_hidden: torch.Tensor, track_hidden: list[torch.Tensor], indices: list[int]) -> torch.Tensor:
        parts = [sequence_hidden, *[track_hidden[j] for j in indices]]
        context = self.context_norm(self.context_proj(torch.cat(parts, dim=-1)))
        if self.position_encoding is not None:
            context = self.position_encoding(context)
        return context

    def forward(
        self,
        query_tracks: list[torch.Tensor],
        sequence_tensor: torch.Tensor,
        context_tracks: list[torch.Tensor],
        position_mask: torch.Tensor | None = None,
        query_present: list[torch.Tensor] | None = None,
        track_present: list[torch.Tensor] | None = None,
        sequence_present: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if len(query_tracks) != self.num_queries:
            raise ValueError(f"expected {self.num_queries} query tracks, got {len(query_tracks)}")
        if len(context_tracks) != self.num_context_tracks:
            raise ValueError(f"expected {self.num_context_tracks} context tracks, got {len(context_tracks)}")

        # ----- per-track hidden reps, flat order [*query_tracks, *context_tracks] -----
        # Each track gets its own proj+norm, its [POS-MASK] at masked positions and
        # its [MODALITY-MASK] when the whole track is absent. Query tracks are
        # projected once and reused both as their own trunk and (in 'all-others'
        # mode) as another query's context — never by re-deriving from raw values.
        track_hidden: list[torch.Tensor] = []
        for i, query_track in enumerate(query_tracks):
            h = self.query_norms[i](self.query_projs[i](query_track))
            if position_mask is not None:
                h = self._apply_position_mask(h, self.query_pos_tokens[i], position_mask)
            if query_present is not None:
                keep = query_present[i]
                h = h * keep + self.query_mask_tokens[i].view(1, 1, -1) * (1.0 - keep)
            track_hidden.append(h)
        for k, context_track in enumerate(context_tracks):
            h = self.context_track_norms[k](self.context_track_projs[k](context_track))
            if position_mask is not None:
                h = self._apply_position_mask(h, self.context_pos_tokens[k], position_mask)
            if track_present is not None:
                keep = track_present[k]
                h = h * keep + self.context_track_mask_tokens[k].view(1, 1, -1) * (1.0 - keep)
            track_hidden.append(h)

        sequence_hidden = self.sequence_norm(self.sequence_proj(sequence_tensor))
        if sequence_present is not None:
            sequence_hidden = (
                sequence_hidden * sequence_present
                + self.sequence_mask_token.view(1, 1, -1) * (1.0 - sequence_present)
            )

        # ----- one independent pass per query stream, each with its OWN context -----
        context_cache: dict[tuple, torch.Tensor] = {}
        preds = []
        for i in range(self.num_queries):
            key = tuple(self.context_specs[i])
            context = context_cache.get(key)
            if context is None:
                context = self._fuse_context(sequence_hidden, track_hidden, self.context_specs[i])
                context_cache[key] = context
            hidden = track_hidden[i]
            if self.position_encoding is not None:
                hidden = self.position_encoding(hidden)
            for block in self.blocks:
                hidden = block(hidden, context)
            preds.append(self.heads[i](self.final_norms[i](hidden)))
        return preds
