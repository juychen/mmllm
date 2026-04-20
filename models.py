import torch
import torch.nn as nn


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


class MinimalHyenaRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        context_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.context_proj = nn.Linear(context_dim, hidden_dim)
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
        hidden = self.backbone(hidden)
        hidden = hidden + self.residual(hidden)
        hidden = self.norm(hidden)
        return self.head(hidden)