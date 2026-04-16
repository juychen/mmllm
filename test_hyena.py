# Multimodal StripedHyena: adds per-position ATAC-seq and methylation features
# as additive embeddings fused into the token embedding before Hyena/attention blocks.
#
# Input shapes (all [batch, seq_len, feature_dim]):
#   atac        – float, shape [B, L, 1], ATAC-seq accessibility signal per position
#   methylation – float, shape [B, L, 1], CpG beta value (0-1) per position
#
# Both tensors are optional; pass None if a modality is absent.

import torch
import torch.nn as nn

from vortex.model.model import StripedHyena
import random


class MultiModalStripedHyena(StripedHyena):
    """StripedHyena extended with ATAC-seq and methylation feature inputs.

    Architecture:
        token_ids  ──► VocabParallelEmbedding ──┐
        atac       ──► ATACEncoder              ─┼─► add ──► Hyena/Attn blocks ──► unembed
        methylation──► MethylEncoder            ─┘

    The extra features are projected to `hidden_size` with a small two-layer MLP
    and added element-wise to the token embedding.  The Hyena backbone (blocks,
    norm, unembed) is unchanged, so you can load a pretrained StripedHyena/Evo2
    checkpoint and fine-tune with the new heads.  Only `atac_encoder` /
    `methyl_encoder` weights need to be initialised from scratch.
    """

    def __init__(self, config):
        force_single_device = config.get("force_single_device", True)
        original_device_count = torch.cuda.device_count

        if torch.cuda.is_available() and force_single_device:
            torch.cuda.device_count = lambda: 1

        try:
            super().__init__(config)
        finally:
            torch.cuda.device_count = original_device_count

        hidden = config.hidden_size
        feature_device = torch.device(self.block_idx_to_device[0])

        # Small MLP: scalar → hidden/4 → hidden  (keeps feature influence proportional)
        def _feature_encoder(in_dim: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden // 4, bias=True),
                nn.SiLU(),
                nn.Linear(hidden // 4, hidden, bias=False),
            )

        self.atac_encoder = _feature_encoder()
        self.methyl_encoder = _feature_encoder()
        self.atac_encoder.to(feature_device)
        self.methyl_encoder.to(feature_device)

        # Initialise feature encoders with small weights so they don't disturb
        # a loaded pretrained backbone at the start of fine-tuning.
        for enc in (self.atac_encoder, self.methyl_encoder):
            nn.init.normal_(enc[0].weight, std=0.02)
            nn.init.zeros_(enc[0].bias)
            nn.init.zeros_(enc[2].weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _backbone_dtype(self):
        for module in getattr(self, "blocks", []):
            for parameter in module.parameters():
                return parameter.dtype
        return self.embedding_layer.weight.dtype

    def forward(
        self,
        x,
        atac=None,
        methylation=None,
        inference_params_dict=None,
        padding_mask=None,
    ):
        """
        Args:
            x:            LongTensor [B, L]  – token ids (DNA sequence)
            atac:         FloatTensor [B, L, 1] or None  – ATAC signal per position
            methylation:  FloatTensor [B, L, 1] or None  – methylation beta per position
            inference_params_dict: passed straight through to Hyena engine
            padding_mask: passed straight through to Hyena engine
        """
        # 1. Token embedding  [B, L] → [B, L, hidden]
        x = self.embedding_layer(x)

        # 2. Fuse optional per-position features
        if atac is not None:
            # Match the feature encoder parameter dtype, then cast back to the
            # backbone activation dtype before fusion.
            atac_encoded = self.atac_encoder(
                atac.to(dtype=self.atac_encoder[0].weight.dtype)
            )
            x = x + atac_encoded.to(dtype=x.dtype)

        if methylation is not None:
            methyl_encoded = self.methyl_encoder(
                methylation.to(dtype=self.methyl_encoder[0].weight.dtype)
            )
            x = x + methyl_encoded.to(dtype=x.dtype)

        x = x.to(dtype=self._backbone_dtype())

        # 3. Run Hyena / attention blocks (same as StripedHyena.forward,
        #    but embedding has already been applied above)
        if inference_params_dict is not None:
            x, inference_params_dict_out = self.stateful_forward(
                x, inference_params_dict=inference_params_dict
            )
        else:
            x, inference_params_dict_out = self.stateless_forward(x, padding_mask=padding_mask)

        # 4. Final norm + unembed
        x = x.to(self.block_idx_to_device[0])
        x = self.norm(x)
        x = self.unembed(x)
        return x, inference_params_dict_out


# ------------------------------------------------------------------
# Loading a pretrained Evo2/StripedHyena checkpoint and resuming fine-tuning
# ------------------------------------------------------------------
#
# from vortex.model.utils import load_config        # or however you load ModelConfig
# from vortex.model.multimodal_model import MultiModalStripedHyena
#
# config = ...  # same config as the pretrained model
# model = MultiModalStripedHyena(config)
#
# # Load backbone weights, ignoring the new encoder layers (strict=False)
# ckpt = torch.load("evo2_7b.pt", map_location="cpu")
# missing, unexpected = model.load_state_dict(ckpt, strict=False)
# # missing will include atac_encoder.* and methyl_encoder.* — that's expected
#
# # Fine-tune: freeze backbone, train only the new feature encoders
# for name, p in model.named_parameters():
#     if "atac_encoder" not in name and "methyl_encoder" not in name:
#         p.requires_grad_(False)
from vortex.model.utils import dotdict


DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}
DNA_ALPHABET = tuple(DNA_VOCAB.keys())


def build_toy_config():
    return dotdict(
        {
            "model_name": "toy-multimodal-hyena",
            "vocab_size": 4,
            "hidden_size": 64,
            "num_filters": 64,
            "attn_layer_idxs": [],
            # Keep the toy stack numerically stable under random initialization.
            # The long-conv HCL block tends to blow up to NaN in this tiny setup.
            "hcl_layer_idxs": [],
            "hcm_layer_idxs": [],
            "hcs_layer_idxs": [0, 1, 2],
            "hcm_filter_length": 16,
            "hcl_filter_groups": 64,
            "hcm_filter_groups": 8,
            "hcs_filter_groups": 8,
            "hcs_filter_length": 5,
            "num_layers": 3,
            "short_filter_length": 3,
            "num_attention_heads": 8,
            "short_filter_bias": False,
            "mlp_init_method": torch.nn.init.normal_,
            "mlp_output_init_method": torch.nn.init.normal_,
            "eps": 1e-5,
            "state_size": 8,
            "rotary_emb_base": 10000,
            "make_vocab_size_divisible_by": 1,
            "inner_size_multiple_of": 8,
            "inner_mlp_size": 128,
            "log_intermediate_values": False,
            "proj_groups": 1,
            "hyena_filter_groups": 1,
            "column_split_hyena": False,
            "column_split": True,
            "interleave": True,
            "evo2_style_activations": True,
            "model_parallel_size": 1,
            "pipe_parallel_size": 1,
            "tie_embeddings": True,
            "mha_out_proj_bias": True,
            "hyena_out_proj_bias": True,
            "hyena_flip_x1x2": False,
            "qkv_proj_bias": False,
            "use_fp8_input_projections": False,
            "max_seqlen": 100,
            "max_batch_size": 1,
            "final_norm": True,
            "use_flash_attn": False,
            "use_flash_rmsnorm": False,
            "use_flash_depthwise": False,
            "use_flashfft": False,
            "use_laughing_hyena": False,
            "inference_mode": False,
            "tokenizer_type": "CharLevelTokenizer",
            "params_dtype": torch.float32,
            "force_single_device": True,
        }
    )


def random_dna_sequence(length=100):
    return "".join(random.choice(DNA_ALPHABET) for _ in range(length))


def encode_dna(sequence):
    return torch.tensor([[DNA_VOCAB[base] for base in sequence]], dtype=torch.long)


def make_random_features(length=100):
    atac = torch.rand(1, length, 1, dtype=torch.float32)
    methylation = torch.rand(1, length, 1, dtype=torch.float32)
    return atac, methylation


def main():
    torch.manual_seed(7)
    random.seed(7)

    config = build_toy_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sequence = random_dna_sequence(length=100)
    input_ids = encode_dna(sequence).to(device)
    atac, methylation = make_random_features(length=100)
    atac = atac.to(device)
    methylation = methylation.to(device)

    model = MultiModalStripedHyena(config).to(device)
    model.eval()

    with torch.no_grad():
        logits, _ = model(input_ids, atac=atac, methylation=methylation)

    print("DNA sequence:")
    print(sequence)
    print()
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"atac shape:      {tuple(atac.shape)}")
    print(f"methylation shape: {tuple(methylation.shape)}")
    print(f"logits shape:    {tuple(logits.shape)}")
    print()
    print("First 5 positions, logits over A/C/G/T:")
    print(logits[0, :5])


if __name__ == "__main__":
    main()