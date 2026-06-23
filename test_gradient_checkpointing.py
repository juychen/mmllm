"""Quick toy test to compare memory: fp32 vs AMP (bfloat16), and checkpointing effect."""
import sys
import torch
import torch.nn as nn
from utils import get_freest_gpu

sys.path.insert(0, ".")

from models import M5CQuerySequenceAtacCrossHyenaRegressorModelB


def build_model(seq_len):
    return M5CQuerySequenceAtacCrossHyenaRegressorModelB(
        seq_len=seq_len,
        hidden_dim=64,
        num_blocks=2,
        fusion_type="cross_hyena",
    )


def run_one_forward(model, x1, x2, x3, target, use_amp=False):
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out = model(x1, x2, x3)
        loss = ((out - target) ** 2).mean()
    loss.backward()
    return loss


def test():
    device = torch.device(f"cuda:{get_freest_gpu()}" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    seq_len = 16384
    batch_size = 32

    x1 = torch.randn(batch_size, seq_len, 1, device=device)
    x2 = torch.randn(batch_size, seq_len, 4, device=device)
    x3 = torch.randn(batch_size, seq_len, 1, device=device)
    target = torch.randn(batch_size, seq_len, 1, device=device)

    # ── Test A: fp32 ──
    model_fp32 = build_model(seq_len).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    run_one_forward(model_fp32, x1, x2, x3, target, use_amp=False)
    mem_fp32 = torch.cuda.max_memory_allocated(device)
    print(f"\nfp32:  peak = {mem_fp32 / 1e9:.2f} GB")

    # ── Test B: AMP (bfloat16) ──
    model_amp = build_model(seq_len).to(device)
    model_amp.load_state_dict(model_fp32.state_dict())
    model_fp32.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)
    run_one_forward(model_amp, x1, x2, x3, target, use_amp=True)
    mem_amp = torch.cuda.max_memory_allocated(device)
    print(f"AMP:   peak = {mem_amp / 1e9:.2f} GB")
    print(f"Saved: {(mem_fp32 - mem_amp) / 1e9:.2f} GB ({(1 - mem_amp / mem_fp32) * 100:.1f}%)")

    # ── Test C: AMP + checkpointing ──
    import torch.utils.checkpoint as ckpt
    model_ckpt = build_model(seq_len).to(device)
    orig_forward = model_ckpt.forward
    def checkpointed_forward(m5c_track, sequence_track, atac_track):
        x = model_ckpt.query_norm(model_ckpt.query_proj(m5c_track))
        seq_h = model_ckpt.sequence_norm(model_ckpt.sequence_proj(sequence_track))
        atac_h = model_ckpt.atac_norm(model_ckpt.atac_proj(atac_track))
        ctx = model_ckpt.context_norm(model_ckpt.context_proj(torch.cat([seq_h, atac_h], dim=-1)))
        if model_ckpt.position_encoding is not None:
            x = model_ckpt.position_encoding(x)
            ctx = model_ckpt.position_encoding(ctx)
        for blk in model_ckpt.blocks:
            x = ckpt.checkpoint(blk, x, ctx, use_reentrant=False)
        return model_ckpt.head(model_ckpt.final_norm(x))
    model_ckpt.forward = checkpointed_forward
    model_ckpt.load_state_dict(model_fp32.state_dict())
    model_amp.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)
    run_one_forward(model_ckpt, x1, x2, x3, target, use_amp=True)
    mem_ckpt = torch.cuda.max_memory_allocated(device)
    print(f"AMP+ckpt: peak = {mem_ckpt / 1e9:.2f} GB")
    print(f"Saved: {(mem_fp32 - mem_ckpt) / 1e9:.2f} GB ({(1 - mem_ckpt / mem_fp32) * 100:.1f}%)")

    # ── Test D: Gradient correctness (AMP vs AMP+ckpt) ──
    print("\n=== Gradient Correctness (AMP vs AMP+ckpt) ===")

    def make_amp_model(use_checkpoint: bool):
        m = build_model(seq_len).to(device)
        if use_checkpoint:
            import torch.utils.checkpoint as ckpt
            orig_fwd = m.forward
            def ckpt_fwd(m5c_track, sequence_track, atac_track):
                x = m.query_norm(m.query_proj(m5c_track))
                seq_h = m.sequence_norm(m.sequence_proj(sequence_track))
                atac_h = m.atac_norm(m.atac_proj(atac_track))
                ctx = m.context_norm(m.context_proj(torch.cat([seq_h, atac_h], dim=-1)))
                if m.position_encoding is not None:
                    x = m.position_encoding(x)
                    ctx = m.position_encoding(ctx)
                for blk in m.blocks:
                    x = ckpt.checkpoint(blk, x, ctx, use_reentrant=False)
                return m.head(m.final_norm(x))
            m.forward = ckpt_fwd
        return m

    torch.manual_seed(0)
    model_a = make_amp_model(use_checkpoint=False)
    seed_state = model_a.state_dict()

    torch.manual_seed(0)
    model_b = make_amp_model(use_checkpoint=True)
    model_b.load_state_dict(seed_state)

    x1b = torch.randn(2, seq_len, 1, device=device)
    x2b = torch.randn(2, seq_len, 4, device=device)
    x3b = torch.randn(2, seq_len, 1, device=device)
    targetb = torch.randn(2, seq_len, 1, device=device)

    out_a = model_a(x1b, x2b, x3b)
    loss_a = ((out_a - targetb) ** 2).mean()
    loss_a.backward()
    grad_a = {k: v.grad.clone() for k, v in model_a.named_parameters() if v.grad is not None}

    out_b = model_b(x1b, x2b, x3b)
    loss_b = ((out_b - targetb) ** 2).mean()
    loss_b.backward()
    grad_b = {k: v.grad.clone() for k, v in model_b.named_parameters() if v.grad is not None}

    all_close = True
    for key in grad_a:
        if not torch.allclose(grad_a[key], grad_b[key], atol=1e-5):
            print(f"  MISMATCH: {key}")
            print(f"    max diff = {(grad_a[key] - grad_b[key]).abs().max().item():.2e}")
            all_close = False

    if all_close:
        print("  All gradients match (atol=1e-5) ✅")
        print(f"  Forward outputs match: {torch.allclose(out_a, out_b, atol=1e-5)}")
    else:
        print("  ❌ Gradients differ!")


if __name__ == "__main__":
    test()
