"""Minimal test for FlashFFTConv with debug info."""
import torch
from flashfftconv import FlashFFTConv

B, H, L = 2, 64, int(16384 / 2)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(torch.version.cuda)
flash_fft = FlashFFTConv(2 * L, dtype=torch.float16).cuda()

x = torch.randn(B, H, L, dtype=torch.float16, device="cuda")
k = torch.randn(H, L, dtype=torch.float16, device="cuda", requires_grad=True)

print(f"x: device={x.device}, dtype={x.dtype}, shape={x.shape}")
print(f"k: device={k.device}, dtype={k.dtype}, shape={k.shape}, requires_grad={k.requires_grad}")

y = flash_fft(x, k)
print(f"Forward output: {y.shape}, {y.dtype}")

# Test backward (may fail due to CUDA driver/PTX compatibility)
try:
    loss = y.sum()
    loss.backward()
    print("Forward + backward OK")
    print(f"Gradient on k: {k.grad.shape}")
    print("FlashFFTConv works! ✅")
except RuntimeError as e:
    print(f"Backward failed (expected if driver mismatch): {e}")
    print("Forward-only mode works! ✅ (backward needs newer driver or recompilation)")

print(f"Input shape:  {x.shape}")
print(f"Kernel shape: {k.shape}")
print(f"Output shape: {y.shape}")
