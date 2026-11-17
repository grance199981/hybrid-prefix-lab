"""A few kernels of the kind an LLM forward pass is made of, small enough to
run under Nsight Compute on a laptop GPU (ncu replays every kernel it profiles,
so it is pointed at single kernels, not at a server).

  decode-like   y = x W^T with 1 token:    reads all of W, almost no arithmetic
  prefill-like  y = x W^T with 1024 tokens: the same bytes, 1024 times the arithmetic
  attention     scaled dot-product attention over 2048 tokens, causal

Each case runs --warmup times untimed and then --iters times, so `ncu
--launch-skip` / `--launch-count` can pick the steady-state launches. Without
ncu the script prints wall-clock numbers, to compare with the profiler.

Example (after installing Nsight Compute; PowerShell, administrator if the
counters are restricted):
  ncu --target-processes all --set full -o probe python ncu_probe.py --case decode --iters 3

run_ncu_local.ps1 runs the three cases under ncu. Needs torch and a CUDA GPU.
"""
import argparse
import time

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["decode", "prefill", "attention", "all"], default="all")
    ap.add_argument("--d-in", type=int, default=2560)
    ap.add_argument("--d-out", type=int, default=9216)
    ap.add_argument("--gib", type=float, default=1.0, help="weight bytes read per pass (bf16)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)}", flush=True)
    K = max(1, int(a.gib * 2 ** 30 / (a.d_in * a.d_out * 2)))
    Ws = [torch.randn(a.d_out, a.d_in, device=dev, dtype=dt) for _ in range(K)]
    wbytes = K * a.d_in * a.d_out * 2

    def linear(n):
        x = torch.randn(n, a.d_in, device=dev, dtype=dt)
        return lambda: [F.linear(x, W) for W in Ws]

    q = torch.randn(1, 16, 2048, 128, device=dev, dtype=dt)
    cases = {"decode": (linear(1), 2.0 * 1 * a.d_in * a.d_out * K, wbytes),
             "prefill": (linear(1024), 2.0 * 1024 * a.d_in * a.d_out * K, wbytes),
             "attention": (lambda: F.scaled_dot_product_attention(q, q, q, is_causal=True),
                           4.0 * 128 * 16 * 2048 * 2049 / 2, 3 * q.numel() * 2)}
    for name, (fn, flops, nbytes) in cases.items():
        if a.case not in ("all", name):
            continue
        for _ in range(a.warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            fn()
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / a.iters
        print(f"{name:9s} {ms:9.3f} ms/pass  {flops / ms * 1e3 / 1e12:8.2f} TFLOP/s  "
              f"{nbytes / ms * 1e3 / 1e9:8.1f} GB/s of inputs read", flush=True)


if __name__ == "__main__":
    main()
