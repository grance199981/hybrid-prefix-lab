"""Q2: measure the roofline of the GPU for the kind of matrix multiply an LLM
forward pass is made of, by changing only the number of tokens n. Needs torch and a
GPU; run before any server starts (scripts/run_probe.sh with PROBE_SCALING=1).

One "pass" multiplies an (n x d_in) activation by K different weight matrices
(d_out x d_in), K chosen so the weights add up to several GiB, the way a forward
pass walks through every layer's weights once. The weights are far larger than
any on-chip cache, so each pass has to read them from device memory again.

  work  per pass = 2 * n * d_in * d_out * K floating point operations
  bytes per pass = K * d_in * d_out * bytes_per_element   (weights only)

If the time per pass does not change with n, the pass is limited by reading the
weights (memory bound); if it grows in proportion to n, it is limited by
arithmetic (compute bound). The n where the two meet is the ridge of the roofline
("knee": where the time first reaches twice the plateau). The same sweep with
fp32 weights doubles the bytes at equal work.

While each point runs, a thread samples what the driver exposes to every user
through NVML (GPU utilization, memory-controller utilization, power, SM clock):
a coarse cross-check, not a per-kernel counter. The GPU is the one named by the
environment variable GATE_GPU_UUID (else device 0).

Output: JSON with, per dtype, every point (ms per pass, TFLOP/s, weight GB/s, NVML
means) and the plateau, knee and best TFLOP/s.

Usage: q2_scaling.py --out scaling.json [--dtypes bf16 fp32] [--ns 1 2 4 ...]
"""
import argparse
import json
import os
import statistics as st
import threading
import time

import torch

DTYPES = {"bf16": (torch.bfloat16, 2), "fp32": (torch.float32, 4)}


class NvmlSampler:
    """Samples utilization.gpu, utilization.memory and power at a fixed rate."""

    def __init__(self, uuid=None, hz=20.0):
        self.rows, self.error, self._stop, self._thread = [], None, threading.Event(), None
        self.period = 1.0 / hz
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid else \
                pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # noqa: BLE001 - reported, the sweep still runs
            self.error = repr(e)[:200]

    def _run(self):
        while not self._stop.is_set():
            try:
                u = self.nv.nvmlDeviceGetUtilizationRates(self.h)
                p = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                try:              # current SM clock in MHz; None if not reported
                    ck = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
                except Exception:  # noqa: BLE001
                    ck = None
                self.rows.append((time.time_ns(), u.gpu, u.memory, p, ck))
            except Exception as e:  # noqa: BLE001
                self.error = repr(e)[:200]
                return
            self._stop.wait(self.period)

    def start(self):
        if self.error is None:
            self.rows = []
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        rows = self.rows[len(self.rows) // 3:]        # drop the ramp-up third
        if not rows:
            return {"n_samples": 0, "error": self.error}
        cks = [r[4] for r in rows if r[4] is not None]
        return {"n_samples": len(rows), "util_gpu": round(st.mean(r[1] for r in rows), 1),
                "util_memory": round(st.mean(r[2] for r in rows), 1),
                "power_w": round(st.mean(r[3] for r in rows), 1),
                "clock_sm_mhz": round(st.mean(cks), 1) if cks else None,
                "clock_sm_min_mhz": min(cks) if cks else None, "error": self.error}


def knee(points, factor=2.0):
    """points: [(n, ms)] sorted by n. Plateau = median time of the three smallest
    n. Returns (plateau_ms, n at which the time first reaches factor * plateau,
    by linear interpolation between the neighbouring points)."""
    plateau = st.median(ms for _, ms in points[:3])
    target = factor * plateau
    for (n0, t0), (n1, t1) in zip(points, points[1:]):
        if t0 < target <= t1:
            return plateau, n0 + (n1 - n0) * (target - t0) / (t1 - t0)
    return plateau, None


def sweep(name, d_in, d_out, gib, ns, min_seconds, reps, sampler):
    dtype, nbytes = DTYPES[name]
    K = max(1, int(gib * 2 ** 30 / (d_in * d_out * 2)))      # same K for both dtypes
    Ws = [torch.randn(d_out, d_in, device="cuda", dtype=dtype) for _ in range(K)]
    weight_bytes = K * d_in * d_out * nbytes
    out = []
    for n in ns:
        x = torch.randn(n, d_in, device="cuda", dtype=dtype)

        def one_pass():
            for W in Ws:
                torch.nn.functional.linear(x, W)

        for _ in range(2):
            one_pass()
        torch.cuda.synchronize()
        times = []
        sampler.start()
        for _ in range(reps):
            passes, t0 = 0, time.perf_counter()
            while True:
                one_pass()
                passes += 1
                if passes % 4 == 0 or n >= 1024:
                    torch.cuda.synchronize()
                    if time.perf_counter() - t0 >= min_seconds / reps:
                        break
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - t0) / passes)
        nv = sampler.stop()
        ms = st.median(times)
        flops = 2.0 * n * d_in * d_out * K
        out.append({"dtype": name, "n": n, "ms_per_pass": round(ms, 4),
                    "ms_min_max": [round(min(times), 4), round(max(times), 4)],
                    "tflops": round(flops / ms * 1e3 / 1e12, 3),
                    "weights_gb_per_s": round(weight_bytes / ms * 1e3 / 1e9, 1), "nvml": nv})
        print(f"{name} n={n:5d} {ms:9.3f} ms/pass  {out[-1]['tflops']:8.2f} TFLOP/s  "
              f"{out[-1]['weights_gb_per_s']:7.1f} GB/s  nvml={nv}", flush=True)
    del Ws
    torch.cuda.empty_cache()
    return {"K": K, "weight_bytes": weight_bytes, "points": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-in", type=int, default=2560)
    ap.add_argument("--d-out", type=int, default=9216)
    ap.add_argument("--gib", type=float, default=4.0, help="bf16 weight bytes per pass")
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 1024, 2048, 4096])
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp32"], choices=sorted(DTYPES))
    ap.add_argument("--min-seconds", type=float, default=2.4, help="timed duration per point")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)}", flush=True)
    sampler = NvmlSampler(os.environ.get("GATE_GPU_UUID"))
    res = {"d_in": a.d_in, "d_out": a.d_out, "ns": a.ns, "nvml_error": sampler.error, "sweeps": {}}
    for name in a.dtypes:
        sw = sweep(name, a.d_in, a.d_out, a.gib, a.ns, a.min_seconds, a.reps, sampler)
        pts = [(p["n"], p["ms_per_pass"]) for p in sw["points"]]
        plateau, k = knee(pts)
        sw.update(plateau_ms=round(plateau, 4), knee_n=None if k is None else round(k, 1),
                  plateau_gb_per_s=round(sw["weight_bytes"] / plateau * 1e3 / 1e9, 1),
                  best_tflops=max(p["tflops"] for p in sw["points"]))
        res["sweeps"][name] = sw
        print(f"== {name}: plateau {sw['plateau_ms']} ms ({sw['plateau_gb_per_s']} GB/s of weights), "
              f"knee n={sw['knee_n']}, best {sw['best_tflops']} TFLOP/s", flush=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
