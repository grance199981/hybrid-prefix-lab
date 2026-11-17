"""Background sampler of what NVML shows every user: GPU utilization,
memory-controller utilization, power and current clocks of one GPU, written as CSV
lines
  epoch_ns,util_gpu,util_memory,power_w,clock_sm_mhz,clock_mem_mhz
until it is terminated. A coarse cross-check for the plain runs of Q2 and Q2b
(started by scripts/run_probe.sh with PROBE_NVML=1). Needs pynvml.

Usage: nvml_sampler.py --uuid <gpu-uuid> --out file.csv [--hz 10]
"""
import argparse
import signal
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=10.0)
    a = ap.parse_args()
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByUUID(a.uuid)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    def clock(kind):          # current clock in MHz; -1 if the driver does not report it
        try:
            return pynvml.nvmlDeviceGetClockInfo(h, kind)
        except Exception:  # noqa: BLE001
            return -1

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("epoch_ns,util_gpu,util_memory,power_w,clock_sm_mhz,clock_mem_mhz\n")
        while not stop:
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            f.write(f"{time.time_ns()},{u.gpu},{u.memory},{p:.1f},"
                    f"{clock(pynvml.NVML_CLOCK_SM)},{clock(pynvml.NVML_CLOCK_MEM)}\n")
            f.flush()
            time.sleep(1.0 / a.hz)


if __name__ == "__main__":
    main()
