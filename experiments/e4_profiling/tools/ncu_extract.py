"""Pick the Speed-of-Light numbers out of a raw-page CSV exported from an Nsight
Compute report:
  ncu --import <case>.ncu-repz --csv --page raw > <case>.raw.csv

For every profiled kernel: name, duration, compute (SM) throughput and memory
throughput as a percentage of the hardware peak, DRAM throughput and bytes read
from device memory, L1 and L2 hit rates.

--medians writes, per case, the median and the range of each metric over the
kernels of the operation itself. ncu_probe.py fills its inputs with torch.randn,
whose kernels (at::...distribution_elementwise...) are captured too; they are
left out. make_report.py reads this file (--ncu).

Usage: ncu_extract.py decode.raw.csv prefill.raw.csv attention.raw.csv
           [--json ncu_summary.json] [--medians ncu_medians.json]
Standard library only.
"""
import argparse
import csv
import json
import os
import statistics

# metric name -> short label; the first column whose name ends with the metric is used
METRICS = [
    ("gpu__time_duration.sum", "duration"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "compute_pct"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "memory_pct"),
    ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "dram_pct"),
    ("dram__bytes_op_read.sum", "dram_read"),
    ("dram__bytes_op_read.sum.per_second", "dram_read_rate"),
    ("dram__bytes_op_write.sum", "dram_write"),
    ("l1tex__t_sector_hit_rate.pct", "l1_hit_pct"),
    ("lts__t_sector_hit_rate.pct", "l2_hit_pct"),
    ("sm__inst_executed.sum", "instructions"),
]
SCALE = {"": 1.0, "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
         "byte/s": 1.0, "Kbyte/s": 1e3, "Mbyte/s": 1e6, "Gbyte/s": 1e9, "Tbyte/s": 1e12,
         "ns": 1e-9, "us": 1e-6, "usecond": 1e-6, "ms": 1e-3, "msecond": 1e-3, "nsecond": 1e-9,
         "s": 1.0, "second": 1.0, "%": 1.0, "inst": 1.0}
INIT_KERNEL = "distribution_elementwise"     # torch.randn filling the inputs, not the operation


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    hdr, units, body = rows[0], rows[1], rows[2:]
    name_col = next(i for i, h in enumerate(hdr) if h in ("Function Name", "Kernel Name"))
    cols = {}
    for metric, label in METRICS:
        exact = [i for i, h in enumerate(hdr) if h == metric]
        tail = [i for i, h in enumerate(hdr) if h.endswith("." + metric) or h == metric]
        pick = exact or tail
        if pick:
            cols[label] = pick[0]
    out = []
    for r in body:
        k = {"name": r[name_col][:70]}
        for label, i in cols.items():
            try:
                k[label] = float(r[i]) * SCALE.get(units[i], 1.0)
            except ValueError:
                k[label] = None
            k[label + "_unit_seen"] = units[i]
        out.append(k)
    return out, {label: hdr[i] for label, i in cols.items()}


def medians(kernels, labels):
    """Median and [min, max] of each metric over the kernels of the operation."""
    ks = [k for k in kernels if INIT_KERNEL not in k["name"]]
    out = {"n_kernels": len(ks), "kernel": ks[0]["name"][:60] if ks else None}
    for label in labels:
        xs = [k[label] for k in ks if k.get(label) is not None]
        if label == "duration":                       # seconds, kept to the microsecond
            xs = [round(x, 6) for x in xs]
        if xs:
            m = statistics.median(xs)
            out[label] = round(m, 6) if label == "duration" else m
            out[label + "_minmax"] = [min(xs), max(xs)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--medians", default=None)
    a = ap.parse_args()
    res, med = {}, {}
    for path in a.csv:
        kernels, cols = load(path)
        case = os.path.basename(path).split(".")[0]
        res[case] = {"columns": cols, "kernels": [{k: v for k, v in x.items() if not k.endswith("_unit_seen")}
                                                  for x in kernels]}
        med[case] = medians(kernels, list(cols))
        print(f"== {case}: {len(kernels)} kernels; columns {cols}")
        for i, k in enumerate(kernels):
            def f(label, fmt, scale=1.0):
                v = k.get(label)
                return "   n/a" if v is None else fmt % (v * scale)
            print(f"  {i:2d} {k['name'][:34]:34s} {f('duration', '%8.3f ms', 1e3)}  SM {f('compute_pct', '%5.1f%%')}  "
                  f"mem {f('memory_pct', '%5.1f%%')}  dram {f('dram_pct', '%5.1f%%')}  "
                  f"read {f('dram_read', '%8.1f MB', 1e-6)}  {f('dram_read_rate', '%6.1f GB/s', 1e-9)}  "
                  f"L1 {f('l1_hit_pct', '%5.1f%%')}  L2 {f('l2_hit_pct', '%5.1f%%')}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(res, fh, indent=1)
    if a.medians:
        with open(a.medians, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(med, fh, indent=1)


if __name__ == "__main__":
    main()
