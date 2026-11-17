#!/usr/bin/env python
"""Communication test of two GPUs without vLLM (step 1 of the investigation).

Two processes, one GPU each, repeat all-reduce and all-gather through torch.distributed
(the NCCL library that vLLM loads) and check every result element by element. Each rank
can compute the payload of both ranks from (rank, step), so every result has an exact
expected value. The payload sizes are those of the two-GPU profiling workload (see SIZES).

Parent:  comm_test.py --out DIR --tag A --seconds 240 [--backend nccl|gloo]
                      [--inject-steps 2,7] [--expect ok|injected] [--sizes 64,1024]
Worker:  the same with --rank R --port P (started by the parent).

With --backend nccl rank r uses cuda:r; the caller sets CUDA_VISIBLE_DEVICES to the two
GPUs in rank order and the NCCL_* variables of the group. With --backend gloo both ranks
run on the CPU: the same loop and the same checks, used as the control of the checker.

--inject-steps: at these steps rank 1 sends a payload with one element off by one. Both
ranks must then report exactly one wrong element at exactly these steps (--expect injected).

Outputs in DIR: rank<r>.json (steps, mismatches, latency per operation and size, environment),
rank<r>.log, summary.json (status, NCCL transports named in the logs, totals).
Exit codes of the parent: 0 as expected, 10 mismatch, 11 error, 12 hang, 13 control failed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback

from hplab.stats import percentile

# Elements per payload (4 bytes each):
#   1,024        a small message
#   163,840      decode with 64 requests: 64 x hidden 2,560
#   1,280,000    prefill of 500 tokens: 500 x 2,560
#   4,861,952    all-gather of the logits of 64 requests: 64 x half of the vocabulary of Qwen3-4B
#   20,480,000   prefill of 8,000 tokens: 8,000 x 2,560
SIZES = (1_024, 163_840, 1_280_000, 4_861_952, 20_480_000)
DTYPES = ("float32", "int32")
MOD = 65_521          # payload values are below MOD, so sums are exact in float32
CHECK_EVERY = 10      # steps between two stop checks (a 1-element all-reduce)
MAX_RECORDS = 50
EXIT = {"ok": 0, "mismatch": 10, "error": 11, "hang": 12, "control_failed": 13}


def plan(step, sizes=SIZES):
    """Operation, payload size and dtype of a step."""
    k = len(sizes)
    return ("all_reduce" if (step // k) % 2 == 0 else "all_gather",
            sizes[step % k],
            DTYPES[(step // (2 * k)) % 2])


def coeffs(rank, step):
    """Payload element i of (rank, step) is (i * a + b) % MOD."""
    return (2 * step + 1 + 7919 * rank) % MOD, (31 * step + 17 * rank + 3) % MOD


def judge(ranks, rcs, timed_out, expect, inject_steps):
    """Status of a run and the exit code of the parent.

    ranks: the two result dicts (None where the file is missing); rcs: return codes.
    """
    if timed_out:
        status = "hang"
    elif any(rc != 0 for rc in rcs) or any(r is None or r.get("error") for r in ranks):
        status = "error"
    elif any(r["mismatches"] for r in ranks):
        status = "mismatch"
    else:
        status = "ok"
    if expect == "ok":
        return status, EXIT[status]
    want = sorted(inject_steps)
    found = status == "mismatch" and all(
        sorted(m["step"] for m in r["mismatches"]) == want
        and all(m["n_diff"] == 1 for m in r["mismatches"]) for r in ranks)
    return status, (0 if found else EXIT["control_failed"])


def worker(a):
    import datetime

    import torch
    import torch.distributed as dist

    rank = a.rank
    dev = torch.device(f"cuda:{rank}" if a.backend == "nccl" else "cpu")
    res = {"rank": rank, "backend": a.backend, "device": str(dev), "steps": 0, "mismatches": [],
           "error": None, "torch": torch.__version__,
           "env": {k: v for k, v in os.environ.items()
                   if k.startswith("NCCL_") or k == "CUDA_VISIBLE_DEVICES"}}
    lat = {}
    t_start = time.monotonic()

    def dump():
        res["seconds"] = round(time.monotonic() - t_start, 2)
        res["latency"] = [{"op": op, "n": n, "count": len(v), "median_ms": round(percentile(sorted(v), 0.5), 3),
                           "p90_ms": round(percentile(sorted(v), 0.9), 3)} for (op, n), v in sorted(lat.items())]
        res["payload_bytes"] = sum(n * 4 * len(v) for (_, n), v in lat.items())
        with open(os.path.join(a.out, f"rank{rank}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    try:
        if dev.type == "cuda":
            p = torch.cuda.get_device_properties(dev)
            res["gpu"] = {"name": p.name, "uuid": str(getattr(p, "uuid", "")),
                          "can_access_peer": bool(torch.cuda.can_device_access_peer(rank, 1 - rank))}
            res["nccl"] = [str(x) for x in torch.cuda.nccl.version()]
        dist.init_process_group(backend=a.backend, init_method=f"tcp://127.0.0.1:{a.port}", rank=rank,
                                world_size=2, timeout=datetime.timedelta(seconds=120))
        idx = {n: torch.arange(n, device=dev, dtype=torch.int64) for n in set(a.sizes)}
        flag = torch.zeros(1, dtype=torch.int32, device=dev)

        def payload(r, step, n, dtype):
            ca, cb = coeffs(r, step)
            return ((idx[n] * ca + cb) % MOD).to(dtype)

        def sync():
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        def check(got, want, step, op, n, dn, slot):
            if torch.equal(got, want):
                return
            ne = (got != want).view(-1)
            first = int(torch.nonzero(ne)[0].item())
            if len(res["mismatches"]) < MAX_RECORDS:
                res["mismatches"].append({
                    "step": step, "op": op, "n": n, "dtype": dn, "slot": slot,
                    "n_diff": int(ne.sum().item()), "first_index": first,
                    "got": got.view(-1)[first:first + 4].tolist(),
                    "want": want.view(-1)[first:first + 4].tolist()})

        step = 0
        beat = t_start
        while True:
            op, n, dn = plan(step, a.sizes)
            dtype = getattr(torch, dn)
            pay = [payload(0, step, n, dtype), payload(1, step, n, dtype)]
            send = pay[rank].clone()
            if rank == 1 and step in a.inject_steps:
                send[n // 2] += 1               # the wrong element goes out; `pay` stays as computed
            outs = [torch.empty_like(send), torch.empty_like(send)] if op == "all_gather" else None
            sync()
            t0 = time.perf_counter()
            if op == "all_reduce":
                dist.all_reduce(send, op=dist.ReduceOp.SUM)
            else:
                dist.all_gather(outs, send)
            sync()
            lat.setdefault((op, n), []).append((time.perf_counter() - t0) * 1e3)
            if op == "all_reduce":
                check(send, pay[0] + pay[1], step, op, n, dn, None)
            else:
                check(outs[0], pay[0], step, op, n, dn, 0)
                check(outs[1], pay[1], step, op, n, dn, 1)
            step += 1
            res["steps"] = step
            now = time.monotonic()
            if now - beat >= 30:
                beat = now
                print(f"[rank {rank}] step {step}, {now - t_start:.0f} s, "
                      f"mismatches {len(res['mismatches'])}", flush=True)
            if step % CHECK_EVERY == 0:
                stop = bool(res["mismatches"]) or now - t_start >= a.seconds \
                    or (a.max_steps > 0 and step >= a.max_steps)
                flag.fill_(1 if stop else 0)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if int(flag.item()) > 0:
                    break
        dist.destroy_process_group()
        dump()
        print(f"[rank {rank}] done: {step} steps, mismatches {len(res['mismatches'])}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001  (a CUDA or NCCL error ends the run; it is recorded, not retried)
        res["error"] = f"{type(e).__name__}: {e}"[:2000]
        res["traceback"] = traceback.format_exc()[-4000:]
        try:
            dump()
        except Exception:  # noqa: BLE001
            pass
        print(f"[rank {rank}] ERROR {res['error']}", flush=True)
        return 2


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def parent(a):
    os.makedirs(a.out, exist_ok=True)
    port = free_port()
    base = [sys.executable, os.path.abspath(__file__), "--out", a.out, "--tag", a.tag,
            "--seconds", str(a.seconds), "--backend", a.backend, "--port", str(port),
            "--max-steps", str(a.max_steps), "--sizes", ",".join(map(str, a.sizes)),
            "--inject-steps", ",".join(map(str, sorted(a.inject_steps)))]
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logs = [open(os.path.join(a.out, f"rank{r}.log"), "w", encoding="utf-8") for r in (0, 1)]
    procs = [subprocess.Popen(base + ["--rank", str(r)], stdout=logs[r], stderr=subprocess.STDOUT)
             for r in (0, 1)]
    print(f"[{a.tag}] started ranks with pids {[p.pid for p in procs]}", flush=True)
    t0 = time.monotonic()
    deadline = t0 + a.seconds + a.grace
    first_fail = None
    timed_out = False
    while True:
        rcs = [p.poll() for p in procs]
        if all(rc is not None for rc in rcs):
            break
        now = time.monotonic()
        if first_fail is None and any(rc not in (None, 0) for rc in rcs):
            first_fail = now            # one rank failed: the other gets 30 s to end by itself
        if now > deadline or (first_fail is not None and now - first_fail > 30):
            timed_out = first_fail is None
            for p in procs:             # by PID only
                if p.poll() is None:
                    p.terminate()
            time.sleep(10)
            for p in procs:
                if p.poll() is None:
                    p.kill()
            break
        time.sleep(0.5)
    rcs = [p.wait() for p in procs]
    for f in logs:
        f.close()
    ranks = []
    for r in (0, 1):
        try:
            with open(os.path.join(a.out, f"rank{r}.json"), encoding="utf-8") as f:
                ranks.append(json.load(f))
        except (OSError, ValueError):
            ranks.append(None)
    status, code = judge(ranks, rcs, timed_out, a.expect, a.inject_steps)
    transports = set()
    for r in (0, 1):
        with open(os.path.join(a.out, f"rank{r}.log"), encoding="utf-8", errors="replace") as f:
            transports.update(re.findall(r" via (\S+)", f.read()))
    summary = {
        "tag": a.tag, "status": status, "expect": a.expect, "exit_code": code, "return_codes": rcs,
        "timed_out": timed_out, "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "inject_steps": sorted(a.inject_steps), "transports": sorted(transports),
        "steps": [r["steps"] if r else None for r in ranks],
        "mismatch_records": [len(r["mismatches"]) if r else None for r in ranks],
        "errors": [r.get("error") if r else "no result file" for r in ranks],
        "payload_bytes": [r.get("payload_bytes") if r else None for r in ranks],
        "seconds": [r.get("seconds") if r else None for r in ranks],
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"[{a.tag}] status={status} expect={a.expect} exit={code} steps={summary['steps']} "
          f"mismatch_records={summary['mismatch_records']} transports={summary['transports']}", flush=True)
    return code


def ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--grace", type=float, default=180.0, help="seconds after --seconds before the run counts as hung")
    ap.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    ap.add_argument("--sizes", type=ints, default=list(SIZES))
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--inject-steps", type=ints, default=[])
    ap.add_argument("--expect", choices=("ok", "injected"), default="ok")
    ap.add_argument("--rank", type=int, default=-1)
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    a.inject_steps = set(a.inject_steps)
    sys.exit(worker(a) if a.rank >= 0 else parent(a))


if __name__ == "__main__":
    main()
