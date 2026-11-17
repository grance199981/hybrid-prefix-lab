#!/usr/bin/env python
"""A reproducer of the two-GPU fault that does not use vLLM (steps 7 to 9 of the
investigation).

Two processes, one GPU each. Each one calls NCCL through ctypes (the libnccl.so.2 that
ships with torch), records one "decode step" into a CUDA graph and replays it:

  inside the graph, LAYERS times:  matmul, matmul, all-reduce, matmul, matmul, all-reduce
  after the graph (eager):         matmul to "logits", all-gather of the logits

The sizes are those of Qwen3-4B with tensor parallelism over two GPUs and 64 concurrent
requests (hidden 2,560, MLP half 4,864, 36 layers, half of the vocabulary 75,968, bfloat16).
A run is GROUPS groups of STEPS steps with a pause between groups, like the profiling
client. Nothing here imports vLLM.

Parent:  graph_repro.py --out DIR [--groups 60 --steps 128 --layers 36 --batch 64]
         starts the two ranks, waits for DIR/go before the measured part (the launcher
         starts the profiler collection in between), watches the progress of the ranks
         and ends them by PID when one stops for --stall seconds.
Worker:  the same with --rank R (started by the parent).

Checks in every step: the all-gather must return this rank's own logits unchanged in its
own slot, the other slot must be finite, and both ranks must hold the same hidden state
after the graph (a 2-element all-reduce of a checksum, MAX and MIN must agree).

--async-output 1 hands the result of a step to a second thread the way vLLM does with
asynchronous scheduling (vllm/v1/worker/gpu/async_utils.py, AsyncOutput): the "sampled
tokens" and the check flags are copied to the CPU on a second CUDA stream (non-blocking,
after wait_stream on the main stream), a blocking CUDA event is recorded on that stream,
and a second thread waits for the event while the main thread already replays the next
step. There is then no device-wide synchronize inside a group. The tokens of every step
must equal those of the first step, because every step computes the same thing. Only this
version faulted under Nsight Systems 2025.1.3 with --cuda-graph-trace=node.
--inject-step N: rank 1 adds 1 to its checksum at step N, so both ranks must report it.

Outputs in DIR: rank<r>.json, rank<r>.log, progress<r>.txt, summary.json.
Exit codes of the parent: 0 as expected, 10 mismatch, 11 error, 12 hang, 13 control failed.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback

EXIT = {"ok": 0, "mismatch": 10, "error": 11, "hang": 12, "control_failed": 13}
NCCL_BF16, NCCL_F32, NCCL_SUM, NCCL_MAX = 9, 7, 0, 2    # values checked against vLLM's pynccl_wrapper.py


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def judge(ranks, rcs, stalled, expect, inject_step):
    """Status of a run and the exit code of the parent (ranks: result dicts or None)."""
    if stalled:
        status = "hang"
    elif any(rc != 0 for rc in rcs) or any(r is None or r.get("error") for r in ranks):
        status = "error"
    elif any(r["mismatches"] for r in ranks):
        status = "mismatch"
    else:
        status = "ok"
    if expect == "ok":
        return status, EXIT[status]
    found = status == "mismatch" and all(
        [m["step"] for m in r["mismatches"]] == [inject_step] and r["mismatches"][0]["kind"] == "hidden"
        for r in ranks)
    return status, (0 if found else EXIT["control_failed"])


class Nccl:
    """The few NCCL calls of this test, on torch's current stream."""

    def __init__(self, torch):
        self.torch = torch
        path = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia", "nccl", "lib", "libnccl.so.2")
        self.lib = ctypes.CDLL(path)
        self.path = path
        v = ctypes.c_void_p
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p
        self.lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        self.lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(UniqueId)]
        self.lib.ncclCommInitRank.argtypes = [ctypes.POINTER(v), ctypes.c_int, UniqueId, ctypes.c_int]
        self.lib.ncclAllReduce.argtypes = [v, v, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, v, v]
        self.lib.ncclAllGather.argtypes = [v, v, ctypes.c_size_t, ctypes.c_int, v, v]
        self.lib.ncclCommDestroy.argtypes = [v]
        self.comm = v()

    def check(self, rc):
        if rc != 0:
            raise RuntimeError("NCCL error: " + self.lib.ncclGetErrorString(rc).decode())

    def version(self):
        n = ctypes.c_int()
        self.check(self.lib.ncclGetVersion(ctypes.byref(n)))
        return n.value

    def new_id(self):
        u = UniqueId()
        self.check(self.lib.ncclGetUniqueId(ctypes.byref(u)))
        return bytes(bytearray(u.internal))

    def init(self, raw_id, rank):
        u = UniqueId()
        ctypes.memmove(u.internal, raw_id, 128)
        self.check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), 2, u, rank))

    def _stream(self):
        return ctypes.c_void_p(self.torch.cuda.current_stream().cuda_stream)

    def all_reduce(self, src, dst, dtype, op=NCCL_SUM):
        self.check(self.lib.ncclAllReduce(ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()),
                                          src.numel(), dtype, op, self.comm, self._stream()))

    def all_gather(self, src, dst, dtype):
        self.check(self.lib.ncclAllGather(ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()),
                                          src.numel(), dtype, self.comm, self._stream()))


def worker(a):
    import torch

    rank = a.rank
    dev = torch.device(f"cuda:{rank}")
    res = {"rank": rank, "steps": 0, "groups": 0, "mismatches": [], "error": None, "torch": torch.__version__,
           "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_") or k == "CUDA_VISIBLE_DEVICES"}}
    out = lambda name: os.path.join(a.out, name)  # noqa: E731
    t_begin = time.monotonic()

    def dump():
        res["seconds"] = round(time.monotonic() - t_begin, 2)
        with open(out(f"rank{rank}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    def progress(text):
        with open(out(f"progress{rank}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    try:
        torch.cuda.set_device(dev)
        p = torch.cuda.get_device_properties(dev)
        res["gpu"] = {"name": p.name, "uuid": str(getattr(p, "uuid", ""))}
        nccl = Nccl(torch)
        res["nccl_version"] = nccl.version()
        res["nccl_path"] = nccl.path
        id_path = out("nccl_id.bin")
        if rank == 0:
            raw = nccl.new_id()
            with open(id_path + ".tmp", "wb") as f:
                f.write(raw)
            os.replace(id_path + ".tmp", id_path)
        else:
            t0 = time.monotonic()
            while not os.path.exists(id_path):
                if time.monotonic() - t0 > 120:
                    raise RuntimeError("rank 0 did not publish the NCCL id")
                time.sleep(0.1)
            raw = open(id_path, "rb").read()
        nccl.init(raw, rank)

        B, H, M, V, L = a.batch, a.hidden, a.mlp, a.vocab, a.layers
        bf = torch.bfloat16
        g = torch.Generator(device="cpu").manual_seed(1000 + rank)

        def rnd(*shape, scale):
            return (torch.randn(*shape, generator=g) * scale).to(dev, bf)

        # per-rank halves of the weights; scaled so that the hidden state keeps its size through the layers
        w1 = [rnd(H, M, scale=1.0 / H ** 0.5) for _ in range(2 * L)]
        w2 = [rnd(M, H, scale=(0.5 / M) ** 0.5) for _ in range(2 * L)]   # the sum over two ranks then has the size of h
        w_lm = rnd(H, V, scale=1.0 / H ** 0.5)
        h0 = (torch.randn(B, H, generator=torch.Generator(device="cpu").manual_seed(7)) * 1.0).to(dev, bf)  # same on both ranks
        h = torch.empty_like(h0)
        mid = torch.empty(B, M, device=dev, dtype=bf)
        part = torch.empty(B, H, device=dev, dtype=bf)
        logits = torch.empty(B, V, device=dev, dtype=bf)
        gathered = torch.empty(2, B, V, device=dev, dtype=bf)
        csum = torch.empty(2, device=dev, dtype=torch.float32)
        cred = torch.empty(2, device=dev, dtype=torch.float32)
        flag = torch.zeros(1, device=dev, dtype=torch.float32)
        fout = torch.zeros(1, device=dev, dtype=torch.float32)

        def body():                         # one decode step up to the last hidden state
            h.copy_(h0)
            for i in range(2 * L):          # two all-reduces per layer
                torch.matmul(h, w1[i], out=mid)
                torch.matmul(mid, w2[i], out=part)
                nccl.all_reduce(part, h, NCCL_BF16)

        def tail(step):                     # eager, like sampling after the graph
            torch.matmul(h, w_lm, out=logits)
            nccl.all_gather(logits, gathered, NCCL_BF16)
            s = h.float().sum()
            csum[0] = s
            csum[1] = -s
            if rank == 1 and a.inject_step >= 0 and step == a.inject_step:
                csum[0] += 1.0              # the control: this rank reports a different hidden state
            nccl.all_reduce(csum, cred, NCCL_F32, NCCL_MAX)   # max(s), max(-s): equal and opposite iff both agree
            torch.cuda.synchronize(dev)

        def verify(step):
            bad = []
            if not torch.equal(gathered[rank], logits):
                bad.append({"step": step, "kind": "own_slot", "n_diff": int((gathered[rank] != logits).sum().item())})
            if not bool(torch.isfinite(gathered[1 - rank]).all().item()):
                bad.append({"step": step, "kind": "other_slot_not_finite"})
            if float(cred[0].item()) != -float(cred[1].item()):
                bad.append({"step": step, "kind": "hidden", "max": float(cred[0].item()), "min": -float(cred[1].item())})
            if not bool(torch.isfinite(h).all().item()):
                bad.append({"step": step, "kind": "hidden_not_finite"})
            if len(res["mismatches"]) < 50:
                res["mismatches"].extend(bad)

        def both_stop(stop):                # both ranks leave the loop at the same step
            flag.fill_(1.0 if stop else 0.0)
            nccl.all_reduce(flag, fout, NCCL_F32, NCCL_MAX)
            torch.cuda.synchronize(dev)
            return float(fout.item()) > 0

        main_stream = torch.cuda.current_stream()
        copy_stream = torch.cuda.Stream(dev)        # carries only the copies to the CPU, as in vLLM
        outq = queue.Queue(maxsize=2)               # at most two steps wait for their output
        astate = {"done": 0, "error": None, "first_tok": None}

        def tail_async(step):               # the same work as tail(), without waiting for the GPU
            torch.matmul(h, w_lm, out=logits)
            nccl.all_gather(logits, gathered, NCCL_BF16)
            s = h.float().sum()
            csum[0] = s
            csum[1] = -s
            if rank == 1 and a.inject_step >= 0 and step == a.inject_step:
                csum[0] += 1.0
            nccl.all_reduce(csum, cred, NCCL_F32, NCCL_MAX)
            st = torch.stack([(gathered[rank] != logits).any().float(),
                              (~torch.isfinite(gathered[1 - rank])).any().float(),
                              (cred[0] + cred[1]).abs(),
                              (~torch.isfinite(h)).any().float()])
            tok = gathered.permute(1, 0, 2).reshape(B, 2 * V).argmax(dim=1)     # the "sampled tokens"
            ev = torch.cuda.Event(blocking=True)
            torch.cuda.set_stream(copy_stream)
            try:
                copy_stream.wait_stream(main_stream)
                tok_cpu = tok.to("cpu", non_blocking=True, copy=True)
                st_cpu = st.to("cpu", non_blocking=True, copy=True)
                ev.record(copy_stream)
            finally:
                torch.cuda.set_stream(main_stream)
            outq.put((step, ev, tok_cpu, st_cpu, tok, st))      # the GPU tensors stay referenced until the copy is done

        def output_thread():                # vLLM: async_output_copy_thread -> get_output() -> copy_event.synchronize()
            kinds = ("own_slot", "other_slot_not_finite", "hidden", "hidden_not_finite")
            while True:
                item = outq.get()
                if item is None:
                    return
                step, ev, tok_cpu, st_cpu = item[:4]
                try:
                    ev.synchronize()
                    st = st_cpu.tolist()
                    tok = tok_cpu.tolist()
                    if astate["first_tok"] is None:
                        astate["first_tok"] = tok
                    bad = [{"step": step, "kind": k, "value": v} for k, v in zip(kinds, st) if v != 0.0]
                    if tok != astate["first_tok"]:
                        bad.append({"step": step, "kind": "tokens",
                                    "n_diff": sum(x != y for x, y in zip(tok, astate["first_tok"]))})
                    if len(res["mismatches"]) < 50:
                        res["mismatches"].extend(bad)
                except Exception as e:  # noqa: BLE001  (reported by the main thread)
                    astate["error"] = f"{type(e).__name__}: {e}"[:2000]
                    astate["traceback"] = traceback.format_exc()[-3000:]
                    return
                astate["done"] = step + 1

        def drain(step):                    # wait until the output thread has seen every step so far
            while astate["done"] < step and astate["error"] is None:
                time.sleep(0.001)
            if astate["error"] is not None:
                raise RuntimeError("output thread: " + astate["error"])

        nvtx = True
        try:                                # NVTX ranges per group, as vLLM marks its steps for the profiler
            torch.cuda.nvtx.range_push("warmup")
            torch.cuda.nvtx.range_pop()
        except Exception:  # noqa: BLE001
            nvtx = False
        res["nvtx"] = nvtx

        # warm-up (eager), then capture, then a few replays; nothing of this is in the measured part
        for _ in range(3):
            body()
            tail(-1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        torch.cuda.synchronize(dev)
        for _ in range(8):
            graph.replay()
            tail(-1)
        verify(-1)
        res["warmup_mismatches"] = len(res["mismatches"])
        res["mismatches"] = []
        progress("ready")
        open(out(f"ready{rank}"), "w").close()
        t0 = time.monotonic()
        while not os.path.exists(out("go")):
            if time.monotonic() - t0 > 300:
                raise RuntimeError("no go file within 300 s")
            time.sleep(0.1)
        both_stop(False)                    # the two ranks meet here before the measured part
        res["async_output"] = bool(a.async_output)
        if a.async_output:
            threading.Thread(target=output_thread, daemon=True).start()

        step = 0
        t_meas = time.monotonic()
        for grp in range(a.groups):
            if nvtx:
                torch.cuda.nvtx.range_push(f"group{grp}")
            for _ in range(a.steps):
                graph.replay()
                if a.async_output:
                    if astate["error"] is not None:
                        raise RuntimeError("output thread: " + astate["error"])
                    tail_async(step)
                else:
                    tail(step)
                    verify(step)
                step += 1
                res["steps"] = step
                if step % 16 == 0:
                    progress(f"{step} {time.time():.1f}")
            if a.async_output:
                drain(step)
            if nvtx:
                torch.cuda.nvtx.range_pop()
            res["groups"] = grp + 1
            if both_stop(bool(res["mismatches"])):
                break
            time.sleep(a.gap)
        if a.async_output:
            outq.put(None)
        res["measured_seconds"] = round(time.monotonic() - t_meas, 2)
        res["collectives"] = step * (2 * a.layers + 2)
        progress(f"{step} done")
        dump()
        print(f"[rank {rank}] done: {step} steps, mismatches {len(res['mismatches'])}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001  (a CUDA or NCCL error ends the run; recorded, not retried)
        res["error"] = f"{type(e).__name__}: {e}"[:2000]
        res["traceback"] = traceback.format_exc()[-4000:]
        try:
            res["output_thread_traceback"] = astate.get("traceback")
        except Exception:  # noqa: BLE001  (the error came before the output machinery existed)
            pass
        try:
            dump()
        except Exception:  # noqa: BLE001
            pass
        print(f"[rank {rank}] ERROR {res['error']}", flush=True)
        return 2


def parent(a):
    os.makedirs(a.out, exist_ok=True)
    for name in ("go", "nccl_id.bin", "ready0", "ready1", "progress0.txt", "progress1.txt", "rank0.json", "rank1.json"):
        if os.path.exists(os.path.join(a.out, name)):     # nothing of an earlier run may be mistaken for this one
            os.remove(os.path.join(a.out, name))
    base = [sys.executable, os.path.abspath(__file__), "--out", a.out, "--groups", str(a.groups), "--steps", str(a.steps),
            "--layers", str(a.layers), "--batch", str(a.batch), "--hidden", str(a.hidden), "--mlp", str(a.mlp),
            "--vocab", str(a.vocab), "--gap", str(a.gap), "--inject-step", str(a.inject_step),
            "--async-output", str(a.async_output)]
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logs = [open(os.path.join(a.out, f"rank{r}.log"), "w", encoding="utf-8") for r in (0, 1)]
    procs = [subprocess.Popen(base + ["--rank", str(r)], stdout=logs[r], stderr=subprocess.STDOUT) for r in (0, 1)]
    print(f"started ranks with pids {[p.pid for p in procs]}", flush=True)

    def mtime(r):
        try:
            return os.path.getmtime(os.path.join(a.out, f"progress{r}.txt"))
        except OSError:
            return None

    stalled = False
    first_fail = None
    t_go = None
    t_start = time.time()
    while True:
        rcs = [p.poll() for p in procs]
        if all(rc is not None for rc in rcs):
            break
        now = time.time()
        if t_go is None and os.path.exists(os.path.join(a.out, "go")):
            t_go = now
        if first_fail is None and any(rc not in (None, 0) for rc in rcs):
            first_fail = now
        last = [mtime(r) for r in (0, 1)]
        quiet = t_go is not None and all(
            procs[r].poll() is not None or (now - max(t_go, last[r] or 0)) > a.stall for r in (0, 1))
        if quiet or (first_fail is not None and now - first_fail > 30) or (t_go is None and now - t_start > a.startup):
            stalled = first_fail is None
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
    status, code = judge(ranks, rcs, stalled, a.expect, a.inject_step)
    prog = []
    for r in (0, 1):
        try:
            prog.append(open(os.path.join(a.out, f"progress{r}.txt"), encoding="utf-8").read().strip())
        except OSError:
            prog.append(None)
    summary = {"status": status, "expect": a.expect, "exit_code": code, "return_codes": rcs, "stalled": stalled,
               "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "last_progress": prog,
               "steps": [r["steps"] if r else None for r in ranks], "groups": [r["groups"] if r else None for r in ranks],
               "mismatch_records": [len(r["mismatches"]) if r else None for r in ranks],
               "errors": [r.get("error") if r else "no result file" for r in ranks],
               "measured_seconds": [r.get("measured_seconds") if r else None for r in ranks],
               "collectives": [r.get("collectives") if r else None for r in ranks],
               "config": {"groups": a.groups, "steps": a.steps, "layers": a.layers, "batch": a.batch, "hidden": a.hidden,
                          "mlp": a.mlp, "vocab": a.vocab, "inject_step": a.inject_step, "async_output": a.async_output}}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"status={status} expect={a.expect} exit={code} steps={summary['steps']} last_progress={prog} "
          f"mismatch_records={summary['mismatch_records']} errors={summary['errors']}", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", type=int, default=60)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--mlp", type=int, default=4864)
    ap.add_argument("--vocab", type=int, default=75968)
    ap.add_argument("--gap", type=float, default=0.8)
    ap.add_argument("--stall", type=float, default=90.0, help="seconds without progress of a rank before the run counts as hung")
    ap.add_argument("--startup", type=float, default=420.0, help="seconds the ranks may take until the go file exists")
    ap.add_argument("--inject-step", type=int, default=-1)
    ap.add_argument("--async-output", type=int, default=0, help="1: a second thread waits for each step's output")
    ap.add_argument("--expect", choices=("ok", "injected"), default="ok")
    ap.add_argument("--rank", type=int, default=-1)
    a = ap.parse_args()
    sys.exit(worker(a) if a.rank >= 0 else parent(a))


if __name__ == "__main__":
    main()
