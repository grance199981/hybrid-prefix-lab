#!/usr/bin/env python
"""Tables of the two-GPU investigation, from the files the runs write.

  summarize.py comm DIR [--json out.json] [--redact-uuids]
      DIR holds the group directories K/ A/ B/ C/ of run_comm_test.sh (comm_test.py).
      Per group: status, transports named in the NCCL log, steps, checked collectives,
      payload bytes, mismatch records, errors, and the median latency of every
      (operation, size). With zero failures in N collectives the 95 % upper bound of the
      failure rate is about 3 / N. --redact-uuids replaces each GPU UUID in the output by
      <gpu-uuid-0>, <gpu-uuid-1>, ... in order of first appearance.
  summarize.py repro SUMMARY.json
      One line per mode from the summary.json of run_graph_repro.sh.
  summarize.py groups FILE.jsonl [...] [--decode-tokens 128]
      One line per probe_client.py --pattern groups record file (a vLLM run of
      scripts/run_probe.sh): measured groups, decode groups, and the groups that did not
      complete (an error, or fewer output tokens than requested). A run that faulted ends
      with such a group, because the client records the group in which the server stopped.
"""
import argparse
import json
import os
import re

UUID = re.compile(r"(?:GPU-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def load_comm(d, g):
    s = json.load(open(os.path.join(d, g, "summary.json"), encoding="utf-8"))
    ranks = []
    for r in (0, 1):
        p = os.path.join(d, g, f"rank{r}.json")
        ranks.append(json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None)
    steps = min(x for x in s["steps"] if x is not None) if any(x is not None for x in s["steps"]) else 0
    lat = {}
    for r in ranks:
        for x in (r or {}).get("latency", []):
            lat.setdefault((x["op"], x["n"]), []).append(x["median_ms"])
    return {
        "group": g, "status": s["status"], "expect": s["expect"], "exit_code": s["exit_code"],
        "transports": s["transports"], "steps": s["steps"], "collectives": steps,
        "rate_bound_95": (3.0 / steps) if steps and s["status"] == "ok" else None,
        "payload_gb_per_rank": round((s["payload_bytes"][0] or 0) / 1e9, 1),
        "seconds": s["seconds"], "mismatch_records": s["mismatch_records"], "errors": s["errors"],
        "started": s["started"], "finished": s["finished"],
        "env": [(r or {}).get("env") for r in ranks],
        "gpu": [(r or {}).get("gpu") for r in ranks], "nccl": (ranks[0] or {}).get("nccl"),
        "median_ms": {f"{op}:{n}": [round(v, 3) for v in vs] for (op, n), vs in sorted(lat.items())},
        "mismatches": [(r or {}).get("mismatches") for r in ranks],
    }


def redact(text):
    """Replace GPU UUIDs (with or without the GPU- prefix) by numbered placeholders."""
    seen = {}

    def sub(m):
        key = m.group(0).removeprefix("GPU-")
        seen.setdefault(key, f"<gpu-uuid-{len(seen)}>")
        return seen[key]
    return UUID.sub(sub, text)


def comm(a):
    out = [load_comm(a.dir, g) for g in "KABC" if os.path.exists(os.path.join(a.dir, g, "summary.json"))]
    for g in out:
        print(f"{g['group']}: status={g['status']} (expect {g['expect']}, exit {g['exit_code']}) "
              f"transports={g['transports']} steps={g['steps']} payload={g['payload_gb_per_rank']} GB/rank "
              f"seconds={g['seconds']} mismatch_records={g['mismatch_records']} errors={g['errors']}")
    keys = sorted({k for g in out for k in g["median_ms"]}, key=lambda k: (k.split(":")[0], int(k.split(":")[1])))
    print("median ms (rank 0 / rank 1)".ljust(30) + "".join(g["group"].rjust(20) for g in out))
    for k in keys:
        print(k.ljust(30) + "".join(" / ".join(f"{v:.2f}" for v in g["median_ms"].get(k, [])).rjust(20) for g in out))
    if a.json:
        text = json.dumps(out, indent=1)
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            f.write(redact(text) if a.redact_uuids else text)


def repro(a):
    s = json.load(open(a.summary, encoding="utf-8"))
    for m, x in s.items():
        print(m, "| status", x["status"], "| steps", x["steps"], "| return codes", x["return_codes"],
              "| stalled", x["stalled"], "| mismatches", x["mismatch_records"], "| last progress", x["last_progress"],
              "| seconds", x["measured_seconds"], "| started", x["started"], "| finished", x["finished"])


def groups(a):
    for path in a.files:
        with open(path, encoding="utf-8") as f:
            rs = [r for r in (json.loads(x) for x in f if x.strip()) if r.get("kind") == "measure"]
        n_dec, bad = 0, []
        for r in rs:
            if r["phase"] == "decode":
                n_dec += 1
            want = r["B"] * (a.decode_tokens if r["phase"] == "decode" else a.prefill_tokens)
            if r["n_errors"] or r.get("error") or r["completion_tokens"] < want:
                bad.append(f"{r['phase']} group {n_dec if r['phase'] == 'decode' else r['rep'] + 1} "
                           f"({r['completion_tokens']} of {want} tokens)")
        print(f"{os.path.basename(path)}: {len(rs)} groups, {n_dec} decode groups, "
              f"incomplete: {', '.join(bad) if bad else 'none'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("comm")
    p.add_argument("dir")
    p.add_argument("--json")
    p.add_argument("--redact-uuids", action="store_true")
    p = sub.add_parser("repro")
    p.add_argument("summary")
    p = sub.add_parser("groups")
    p.add_argument("files", nargs="+")
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--prefill-tokens", type=int, default=1)
    a = ap.parse_args()
    {"comm": comm, "repro": repro, "groups": groups}[a.cmd](a)


if __name__ == "__main__":
    main()
