#!/usr/bin/env python
"""What an Nsight Systems trace (exported to SQLite) contains, per process: used to
check that a collection really traced CUDA graph nodes and NCCL kernels.

Usage: trace_counts.py <trace.sqlite> [more.sqlite ...]

Prints, for every process, the number of GPU kernels and how many of them are NCCL
kernels, the names of the NCCL kernels, and how many kernel rows carry a CUDA graph
node id (only there when the trace was taken with --cuda-graph-trace=node).
Read-only; CPU only.
"""
import sqlite3
import sys


def counts(path):
    c = sqlite3.connect(path)
    cols = [r[1] for r in c.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    graph_cols = [x for x in cols if "raph" in x]
    out = {"path": path, "graph_columns": graph_cols, "processes": [], "nccl_names": [], "rows_with_graph_node": None}
    q = ("SELECT (k.globalPid >> 24) & 16777215, COUNT(*), SUM(s.value LIKE '%nccl%') "
         "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id GROUP BY 1")
    for pid, n, nccl in c.execute(q):
        out["processes"].append({"pid": pid, "kernels": n, "nccl_kernels": nccl})
    q = ("SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id "
         "WHERE s.value LIKE '%nccl%' GROUP BY 1")
    out["nccl_names"] = [[name[:70], n] for name, n in c.execute(q)]
    out["rows_with_graph_node"] = {}
    for col in graph_cols:                  # newer exports have graphNodeId, graphId and two "source" columns
        total, with_id = c.execute(f"SELECT COUNT(*), COUNT({col}) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        out["rows_with_graph_node"][col] = {"rows": total, "with_id": with_id}
    return out


def main():
    for p in sys.argv[1:]:
        r = counts(p)
        print(p)
        print("  kernel rows with a value in each graph column:", r["rows_with_graph_node"])
        for x in r["processes"]:
            print("  pid", x["pid"], "kernels", x["kernels"], "nccl kernels", x["nccl_kernels"])
        for name, n in r["nccl_names"]:
            print("   ", name, n)


if __name__ == "__main__":
    main()
