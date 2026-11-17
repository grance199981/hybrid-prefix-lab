"""Reading an Nsight Systems trace that was exported to SQLite
(`nsys export --type sqlite`). Standard library only.

How the profiling experiments slice a trace:
  - requests go one at a time with a pause in between, so the GPU kernels form
    one burst per request (`bursts`);
  - vLLM's scheduler emits the NVTX range "schedule: update_after_schedule" once
    per scheduling step, right before that step's GPU work, so the GPU work of
    step i lies between the end of marker i and the start of marker i + 1
    (`step_windows`);
  - for every step: wall time, GPU busy time (union of kernel intervals), idle
    gaps, kernel time by class, CUDA-graph vs one-by-one launches (`step_stats`).
"""
import collections
import re
import sqlite3

STEP_MARK = "schedule: update_after_schedule"

# Kernel classes, matched against the demangled kernel name; the first match wins.
CLASSES = [
    ("gdn", re.compile(r"gdn|gated_delta|delta_rule|chunk_fwd|chunk_scaled|chunk_local|causal_conv|"
                       r"recompute_w_u|merge_16x16|_fused_post_conv|_gather_initial_states|"
                       r"_scatter_states|_fused_sigmoid_gate", re.I)),
    ("attention", re.compile(r"flash_fwd|fmha|reshape_and_cache|_fused_qk_rmsnorm_rope", re.I)),
    ("gemm", re.compile(r"gemm|gemv|cutlass::Kernel|matmul", re.I)),
    ("sampling", re.compile(r"topk|sort|multinomial|argmax|cumsum|log_softmax|softmax|sampl", re.I)),
    ("norm_elementwise", re.compile(
        r"elementwise|rmsnorm|rms_norm|act_and_mul|silu|sigmoid|reduce|index|gather|scatter|"
        r"cat|where|fill|copy|arange|pow|rsqrt|_apply_write|_zero_kv", re.I)),
]
CLASS_NAMES = [c for c, _ in CLASSES] + ["other"]


def classify(name):
    for cls, rx in CLASSES:
        if rx.search(name):
            return cls
    return "other"


def union_ms(intervals):
    """Total length of the union of [start, end) intervals (ns in, ms out)."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1e6


def bursts(kernels, gap_ns):
    """Split a start-sorted kernel list where the GPU is idle longer than gap_ns."""
    out, cur, last_end = [], [], None
    for k in kernels:
        if last_end is not None and k[0] - last_end > gap_ns:
            out.append(cur)
            cur = []
        cur.append(k)
        last_end = k[1] if last_end is None else max(last_end, k[1])
    if cur:
        out.append(cur)
    return out


def step_windows(marks, lo, hi):
    """marks: [(start, end)] of the step marker inside one request. Returns the
    GPU windows [(begin, finish)]: from the end of marker i to the start of
    marker i + 1 (the last one runs to hi)."""
    ms = [m for m in marks if lo <= m[1] <= hi]
    return [(m[1], ms[i + 1][0] if i + 1 < len(ms) else hi) for i, m in enumerate(ms)]


def step_stats(window, kernels, memcpys, api, classify=classify):
    """Breakdown of one step window. kernels: (start, end, name, graph_node, pid)."""
    b, f = window
    ks = [k for k in kernels if b <= k[0] < f]
    calls = [a for a in api if b <= a[0] < f]
    out = {"wall_ms": (f - b) / 1e6, "n_kernels": len(ks),
           "n_api_calls": len(calls), "api_ms": sum(a[1] - a[0] for a in calls) / 1e6,
           "n_kernel_launch_calls": sum(1 for a in calls if "LaunchKernel" in a[2]),
           "n_graph_launch_calls": sum(1 for a in calls if "GraphLaunch" in a[2])}
    if not ks:
        out.update(pre_ms=None, busy_ms=0.0, gaps_ms=0.0, post_ms=None, by_class={}, n_graph=0,
                   busy_graph_ms=0.0, busy_eager_ms=0.0)
        return out
    first, last = min(k[0] for k in ks), max(k[1] for k in ks)
    busy = union_ms([(k[0], k[1]) for k in ks])
    by = collections.defaultdict(float)
    for k in ks:
        by[classify(k[2])] += (k[1] - k[0]) / 1e6
    mc = [m for m in memcpys if b <= m[0] < f]
    out.update(pre_ms=(first - b) / 1e6, busy_ms=busy, gaps_ms=(last - first) / 1e6 - busy,
               post_ms=(f - last) / 1e6, by_class={k: round(v, 4) for k, v in by.items()},
               n_graph=sum(1 for k in ks if k[3]),
               busy_graph_ms=union_ms([(k[0], k[1]) for k in ks if k[3]]),
               busy_eager_ms=union_ms([(k[0], k[1]) for k in ks if not k[3]]),
               memcpy_ms=sum(m[1] - m[0] for m in mc) / 1e6,
               memcpy_bytes=sum(m[2] or 0 for m in mc))
    return out


def label_steps(n_steps, completion_tokens):
    """Step kinds of one request, in order: the prefill steps, one short step right
    after the prefill, then one decode step per further output token. So
    number of prefill steps = number of steps - completion_tokens."""
    n_pre = n_steps - (completion_tokens or 0)
    if n_pre < 1 or n_steps < n_pre + 1:
        return None, ["unknown"] * n_steps
    return n_pre, ["prefill"] * n_pre + ["post_prefill"] + ["decode"] * (n_steps - n_pre - 1)


def load_trace(path):
    """Returns (session start in epoch ns, kernels, memcpys, nvtx ranges, CUDA API calls).
    kernels: (start, end, name, graph node id or 0, global pid), sorted by start."""
    c = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    strings = dict(c.execute("select id, value from StringIds"))
    t0 = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    kcols = {r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    name_col = "demangledName" if "demangledName" in kcols else "shortName"
    kernels = [(s, e, strings.get(n) or strings.get(sn) or str(n), g or 0, pid)
               for s, e, n, sn, g, pid in c.execute(
        f"select start, end, {name_col}, shortName, graphNodeId, globalPid "
        "from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    memcpys = list(c.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY "
                             "order by start")) if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
    nvtx = []
    if "NVTX_EVENTS" in tables:
        cols = {r[1] for r in c.execute("pragma table_info(NVTX_EVENTS)")}
        text = "coalesce(text, (select value from StringIds where id = textId))" \
            if "textId" in cols else "text"
        nvtx = [(s, e, t) for s, e, t in c.execute(
            f"select start, end, {text} from NVTX_EVENTS where end is not null order by start")]
    api = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        rcols = {r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_RUNTIME)")}
        if "nameId" in rcols:
            api = [(s, e, strings.get(n, "")) for s, e, n in c.execute(
                "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
        else:
            api = [(s, e, "") for s, e in c.execute(
                "select start, end from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
    return t0, kernels, memcpys, nvtx, api
