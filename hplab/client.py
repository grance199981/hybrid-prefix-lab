"""Talking to a vLLM OpenAI-compatible server.

Two flavours, because the experiments need both:
  - blocking (http.client): one request at a time, timed precisely; used by the
    cold-prompt probes, the profiling clients and the concurrency workers (one
    thread per worker);
  - asyncio (aiohttp): the multi-turn replay, which keeps a session's requests in
    order while several sessions run at once.

/reset_prefix_cache needs the server to run with VLLM_SERVER_DEV_MODE=1.
"""
import http.client
import json
import time


# ------------------------------------------------------------------ blocking
def post_json(host, port, path, body=None, timeout=60):
    """POST a JSON body (or nothing). Returns (status, raw response bytes)."""
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    c.request("POST", path, body=json.dumps(body) if body is not None else None,
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def http_get(host, port, path, timeout=30):
    """GET. Returns (status, decoded text)."""
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data.decode("utf-8", "replace")


def stream_completion(host, port, body, timeout=600):
    """One streamed /v1/completions request. Returns wall-clock send and end
    times (epoch ns, for lining up with a profiler trace), TTFT, end-to-end time
    and the usage numbers, including how many prompt tokens came from the cache.

    ttft_ms: first chunk that carries text. first_chunk_ms: first chunk at all."""
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    t_send = time.time_ns()
    p0 = time.perf_counter()
    c.request("POST", "/v1/completions", body=json.dumps(body),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    first_chunk = ttft = None
    usage, finish = {}, None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        obj = json.loads(payload)
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            if first_chunk is None:
                first_chunk = time.perf_counter() - p0
            if ch.get("text") and ttft is None:
                ttft = time.perf_counter() - p0
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    e2e = time.perf_counter() - p0
    c.close()
    details = usage.get("prompt_tokens_details") or {}
    return {"t_send_ns": t_send, "t_end_ns": t_send + int(e2e * 1e9),
            "ttft_ms": None if ttft is None else 1000 * ttft,
            "first_chunk_ms": None if first_chunk is None else 1000 * first_chunk,
            "e2e_ms": 1000 * e2e, "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens") or 0,
            "completion_tokens": usage.get("completion_tokens"), "finish_reason": finish}


def completion_body(model, ids, n_out):
    """A greedy streamed completion of exactly n_out tokens for a token-id prompt."""
    return {"model": model, "prompt": ids, "max_tokens": n_out, "min_tokens": n_out,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}


def reset_prefix_cache(host, port, tries=20, pause=1.0):
    """POST /reset_prefix_cache until it answers 200 without `"success": false`."""
    for i in range(tries):
        try:
            status, data = post_json(host, port, "/reset_prefix_cache")
            if status == 200:
                try:
                    if json.loads(data).get("success", True):
                        return True
                except (ValueError, AttributeError):      # empty or non-JSON body
                    return True
        except OSError:
            pass
        if i + 1 < tries:
            time.sleep(pause)
    return False


# ------------------------------------------------------------------- metrics
def parse_metrics(text, names, fill_missing=False):
    """Sum each named Prometheus metric over its label sets (a `_total` suffix is
    accepted). Names that do not appear are left out, or set to 0 with fill_missing."""
    names = set(names)
    out = {n: 0.0 for n in names} if fill_missing else {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        head, _, value = line.rpartition(" ")
        name = head.split("{", 1)[0]
        if name.endswith("_total"):
            name = name[:-6]
        if name in names:
            try:
                out[name] = out.get(name, 0.0) + float(value)
            except ValueError:
                pass
    return out


def read_metrics(host, port, names):
    status, text = http_get(host, port, "/metrics")
    if status != 200:
        raise RuntimeError(f"/metrics HTTP {status}")
    return parse_metrics(text, names)


# -------------------------------------------------------------------- asyncio
async def post_stream(http, url, body):
    """Streamed /v1/chat/completions. Returns usage, TTFT, finish reason.

    ttft_ms: first chunk whose delta carries content, reasoning or tool calls.
    first_chunk_ms: first chunk that carries a choice at all (the role chunk
    included); it exists even when a streaming parser withholds every delta."""
    t0 = time.perf_counter()
    ttft, first_chunk, usage, finish, kinds = None, None, {}, None, set()
    async with http.post(url + "/v1/chat/completions", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if first_chunk is None:
                    first_chunk = time.perf_counter() - t0
                kinds.update(k for k, v in d.items() if v)
                if ttft is None and (d.get("content") or d.get("reasoning")
                                     or d.get("reasoning_content") or d.get("tool_calls")):
                    ttft = time.perf_counter() - t0
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return {"usage": usage, "ttft_ms": 1000 * ttft if ttft is not None else None,
            "first_chunk_ms": 1000 * first_chunk if first_chunk is not None else None,
            "delta_keys": sorted(kinds),
            "e2e_ms": 1000 * (time.perf_counter() - t0), "finish_reason": finish}


async def post_full(http, url, body):
    """Non-streaming /v1/chat/completions. Returns the whole response object."""
    t0 = time.perf_counter()
    async with http.post(url + "/v1/chat/completions", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        obj = await resp.json()
    obj["_e2e_ms"] = 1000 * (time.perf_counter() - t0)
    return obj


def usage_fields(usage):
    details = usage.get("prompt_tokens_details") or {}
    return {"prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens") or 0,
            "completion_tokens": usage.get("completion_tokens")}


async def reset_prefix_cache_async(http, url, tries=20):
    import asyncio
    for _ in range(tries):
        async with http.post(url + "/reset_prefix_cache") as resp:
            if resp.status == 200 and (await resp.json()).get("success"):
                return True
        await asyncio.sleep(1)
    return False


async def read_counters_async(http, url, names):
    """The named counters, summed over label sets; 0 for a counter that is absent."""
    async with http.get(url + "/metrics") as resp:
        return parse_metrics(await resp.text(), names, fill_missing=True)
