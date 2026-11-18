# hybrid-prefix-lab

[中文](README.md) | **English**

How much prefix-cache reuse does a hybrid-attention model such as Qwen3.5 (Gated DeltaNet linear attention plus a few full-attention layers) lose on [vLLM](https://github.com/vllm-project/vllm), compared with a plain full-attention model? How much do vLLM's existing server flags win back? And once the hit rate is back, does the first token arrive sooner or later, and where does the time go?

This repository holds the measurement code and the result summaries. The workloads are real multi-turn chats (WildChat) and two sets of coding-agent trajectories (SWE-smith, SWE-rebench); the reference model is the full-attention Qwen3-4B of the same size. The measurement report was posted to vLLM as [issue #67674](https://github.com/vllm-project/vllm/issues/67674).

## Findings

One NVIDIA L40, vLLM main (commit `1479deb0`), Qwen3.5-4B against Qwen3-4B, sessions replayed one at a time. E1 alone replayed 114,681 requests with no errors.

- **By default the gap is real.** The hybrid model recomputes 1.6 to 1.7 times as many tokens per turn as the full-attention model; the multi-turn chat hit rate is 56% against 71%. The cause is the cache block: 576 tokens for the hybrid model, 64 for full attention.
- **An existing flag closes the hit-rate gap, but the first token gets slower.** With `--prefix-match-unit 16` the hit rates differ by less than one percentage point, yet about 96% of the requests wait about 30 ms longer for their first token, more than the recomputation they save.
- **The 30 ms are one extra scheduler step.** On the L40, Qwen3.5's linear-attention layers run on the Triton backend, which cannot save the state inside a forward pass, so the scheduler cuts the prefill once more near the end of the prompt. Nsight Systems shows a full extra forward pass (658 more kernels); of the 45.2 ms the request gains, the GPU is busy for 15.1 ms and waits for the CPU to launch the next kernel for the rest.
- **The larger the model, the better the trade.** The extra step costs 38.2 ms on 9B, almost the same, and 59.8 ms on 27B (FP8, 64 layers), but every recomputed token is dearer: on 27B the flag makes the later requests 97 to 241 ms faster.
- **The upstream fix works with the right setting.** PR #65186 (merged) lets the Triton backend save the state on a 64-token grid. On the PR branch (its base already contains #66938; this is the version that was merged) with `--prefix-match-unit 64`, agent requests are no longer slower; chat requests are still 8 to 9 ms slower. Another PR, #61952, made requests with a prefill step of at most 512 tokens 67 to 275 ms slower at the server's default concurrency limit; its author confirmed the defect in the issue.
- **Under cache pressure the hybrid model loses whole prefixes.** With a cache pool just at the peak working set, the full-attention model keeps 95% to 100% of its hits and the hybrid model 0% to 66%. The hybrid model needs 1.5 to 2 times its peak working set to lose nothing.
- **The whole-prefix loss is an implementation defect, and the upstream fix works.** The one state block that each request stores at the end of its prompt returns to the free queue before the request ends, and when the request ends it is freed ahead of the request's attention blocks, so it is evicted first; without it the attention blocks cannot be hit. Among zero-hit requests, 68% to 100% lack only this block. With PR #61282 ported onto our commit, retention at 1.5x on synthetic sessions goes from 0.40–0.60 to 0.997, and on recorded agent sessions at the peak from 0.275 to 0.850. Issue #61281 described the mechanism first; this experiment locates the misses request by request on real traffic and verifies the fix.

Every experiment wrote down its predictions and judgement criteria before it ran, and every verdict comes with a known-good and a known-bad control (for example the same configuration run twice, and an artificial 30 ms delay that the analysis has to detect).

## Experiments

| | Question | Result |
|---|---|---|
| [E1](experiments/e1_prefix_hits) | How much reuse does the hybrid model lose by default; how much do existing flags win back | 1.6–1.7x recomputation; with `--prefix-match-unit 16` the hit rates differ by under 1 point |
| [E2](experiments/e2_ttft_cost) | Is it faster once the hits are back; where is the cost | ~96% of requests ~30 ms slower; one extra step +34.5 ms, none +0.3 ms |
| [E3](experiments/e3_9b) | And with 9B | The step costs 38.2 ms, about the same; the net loss shrinks but remains |
| [E4](experiments/e4_profiling) | Where does the time go; compute- or memory-bound; how much does two-GPU communication cost | GPU busy one third of the step; roofline knee at 217 tokens, plateau 691 GB/s; communication ~10% of GPU time in decode, about half in prefill |
| [E5](experiments/e5_cache_pressure) | Does the hybrid model have an extra problem under concurrency and a tight cache | Yes, whole prefixes are lost; it needs a 1.5–2x larger pool |
| [E6](experiments/e6_27b) | Does the flag still lose on 27B | No, later requests become 97–241 ms faster |
| [E7](experiments/e7_upstream_prs) | Do upstream PRs #65186 and #66881 help | #65186 with unit 64: agent requests no longer slower; #66881's cost-aware eviction keeps more hits under pressure |
| [E8](experiments/e8_pr61952) | PR #61952, measured at its author's request | 67–275 ms slower at the default concurrency limit, defect confirmed by the author; with `--max-num-seqs 8` no slower, and the extra step falls from 29.7 to 18.0 ms |
| [E9](experiments/e9_two_gpu_debug) | Why did two-GPU runs fault while being profiled | Caused by Nsight Systems 2025.1.3 tracing CUDA graph nodes; gone with 2026.6.1.251; not P2P, not the two-socket layout |
| [E10](experiments/e10_state_eviction) | Why E5 loses whole prompts, and does the upstream fix help | The misses lack the prompt-boundary state, which is released early and evicted first; with PR #61282 ported, 1.5x on synthetic sessions goes 0.40–0.60 → 0.997, recorded sessions at the peak 0.275 → 0.850 |

The README in each experiment directory is its result summary: setup, numbers, which predictions held, limits, how to rerun.

## Layout

```
hplab/                  shared library
  client.py             talk to a vLLM server: streamed requests, /metrics, prefix-cache reset
  prompts.py            synthetic prompts of an exact token length
  sessions.py           load multi-turn sessions from the three datasets and render them
  cache_model.py        vLLM's rules for the expected cache hit and the number of prefill steps
  stats.py              medians, paired-difference summaries, line fits
  nsys.py               read an Nsight Systems trace exported to SQLite
  records.py            read the per-request records of a four-pass run (C0a, C1a, C1b, C0b)
scripts/
  serve_lib.sh          shell helpers for the launchers: GPU gate, stopping processes by PID
  run_replay.sh         multi-turn replay: one vLLM server per configuration (E1, E2, E3, E6, E7, E8)
  run_probe.sh          one-request-at-a-time probes and profiling (E4, E7, E8, E9)
  setup/                build the environments, download models and data, verify hashes
experiments/            one directory per experiment, see the table
refs/                   official chat templates (inputs of the prompt rendering)
tests/                  tests of the library, the judges and the analysis scripts
```

## Reproducing

You need a Linux server with an NVIDIA GPU (an L40 with 48 GB here) and vLLM.

```bash
pip install -e ".[replay,test]"
python -m pytest -q            # no GPU needed
```

1. Environments, models and datasets: see [scripts/setup](scripts/setup). The vLLM under test is main at commit `1479deb0`; E7 and E8 add environments for the PR branches.
2. Serve and replay: the top of `scripts/run_replay.sh` and `scripts/run_probe.sh` lists every environment variable; each experiment README gives its settings.
3. Judge and summarise: the Reproduce section of each experiment README lists the commands that read the replay records and print the verdicts and tables.

The raw per-request records are not in the repository; ask in an issue if you need them.

## Scope

- One GPU type (L40) and the Triton backend. According to the vLLM source, GPUs that use the FlashInfer backend (H100 and similar) do not need the extra step; this was not measured.
- Mostly 4B, plus one four-pass comparison each for 9B and 27B (FP8).
- E1 to E4 send one session at a time (E1 adds one level of 8 sessions in round robin); concurrency and cache pressure were measured separately in E5.
- The server's power cap changed from 300 W to 250 W partway through; comparisons across dates re-measured their baseline in the same batch.

## License

Apache License 2.0, see [LICENSE](LICENSE). The patches in `experiments/e7_upstream_prs/patches/` come from vLLM pull requests and carry vLLM's license (Apache 2.0); see [NOTICE](NOTICE).
