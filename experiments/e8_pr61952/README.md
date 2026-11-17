# E8：PR #61952 原样分支 / E8: PR #61952 as it is

## 问题

测量报告（issue #67674）发出后，PR #61779 和 #61952 的作者请我们在 #61952 的分支上跑同样的四遍对比。#61952 把 GDN 的核心算子收进 CUDA graph，作者报告在 8 张 A100、32 个并发会话下，带 prefill 的一步里主机时间从 99.3 毫秒降到 55.4 毫秒（他人报告，未自测）。E8 问：在 L40、Qwen3.5-4B、一次一个请求的条件下，这个 PR 能不能让设了 `--prefix-match-unit` 以后多停的那一步变便宜。

## 做法

- 基线环境：main `5ec99956` 的 vLLM wheel。PR 环境：同一份环境换上分支头 `6891ee3e` 的 13 个 Python 文件。分支比基点多 19 个提交、落后 0，所以这就是原样的 PR，没有移植。文件清单和 sha256 在 `patches/`。
- 一张 L40（功耗上限 250 W），Qwen3.5-4B，Triton/FLA 后端，块大小 576，两个环境在同一张卡上、同一天、绑同一组核。
- 四步，依次做：
  1. 正确性对照（`experiments/e7_upstream_prs/check_cache_correctness.py`）。
  2. 冷请求探针，长度 16、560、580、2,000、8,000，默认和粒度 16，每个长度 5 次。
  3. 四遍对比（同 E2：C0a、C1a、C1b、C0b，种子 4、5、6，会话数 30 / 10 / 6，输出 16）。两个环境交错跑四段：基线 C0a、C1a，PR C0a、C1a，PR C1b、C0b，基线 C1b、C0b。
  4. Nsight Systems 时间线，560 和 16 个 token 的冷请求。
- 预测和判据在开跑前写下。探针出结果以后加了诊断 D1（`--max-num-seqs 8`，判读规则写在运行之前）和补充实验 F（两个环境都加 `--max-num-seqs 8` 的四遍对比，四段在一次运行里连着跑，`run_followup_f.sh`）。

## 结果

正确性对照：两个环境、两种配置共 36 组全部一致，D1 的服务上又 36 组一致。

**探针**（首个 token 的等待，5 次的中位数，毫秒，服务用默认参数）：

| prompt 长度 | 基线：默认 | 基线：粒度 16 | PR：默认 | PR：粒度 16 |
|---|---|---|---|---|
| 16 | 86.5 | 86.0 | 153.6 | 181.8 |
| 560 | 114.4 | 145.3 | 114.4 | 273.3 |
| 580 | 145.3 | 145.7 | 247.9 | 267.7 |
| 2,000 | 219.3 | 242.1 | 369.2 | 507.1 |
| 8,000 | 898.0 | 940.2 | 1023.1 | 1214.7 |

和预测相反：在服务的默认 `max_num_seqs` 下，PR 让有一步不超过 512 个 token（CUDA graph 的捕获上限）的请求慢 67–275 毫秒。只有一步 560 个 token 的请求两边一样快（114.4 对 114.4）。decode 没有变（后 3 个 token 36.2–40.0 毫秒）。默认参数下的四遍对比两个环境的对照都没过，只当描述：PR 上设粒度 16 的代价是 +143 到 +231 毫秒（基线 +12 到 +34），默认配置的中位数比基线高 73–117 毫秒。

**诊断 D1**（两个环境都加 `--max-num-seqs 8`）：PR 在每个长度上都不比基线慢超过 6.2 毫秒。560 个 token 时粒度 16 减默认，基线 +32.9 毫秒，PR +19.2 毫秒。16 个 token 的默认配置，基线 86.2 毫秒，PR 37.0 毫秒。graph 捕获占用的显存 PR 从 2.46 GiB 降到 0.20 GiB。

**时间线**（560 个 token，粒度 16 减默认，到出第一个 token 为止，5 个请求的中位数）：

| 环境 | 墙上时间 | GPU 在跑 | GPU 空着 | 执行的 kernel | 逐个发起 | graph 回放 |
|---|---|---|---|---|---|---|
| 基线 | +43.5 毫秒 | +15.2 | +28.3 | +663 | +335 | +33 |
| PR，默认上限 | +135.0 | +132.3 | +2.6 | +1444 | +88 | +10 |
| PR，上限 8 | +21.4 | +17.9 | +3.7 | +1433 | +88 | +10 |

PR 做到了它要做的事：主机一侧的等待几乎没有了。代价是 GPU 上的活多了，kernel 数不随上限变，耗时随上限变：上限 8 时 GPU 多跑 17.9 毫秒，默认上限时多跑 132.3 毫秒。读源码的解释是 PR 的静态形式按 `max_num_seqs` 的容量建网格，这一点只验证了「开销随这个参数变」，没有验证是否成正比。GPU 是异步执行的，按调度步切窗口会把 kernel 算到下一步里，所以表里按「到出第一个 token 为止」合计。

**补充实验 F**（`--max-num-seqs 8`，四段连着跑）：基线九格对照全过，PR 有一格（之后，SWE-rebench，思考开）默认配置两遍差 +14.1 毫秒，所以 F1–F3 不下正式判定，下面只当描述。多停一步的代价从 29.7 毫秒降到 18.0 毫秒（第 1 个请求里没有命中的 194 个）。默认配置没有变快：之后的请求 PR 减基线 +0.2 到 +7.2 毫秒，只有不超过 16 个 token 的 prompt 快约 47 毫秒（22 个请求）。粒度 16 下 PR 减基线：对话 −11.2 毫秒，SWE-smith −2.6 和 −5.0，SWE-rebench +0.7 和 +2.0，收益随上下文变长而变小，原因没有查。设粒度 16 在 PR 上仍让九格都变慢（+12.8 到 +49.5 毫秒）。

| 编号 | 判定 |
|---|---|
| K1 基线上那一步还在 | 探针那一半成立（+30.8 毫秒，区间 +28 到 +46） |
| K2 PR 让那一步变便宜 | 在预注册的条件（默认参数）下不成立，方向相反。上限 8 时便宜约 14 毫秒（探针）或 22 毫秒（时间线） |
| K3 默认配置也受益 | 默认上限下不成立（580 个 token 慢 103 毫秒），上限 8 时快 14 毫秒 |
| K4 净效果仍是变慢 | 四遍对比的对照没过，不判 |

这些结果回报到了 issue #67674。PR 作者在 issue 里确认，默认 `max_num_seqs` 下变慢是 #61952 的缺陷。

## 局限

- 一个模型，一张卡，一次一个请求。`--max-num-seqs 8` 会限制同时处理的请求数，它对并发高的部署是否可接受，这里没有量。只量了默认和 8 两个上限。
- 默认参数下的四遍对比因为四段之间等了几个小时，对照没过，只当描述。补充实验 F 也有一格对照没过。
- 基线是 `5ec99956`，比 E1–E7 的 `1479deb0` 新 315 个提交，E8 的绝对数不和 E7 比。
- 时间线的汇总（`results/timeline/summary.json`）是把 `experiments/e4_profiling/analyze_q1.py` 的逐请求输出按「到出第一个 token 为止」加起来的，这一步没有单独的脚本。
- 逐请求的原始记录不公开，需要可以索取。

## 复现

```bash
export HP_ROOT=<experiment root> GATE_GPU_UUID=<gpu-uuid> HP_QWEN35_DIR=$HP_ROOT/models/Qwen3.5-4B \
       HP_QWEN35_4B_DIR=$HP_ROOT/models/Qwen3.5-4B HP_QWEN3_DIR=$HP_ROOT/models/Qwen3-4B
BASE=$HP_ROOT/envs/vllm-5ec99956; PR=$HP_ROOT/envs/vllm-pr61952      # scripts/setup/setup_env_pr61952.sh
# steps 1 and 2: correctness control and probe, once per environment
HP_ENV=$PR PROBE_CONFIGS="h4_C0 h4_C1" PROBE_MODES=plain PROBE_LENGTHS="16 560 580 2000 8000" \
  PROBE_CHECK=experiments/e7_upstream_prs/check_cache_correctness.py bash scripts/run_probe.sh $HP_ROOT/runs/e8_probe_pr
# diagnostic D1: the same with PROBE_SERVE_EXTRA="--max-num-seqs 8" PROBE_LENGTHS="16 560 580 2000"
# step 4: timeline, PROBE_MODES=nsys PROBE_LENGTHS="16 560" (PROBE_SERVE_EXTRA for the cap-8 group)
# step 3 and follow-up F: four passes of both environments, interleaved
export REPLAY_SEEDS="4 5 6" REPLAY_MODES=R REPLAY_N_CHAT=30 REPLAY_N_SMITH=10 REPLAY_N_REBENCH=6 \
       REPLAY_OUT_CAP=16 REPLAY_O_OUT_CAP=16 REPLAY_KV_BYTES=25769803776
BASE_ENV=$BASE PR_ENV=$PR REPLAY_SERVE_EXTRA="--max-num-seqs 8" \
  bash experiments/e8_pr61952/run_followup_f.sh $HP_ROOT/runs/e8_f
# analysis: merge seg*_base/e1 and seg*_pr/e1 into one directory per build, then
python experiments/e2_ttft_cost/judge.py <build dir> --json e2_judge.json
python experiments/e2_ttft_cost/locate_overhead.py <build dir> --block 576 --json e2_locate.json
python experiments/e8_pr61952/pair_envs.py <base dir> <PR dir> --json pair_envs.json
```

默认参数下的四遍对比当时是四次单独的运行（顺序同上），用 `scripts/run_replay.sh` 加 `HP_ENV` 和 `REPLAY_CONFIGS`，不带 `REPLAY_SERVE_EXTRA`。

## 文件

- `pair_envs.py`：同一个请求在两个环境之间配对（PR 减基线），也被 E7 的 R3 使用。
- `run_followup_f.sh`：两个环境交错的四遍对比，四段连着跑，每段调用一次 `scripts/run_replay.sh`。
- `patches/`：`fetch_pr61952.sh`（取 PR 改动的 13 个文件）和基点、分支头的 sha256 清单。
- `results/`：默认参数下两个环境的判定（`fourpass_*`，对照没过，只当描述），补充实验 F 两个环境的判定、定位和配对（`f_*`），时间线的汇总（`timeline/summary.json`）。

---

# E8: PR #61952 as it is

## Question

After the measurement report (issue #67674), the author of PRs #61779 and #61952 asked us to run the same four-pass comparison on the #61952 branch. #61952 captures the GDN core kernels in CUDA graphs; the author reported that on 8 A100s with 32 concurrent sessions the host time of a step with prefill fell from 99.3 ms to 55.4 ms (their numbers, not reproduced here). E8 asks whether, on an L40 with Qwen3.5-4B and one request at a time, the PR makes the extra step caused by `--prefix-match-unit` cheaper.

## Setup

Base environment: the vLLM wheel of main `5ec99956`. PR environment: the same with the 13 Python files of the branch head `6891ee3e` (19 commits ahead of the base, 0 behind, so this is the PR as it is; file lists and sha256 in `patches/`). One L40 (250 W cap), Qwen3.5-4B, Triton/FLA, block 576, both environments on the same GPU, same day, same CPU cores. Steps: (1) correctness control; (2) cold-prompt probe at 16, 560, 580, 2,000, 8,000 tokens, default and unit 16, 5 repetitions; (3) the four-pass comparison of E2, interleaved in four segments (base C0a C1a, PR C0a C1a, PR C1b C0b, base C1b C0b); (4) Nsight Systems timelines of 560- and 16-token cold prompts. Predictions were written down before the runs; the diagnostic D1 (`--max-num-seqs 8`) and follow-up F (both environments with `--max-num-seqs 8`, four segments in one run with `run_followup_f.sh`) were added after the probe, each with its reading rules written before it ran.

## Results

- Correctness: all 36 comparisons equal, and 36 more on the D1 servers.
- Probe with the server's default parameters (table above): opposite to the prediction, the PR makes requests with a prefill step of at most 512 tokens (the CUDA graph capture limit) 67–275 ms slower; a single 560-token step is equally fast on both (114.4 ms); decode is unchanged. The default-parameter four-pass run failed its controls on both builds and is descriptive only: unit 16 costs +143 to +231 ms on the PR build (+12 to +34 ms on the base), the default config is 73–117 ms slower.
- D1 (`--max-num-seqs 8` on both): the PR is never more than 6.2 ms slower; at 560 tokens unit 16 costs +32.9 ms on the base and +19.2 ms on the PR; a 16-token prompt takes 86.2 ms on the base and 37.0 ms on the PR. Graph capture memory of the PR falls from 2.46 GiB to 0.20 GiB.
- Timeline (560 tokens, unit 16 minus default, up to the first token, table above): the PR removes the host-side wait (GPU idle +28.3 ms on the base, +2.6 to +3.7 ms on the PR) but runs about twice as many kernels; their GPU time depends on the cap: +17.9 ms with cap 8, +132.3 ms with the default cap. Reading the source, the PR's static form sizes its grid by the `max_num_seqs` capacity; only "the cost changes with this parameter" was verified, not proportionality. Because the GPU runs asynchronously, per-step windows put kernels into the next step, so the table sums up to the first token.
- Follow-up F: base controls pass in all nine cells, the PR fails one (+14.1 ms between its two default passes), so F1–F3 get no formal verdict. Descriptively: the extra step falls from 29.7 to 18.0 ms; the default config is not faster (+0.2 to +7.2 ms on later requests) except prompts of at most 16 tokens (about 47 ms faster, 22 requests); with unit 16 the PR gains 11.2 ms on chat, 2.6–5.0 ms on SWE-smith and nothing on SWE-rebench; unit 16 still costs +12.8 to +49.5 ms on the PR build.
- Predictions: K1 held on the probe (+30.8 ms); K2 failed in the pre-registered condition (default parameters), in the opposite direction, while with cap 8 the step became 14 ms (probe) or 22 ms (timeline) cheaper; K3 failed at the default cap and held with cap 8; K4 not judged.
- The results were reported in issue #67674; the PR author confirmed there that the slowdown at the default `max_num_seqs` is a defect of #61952.

## Limitations

One model, one GPU, one request at a time; `--max-num-seqs 8` limits concurrency and its cost for busy deployments was not measured; only the default and 8 were tried. The default-parameter four-pass run and one cell of follow-up F failed their controls. The base `5ec99956` is 315 commits newer than `1479deb0`, so E8 numbers are not compared with E7. `results/timeline/summary.json` sums the per-request output of `experiments/e4_profiling/analyze_q1.py` up to the first token; that summing step is not a separate script. Per-request raw records are available on request.

## Reproduce and files

See the Chinese sections above. `pair_envs.py` pairs requests across two builds (also used by E7 R3); `run_followup_f.sh` runs the interleaved four segments; `patches/` holds `fetch_pr61952.sh` and the sha256 lists; `results/` the judgements, locations, pairing and timeline summary.
