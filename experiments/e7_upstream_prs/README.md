# E7：在社区 PR 上重跑 / E7: measurements on upstream PR branches

## 问题

E2 量到，Qwen3.5-4B 设 `--prefix-match-unit` 以后 prefill 在 prompt 末尾多停一个调度步，首个 token 慢约 30 毫秒。E7 在两个上游 PR 上重跑同样的测量：#65186 让 Triton 的 GDN 后端能在一段 prefill 中间存状态，问它能不能去掉多停的那一步。#66881 加了按收益淘汰的策略，问它能不能保住 E5 里丢掉的命中。

## 做法

- 单张 L40（功耗上限 250 W），Qwen3.5-4B，Triton/FLA 后端，块大小 576。
- 四遍对比（和 E2 相同）：同一批请求跑四遍 C0a、C1a、C1b、C0b（C0 默认，C1 设粒度），R 回放，种子 4、5、6，每种子对话 30 个、SWE-smith 10 个、SWE-rebench 6 个会话，输出 16，KV 24 GiB。逐请求配对，判定用 `experiments/e2_ttft_cost/judge.py`：同配置两遍之差要在 ±10 毫秒以内，人为加的 30 毫秒要能读出来，否则不下结论。
- 冷请求探针：`scripts/run_probe.sh`，每个长度 5 次，预热 1 次，输出 4 个 token。每个服务先跑 `check_cache_correctness.py`（数数 prompt，命中和不命中时贪心文本和第一名的 token 要相同）。
- 每一轮的预测和判据都在开跑前写下。

| 轮 | 被测对象 | 内容 |
|---|---|---|
| R0、R0b | 主线 `1479deb0`，GPU0 和 GPU1 | 同日基线，粒度 16 |
| R1、R1b | #65186 旧分支头 `754f73e0` 移植到 `1479deb0`（`patches/port_pr65186_onto_1479deb0.patch`，两处冲突手工合并） | 粒度 16（R1）、粒度 64（R1b） |
| R2 | #66881 分支头 `c19296d4` 移植到 `1479deb0`（0 处冲突，`patches/port_pr66881.sh`） | E5 的 S 档请求，三种开关 |
| R3 | #65186 分支头 `e4ee672b` 对它的合并基点 `2f70c32d`（基点已含 #66938），只差 3 个 Python 文件，不用移植 | 粒度 64，探针、四遍对比、从日志直接数步数 |

## 结果

**R0（主线，GPU0）**：对照通过。多停一步的请求慢 36.4 毫秒（E2 在 300 W 下是 34.5），每个 token 的 prefill 代价从 0.080–0.083 涨到 0.096–0.109 毫秒，之后的请求六格慢 26.5–33.0 毫秒。四条预测都成立：功耗降了，多停的那一步几乎不变。**R0b（主线，GPU1）**：对照通过，但九格里有四格和 R0 差 5 毫秒以上（最多 10.5 毫秒）。所以不同时间、不同卡的两次四遍对比相减时，10 毫秒以内的差不当结论。

**R1**（移植版，粒度 16）的结论已被 R3 取代：粒度 16 只在存档点正好落在 64 的格点上才不多停，这个 PR 要配粒度 64。

**R1b**（移植版，粒度 64，第一次的对照因卡上有别人的任务没过，在空卡上重跑）：对照通过。第 1 个请求三格 +2.2、+3.9、+1.0 毫秒（D1 成立）。之后的请求六格仍慢 +23.0 到 +34.3 毫秒（D2 不成立）。移植版的默认配置在 SWE 四格比主线快 20.5–27.0 毫秒（D3 按判据成立）。事后按调度器规则给每个请求算步数（`split_steps.py`）：`1479deb0` 的 `is_mamba_prefill_checkpoint_valid` 要求这一段越过一个块，3,299 个请求里 2,486 个（75%）仍要多停，它们慢 30–38 毫秒。这条规则上游已经知道（#60888，草稿），#66938 放宽了一部分，按新规则算仍要多停的是 1,264 个（38%，计算不是实测）。

**R2**（#66881，E5 S 档的合成会话，16 个并发，粒度 16，每格 528 个计分请求）。P0 不加开关，P1 加 `--mamba-frontier-checkpoint`，P2 再加 `--prefix-cache-eviction gds --prefill-cost-coefficients 5.06e-5 1.16e-3`：

| 池（MiB） | 输出 | 主线（E5） | P0 | P1 | P2 |
|---|---|---|---|---|---|
| 3,456（1.0 倍峰值） | 16 / 256 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.369 / 0.117 |
| 5,184（1.5 倍） | 16 / 256 | 0.574 / 0.492 | 0.542 / 0.384 | 0.087 / 0.056 | 0.981 / 0.863 |
| 8,640（充足） | 16 / 256 | 1.000 / 没有量 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

G1（只开末尾存档不比不开好）成立，而且明显更差。G2、G3 只有输出 16 那一半满足。两条对照各有一格没到线：P2 充足池、输出 16 有 97.3% 的请求命中等于预测（线是 99.5%），P0 在 5,184 MiB、输出 256 是 0.384（线是 0.40）。这一轮卡上同时有别人的任务，只报命中数。「两个开关要一起开」是 PR 说明里已经写的，这里的新信息是按收益淘汰单独的量和全注意力的对照（E5）。

**R3**（新主线上的 #65186，粒度 64。探针和四遍对比时卡上没有别人的任务，数步数那一步不量耗时，和别人的任务共用了卡）：

- 从 `--enable-logging-iteration-details` 的日志直接数步数（`steps_from_log.py`）：PR 环境粒度 64 下 319 个请求里 317 个和规则算的相同（T7 成立）。故意用错的规则读同一份日志只有 46.1%（没有存档格点）和 63.3%（旧规则）。之后的请求里粒度 64 比默认多停的占比，PR 环境对话 45.9%、SWE-smith 39.3%，基线环境 86.5% 和 93.6%。
- 冷请求探针，PR 环境粒度 64 减默认（毫秒）：100、300、500 个 token +37.1、+37.4、+34.8，580、1,000、2,000、8,000 个 token +1.0、+1.5、+2.0、+8.9（T6 成立）。不超过一个块的冷 prompt，这个 PR 帮不上。
- 四遍对比，逐请求配对差的中位数（毫秒），两个环境的对照都通过：

| 格 | 基线：64 减默认 | PR：64 减默认 | 默认：PR 减基线 | 粒度 64：PR 减基线 |
|---|---|---|---|---|
| 第 1 个请求，对话 / SWE-smith / SWE-rebench | +2.1 / +34.3 / +19.6 | +0.3 / +0.8 / −0.3 | +2.3 / −27.6 / −16.1 | +1.2 / −58.4 / −36.4 |
| 之后，对话，思考开 / 关 | +26.3 / +27.9 | +8.0 / +9.0 | +0.1 / +0.4 | −1.5 / −0.3 |
| 之后，SWE-smith，思考开 / 关 | +17.6 / +17.5 | −2.4 / −2.0 | −20.0 / −21.8 | −34.9 / −34.8 |
| 之后，SWE-rebench，思考开 / 关 | +16.3 / +15.1 | −9.7 / −9.0 | −21.8 / −21.9 | −42.9 / −42.4 |

T0–T4 成立，T5（粒度 64 下 PR 比基线快至少 10 毫秒，六格）在对话两格不成立。结论：在含 #66938 的主线上加 #65186，粒度 64 在 Agent 负载上不再是亏的，对话上仍慢 8–9 毫秒。没有这个 PR 时粒度 64 仍慢 15–28 毫秒，所以改善来自这个 PR。对话上 45% 的请求在 PR 环境里少停一步、快 30 毫秒以上，不到一半，中位数看不出（`pair_by_steps.py`）。PR #65186 在 2026-11-17 合并，合并的就是 R3 测的这一版，合并后的主线没有单独测。

## 局限

- 一个模型（Qwen3.5-4B），一张卡，回放一次一个请求。并发下 #65186 按 `max_num_seqs` 分配的存档工作区有没有别的代价，这里测不到。
- R3 里粒度 64 下 PR 环境每一遍有 27 个请求（0.8%）命中得比基线少 64 到 1,344 个 token，两遍相同，原因没有查。
- R3 基线环境粒度 64 的探针离散很大，那一行不可靠，基线的代价以四遍对比为准。
- R1、R1b 测的是手工合并过的移植版。R2 测的也是移植版，和 PR 分支头只差 `scheduler.py` 里主线后加的 2 行。
- R2 的末尾存档在这批请求上用不上（下一轮不带回模型自己的输出），这里测的是淘汰策略。
- `experiments/e2_ttft_cost/locate_overhead.py` 的步数模型不适用于 #66938 之后的规则，R3 的按规则分组用 `split_steps.py --rule 66938`。
- 逐请求的原始记录不公开，需要可以索取。

## 复现

```bash
export HP_ROOT=<experiment root> GATE_GPU_UUID=<gpu-uuid>
export HP_QWEN35_DIR=$HP_ROOT/models/Qwen3.5-4B HP_QWEN35_4B_DIR=$HP_ROOT/models/Qwen3.5-4B HP_QWEN3_DIR=$HP_ROOT/models/Qwen3-4B
# the four-pass settings, shared by R0, R0b, R1b and R3
export REPLAY_SEEDS="4 5 6" REPLAY_MODES=R REPLAY_N_CHAT=30 REPLAY_N_SMITH=10 REPLAY_N_REBENCH=6 \
       REPLAY_OUT_CAP=16 REPLAY_O_OUT_CAP=16 REPLAY_KV_BYTES=25769803776 GATE_ANY_USER=1
# R0 (R1b: HP_ENV of the port and REPLAY_UNIT=64)
HP_ENV=$HP_ROOT/envs/vllm-main REPLAY_CONFIGS="C0a C1a C1b C0b" REPLAY_DIAG_CONFIGS=C0b \
  bash scripts/run_replay.sh $HP_ROOT/runs/r0
# R3: two environments, interleaved in one run
REPLAY_UNIT=64 BASE_ENV=$HP_ROOT/envs/vllm-2f70c32d PR_ENV=$HP_ROOT/envs/vllm-pr65186-e4ee672b \
  bash experiments/e8_pr61952/run_followup_f.sh $HP_ROOT/runs/r3
# R3 probes (run once per environment)
HP_ENV=$HP_ROOT/envs/vllm-pr65186-e4ee672b PROBE_CONFIGS="h4_C0 h4_C1 h4_C64" PROBE_MODES=plain \
  PROBE_LENGTHS="16 100 300 500 560 580 1000 2000 8000" PROBE_GATE_ANY_USER=1 \
  PROBE_CHECK=experiments/e7_upstream_prs/check_cache_correctness.py bash scripts/run_probe.sh $HP_ROOT/runs/r3_probe_pr
# R3 step counts (once per environment; then steps_from_log.py on each server log)
HP_ENV=$HP_ROOT/envs/vllm-pr65186-e4ee672b REPLAY_CONFIGS="C0a C1a" REPLAY_UNIT=64 REPLAY_SEEDS=4 \
  REPLAY_DATASETS="wildchat smith" REPLAY_THINKS=off REPLAY_KV_BYTES=8589934592 \
  REPLAY_SERVE_EXTRA="--enable-logging-iteration-details" bash scripts/run_replay.sh $HP_ROOT/runs/r3_steps_pr
# R2: the E5 launcher once per arm, in the environment with the port of #66881
HP_ENV=$HP_ROOT/envs/vllm-pr66881 R2_EXTRA_P1="--mamba-frontier-checkpoint" \
  R2_EXTRA_P2="--mamba-frontier-checkpoint --prefix-cache-eviction gds --prefill-cost-coefficients 5.06e-5 1.16e-3" \
  E5_STAGE=S E5_WORKLOAD=synth E5_MODE=closed E5_INFLIGHT=16 E5_N_SESSIONS=16 E5_TURNS=12 E5_P0=2000 E5_STEP=250 \
  E5_OUT_LENS="16 256" E5_MAX_NUM_SEQS=32 E5_STOP_ON_FAIL=0 E5_CELLS="C1@8640 C1@3456 C1@5184" \
  bash experiments/e7_upstream_prs/run_r2.sh $HP_ROOT/runs/r2
# analysis (CPU only)
python experiments/e2_ttft_cost/judge.py <four-pass dir> --json e2_judge.json
python experiments/e7_upstream_prs/split_steps.py <four-pass dir> --unit 64 --alignment 64 --rule 66938 --json split_steps.json
python experiments/e7_upstream_prs/steps_from_log.py --log <run>/serve_C1a.log \
  --records <run>/e1/C1a_wildchat_thinkoff_R_s1_seed4.jsonl <run>/e1/C1a_smith_thinkoff_R_s1_seed4.jsonl --unit 64 --alignment 64
python experiments/e7_upstream_prs/probe_table.py <probe run>/probe --json probe_table.json
python experiments/e8_pr61952/pair_envs.py <base four passes> <PR four passes> --json pair_envs.json
python experiments/e7_upstream_prs/pair_by_steps.py <dir with main/ and pr/> --json t5_posthoc.json
python experiments/e5_cache_pressure/pressure_table.py <r2>/P2/e5 --json table_P2.json
```

R3 的两个环境用 `scripts/setup/setup_env_pr65186_head.sh` 建，`patches/` 里的哈希清单用来核对。R1 的移植版是在 `1479deb0` 的环境上打 `patches/port_pr65186_onto_1479deb0.patch`，R2 的是把 `patches/port_pr66881.sh` 合并出的文件盖上去，再按 `patches/port_pr66881.sha256` 核对。

## 文件

- `check_cache_correctness.py`：命中和不命中时输出是否相同的对照，`run_probe.sh` 通过 `PROBE_CHECK` 调用。
- `split_steps.py`：四遍对比的请求按规则算的多停步数分组。
- `steps_from_log.py`：从服务的逐步日志数每个请求的 prefill 步数，和规则对照。
- `pair_by_steps.py`：R3 的事后分组，两个环境之间按步数差分。
- `probe_table.py`：冷请求探针的中位数表。
- `run_r2.sh`：R2 的启动脚本，每种开关调用一次 `experiments/e5_cache_pressure/run_e5.sh`。
- `patches/`：#65186 旧版的移植补丁、#66881 的合并脚本、#65186 新版的取文件脚本和三份 sha256 清单。
- `results/`：R0、R0b、R1b 重跑、R3 两个环境的判定（`e2_judge.*`、`e2_locate.json`）、按规则分组（`split_steps*.json`）、探针表、步数汇总（`steps_*_summary.json`，去掉了逐请求的行）、两个环境的配对（`R3_pairs/`），R2 的三张表。

---

# E7: measurements on upstream PR branches

## Question

E2 found that `--prefix-match-unit` makes Qwen3.5-4B stop one extra scheduler step at the end of the prompt, which delays the first token by about 30 ms. E7 repeats the measurement on two upstream PRs: #65186 lets the Triton GDN backend save the state inside a prefill chunk (does the extra step go away?), and #66881 adds cost-aware eviction (does it keep the hits that E5 lost?).

## Setup

One L40 (250 W cap), Qwen3.5-4B, Triton/FLA backend, block 576. The four-pass comparison of E2 (C0a, C1a, C1b, C0b on the same requests; R replay; seeds 4, 5, 6; 30 / 10 / 6 sessions of WildChat / SWE-smith / SWE-rebench per seed; output 16; KV 24 GiB), judged by `experiments/e2_ttft_cost/judge.py` (same config twice within ±10 ms and an injected 30 ms shift recovered, otherwise no verdict). Cold-prompt probes with `scripts/run_probe.sh` (5 repetitions, 1 warm-up, 4 output tokens), each server first passing `check_cache_correctness.py`. Rounds: R0 / R0b main `1479deb0` on GPU0 / GPU1; R1 / R1b the old head `754f73e0` of #65186 ported onto `1479deb0` (two conflicts merged by hand) with unit 16 / 64; R2 the head `c19296d4` of #66881 ported (no conflicts) on the E5 stage-S requests; R3 the head `e4ee672b` of #65186 against its merge base `2f70c32d` (which contains #66938), three Python files apart, unit 64. Predictions and criteria were written down before each round.

## Results

- R0: controls pass; the extra step costs 36.4 ms (34.5 ms in E2 at 300 W), the per-token prefill cost rose from 0.080–0.083 to 0.096–0.109 ms; later requests are 26.5–33.0 ms slower. R0b: controls pass, but four of nine cells differ from R0 by more than 5 ms (up to 10.5 ms), so differences below 10 ms between runs on different days or GPUs are not read as results.
- R1 is superseded by R3: with unit 16 the port avoids the stop only when the checkpoint falls on the 64-token grid; the PR needs unit 64.
- R1b (rerun on an idle GPU; the first attempt failed its controls): first requests +2.2, +3.9, +1.0 ms (D1 holds), later requests still +23.0 to +34.3 ms (D2 fails), the port's default config 20.5–27.0 ms faster than main on the SWE cells (D3 holds by its criterion). By the scheduler rule (`split_steps.py`), `is_mamba_prefill_checkpoint_valid` on `1479deb0` requires the chunk to cross a block; 2,486 of 3,299 requests (75 %) still stop and are 30–38 ms slower. The rule was known upstream (#60888, draft); #66938 relaxed it, which leaves 1,264 requests (38 %, computed, not measured).
- R2: retention (output 16 / 256) at 3,456 MiB: main 0 / 0, P0 0 / 0, P1 0 / 0, P2 (frontier checkpoint plus GDS eviction) 0.369 / 0.117; at 5,184 MiB: main 0.574 / 0.492, P0 0.542 / 0.384, P1 0.087 / 0.056, P2 0.981 / 0.863; at 8,640 MiB all 1.000. G1 (the frontier checkpoint alone is no better) held and is clearly worse; G2 and G3 held for output 16 only. Two control cells missed their lines (P2 ample pool, output 16: 97.3 % of requests at the prediction, line 99.5 %; P0 at 5,184 MiB, output 256: 0.384, line 0.40). Another job shared the GPU, so only hit counts are reported. The PR's own description already says to enable both flags together; the new data is the effect of the eviction policy alone, next to a full-attention control (E5).
- R3: measured step counts match the rule for 317 of 319 requests (T7 holds; wrong rules match 46.1 % and 63.3 %). Probe, unit 64 minus default on the PR build: +37.1, +37.4, +34.8 ms at 100, 300, 500 tokens and +1.0 to +8.9 ms from 580 to 8,000 tokens (T6 holds). Four passes (table above): T0–T4 hold, T5 fails on the two chat cells. With #66938 and #65186, unit 64 no longer costs time on the agent workloads (SWE −2.0 to −9.7 ms) and still costs 8–9 ms on chat; without the PR it costs 15–28 ms. On chat 45 % of the requests take one step less with the PR and are faster by 30 ms or more, but the median does not show it (`pair_by_steps.py`). PR #65186 was merged on 2026-11-17 in the version measured here; merged main was not measured separately.

## Limitations

One model, one GPU, one request at a time (the PR's checkpoint workspace sized by `max_num_seqs` may cost something under concurrency). In R3, 27 requests per pass (0.8 %) hit 64 to 1,344 fewer tokens on the PR build with unit 64, identically in both passes; not investigated. The base build's unit-64 probe in R3 was noisy and is not used. R1, R1b and R2 measured ports, not the PR branches themselves. The frontier checkpoint of #66881 cannot pay off on the R2 requests (the model's own output is not carried into the next turn). `locate_overhead.py`'s step model does not apply after #66938. Per-request raw records are available on request.

## Reproduce

Commands are in the Chinese section above (`scripts/run_replay.sh`, `experiments/e8_pr61952/run_followup_f.sh`, `scripts/run_probe.sh`, `run_r2.sh`, then the CPU-only analysis scripts). Environments: `scripts/setup/`; the port patch, the merge script and the sha256 lists are in `patches/`.

## Files

`check_cache_correctness.py` correctness control; `split_steps.py` grouping by rule-computed extra steps; `steps_from_log.py` measured step counts from the server log; `pair_by_steps.py` R3 post-hoc split; `probe_table.py` probe medians; `run_r2.sh` R2 launcher; `patches/` port patch, merge and fetch scripts, sha256 lists; `results/` the summaries listed above.
