# E5：并发加缓存吃紧 / E5: prefix cache under concurrency and a tight KV pool

## 问题

E1 到 E4 都是一次一个会话，缓存从来没有被逼到淘汰。E5 问：KV 池放不下所有活跃会话时，混合模型（Qwen3.5-4B，`--prefix-match-unit 16`）保住的命中比全注意力模型（Qwen3-4B）少多少。上游 issue #61281 说 prompt 末尾的状态会过早回到空闲队列，E5 也看这件事在真实 GPU 上会不会出现。

## 做法

- vLLM `1479deb0`，单张 L40（功耗上限 250 W），`--max-model-len 16384 --max-num-seqs 32 --kv-cache-metrics`，池的大小用 `--kv-cache-memory-bytes` 固定。
- 配置：F0 是 Qwen3-4B（64 个 token 一块，9 MiB），C1 是 Qwen3.5-4B 加粒度 16（576 个 token 一块，18 MiB），C0 是 Qwen3.5-4B 默认配置，只跑标明的几格。
- 请求是 token id 的 `/v1/completions`，输出长度固定。指标是保留率，即命中的 token 数除以缓存无限大时的预测（`pressure_sessions.infinite_cache_prediction`），只算每个会话第 1 个之后的请求。另记命中为 0 的占比、抢占次数和缓存占用。
- 四档到达方式，种子 1、2、3：

| 档 | 到达方式 | 会话 | 池 |
|---|---|---|---|
| K 对照 | 同 S，种子 1，输出 16 | 合成 | 充足：F0 28,103 MiB，C1 8,640 MiB。最小：F0 4,608 MiB，C1 1,440 MiB |
| S 饱和 | 16 个 worker 各把一个会话从头发到尾 | 合成，每会话 12 轮，prompt 从 2,000 涨到 4,750 个 token，会话之间不共享前缀，输出 16 和 256 | 实测峰值占用 W 的 1.0、1.05、1.1、1.5 倍，补测 1.75、2.0、2.25 倍。W 是 F0 1,249 块（11,241 MiB），C1 192 块（3,456 MiB） |
| I 空闲淘汰 | 12 个活跃会话，最多 2 个请求同时在跑 | SWE-smith，思考开，每种子 36 个会话，输出 16 | 模拟器给出的不丢命中的最小池 F\* 的 1.3、1/1.25、1/2、1/3 倍 |
| L 真实饱和 | 8 个 worker | SWE-smith，思考开，每种子 24 个会话，输出 16 | 实测峰值 W_L（F0 10,737 MiB，C1 3,330 MiB）的 1.0、1.15、1.5、2.0、3.0 倍 |

- 对照：充足的池里 99.5% 以上的请求命中等于预测，最小的池保留率低于 0.25，故意给不够的池（F0 1,024 MiB）服务拒绝启动，没有抢占的轮次里逐请求之和等于 `/metrics` 的计数。对照不过就不读判定。
- 每档的预测、判据和停止条件都在开跑前写下。`lru_sim.py` 按 vLLM 的空闲队列规则逐块模拟，两套规则 M0（状态块在请求结束时释放）和 M1（prompt 算完就回到空闲队列，即 #61281 的描述）先给出预测。

## 结果

对照全部通过：充足的池 100% 的请求命中等于预测，最小的池保留率 F0 0.015、C1 0.000，F0 给 1,024 MiB 时服务拒绝启动，所有轮次 0 报错。

S 档，保留率（输出 16 / 256），每格 528 个计分请求：

| 池 / W | F0 | C1 | C1 命中为 0 的占比 |
|---|---|---|---|
| 1.0 | 1.000 / 1.000 | 0.000 / 0.000 | 100% / 100% |
| 1.05 | 1.000 / 1.000 | 0.003 / 0.004 | 99.4% / 99.2% |
| 1.1 | 1.000 / 1.000 | 0.040 / 0.040 | 93.2% / 93.2% |
| 1.5 | 1.000 / 1.000 | 0.574 / 0.492 | 35.0% / 41.7% |
| 1.75 | 没有跑 | 0.992 / 0.992 | 0.6% / 0.6% |
| 2.0 | 没有跑 | 1.000 / 1.000 | 0.0% / 0.0% |

- C0 在 1.0 W 也只有 0.087 / 0.074，所以丢命中不是粒度 16 带来的。1.0 W、输出 16 时 C1 的缓存占用最高只有 64%，命中已经是 0。
- 模拟器按 M1 预测 1.0 W、输出 256 的保留率是 0.815，按 M0 是 1.000，实测 0.000。预测 S1（F0 不丢）成立，S2（C1 在 1.5 W 不丢）不成立，S3（C1 在 1.0 W 低于 0.95）成立。停止条件第 1 条不成立，所以接着做了 L 档。

I 档，保留率，F0 每格 1,377 个计分请求，C1 每格 1,331 个，抢占都是 0：

| 池 | F0：MiB / 保留率 | C1：MiB / 保留率 |
|---|---|---|
| 1.3 F\* | 30,432 / 1.000 | 8,354 / 0.975 |
| F\*/1.25 | 18,727 / 0.995 | 5,141 / 0.840 |
| F\*/2 | 11,705 / 0.865 | 3,213 / 0.640 |
| F\*/3 | 7,803 / 0.671 | 2,142 / 0.452 |
| 字节数相同 | 11,705 / 0.865 | 11,705 / 0.9985 |

I1 不成立，I2 不成立（保留率差 15.5 和 22.5 个百分点，重算量中位数之比 1.37 和 2.10），I3（字节数相同时 C1 不低于 F0）成立，I4（池越小越低）成立。C0 在 5,141 MiB 是 0.945。

L 档，保留率，F0 每格 926 个计分请求，C1 每格 895 个：

| 池 / W_L | 1.0 | 1.15 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| F0 | 0.952 | 1.000 | 1.000 | 1.000 | 1.000 |
| C1 | 0.296 | 0.663 | 1.003 | 1.003 | 1.003 |

1.0 W_L 时首个 token 等待的中位数 F0 332 毫秒、C1 4,475 毫秒。L1、L3、L4 成立，L2（C1 在 1.0 和 1.15 倍都低于 0.5）不成立。高于 1 是会话之间共享的系统提示带来的命中。

读法：按各自峰值的同样倍数收紧，混合模型掉得多，丢的时候是整段丢。按字节数比，混合模型在测过的每一档都不比全注意力差：S 档 F0 要 11.0 GiB，C1 给 6,048 MiB 就保住 99.2%。L 档 F0 要 12.1 GiB，C1 要 4.9 GiB。原因和 #61281 描述的方向一致，这是推断，没有验证。E7 的 R2 在同一批请求上测了 #66881 的淘汰策略。

## 局限

- 一个规模（4B），一张卡，`--max-num-seqs 32`。S 档是合成会话，I、L 档只有 SWE-smith。
- 模拟器对混合模型的块数高估约四分之一（S 档峰值算 240 块，实测 192 块）。I 档 C1 的池按模拟器 F\* 乘 0.8 取，可能偏紧，I2 不成立有一部分可能来自这里。
- C1 不丢命中的最小池在合成会话上落在 1.5 到 1.75 W 之间，没有再细分。
- S 档 1.75–2.25 W 和 L 档 C1 的 3.0 W_L 是补测，卡上同时有别人的任务，只报命中数，不报耗时。
- L 档的数用「全部 k > 0」的口径（`pressure_table.py`）。按 `pressure_judge.py` 去掉每个 worker 第一个会话的口径，1.0 和 1.15 W_L 是 F0 0.972、1.000，C1 0.376、0.779，判定不变。
- 逐请求的原始记录不公开，需要可以索取。

## 复现

```bash
export HP_ROOT=<experiment root> HP_ENV=$HP_ROOT/envs/vllm-main GATE_GPU_UUID=<gpu-uuid>
export HP_QWEN35_DIR=$HP_ROOT/models/Qwen3.5-4B HP_QWEN3_DIR=$HP_ROOT/models/Qwen3-4B
COMMON="E5_MAX_NUM_SEQS=32 E5_KV_METRICS=1 E5_STOP_ON_FAIL=0"
SYNTH="E5_WORKLOAD=synth E5_MODE=closed E5_INFLIGHT=16 E5_N_SESSIONS=16 E5_TURNS=12 E5_P0=2000 E5_STEP=250"
# K and S
env $COMMON $SYNTH E5_STAGE=K E5_SEEDS=1 E5_OUT_LENS=16 E5_CELLS="F0@28103 F0@4608 C1@8640 C1@1440" \
  bash experiments/e5_cache_pressure/run_e5.sh $HP_ROOT/runs/e5_k
env $COMMON $SYNTH E5_STAGE=S E5_OUT_LENS="16 256" \
  E5_CELLS="F0@11241 C1@3456 F0@11799 C1@3636 F0@12366 C1@3798 F0@16862 C1@5184 C0@3456" \
  bash experiments/e5_cache_pressure/run_e5.sh $HP_ROOT/runs/e5_s
# I and L
env $COMMON E5_STAGE=I E5_WORKLOAD=smith E5_THINKING=on E5_MODE=window E5_INFLIGHT=2 E5_LIVE=12 \
  E5_N_SESSIONS=36 E5_OUT_LENS=16 E5_CELLS="F0@18727 C1@5141 F0@11705 C1@3213 C1@11705 F0@30432 C1@8354 C0@5141 F0@7803 C1@2142" \
  bash experiments/e5_cache_pressure/run_e5.sh $HP_ROOT/runs/e5_i
env $COMMON E5_STAGE=L E5_WORKLOAD=smith E5_THINKING=on E5_MODE=closed E5_INFLIGHT=8 E5_N_SESSIONS=24 \
  E5_OUT_LENS=16 E5_CELLS="F0@10737 C1@3330 F0@12348 C1@3830 F0@16106 C1@4995 F0@21474 C1@6660 F0@32211 C1@9990" \
  bash experiments/e5_cache_pressure/run_e5.sh $HP_ROOT/runs/e5_l
# tables and verdicts from the records (CPU only)
python experiments/e5_cache_pressure/pressure_table.py $HP_ROOT/runs/e5_s/e5 --json table_s.json
python experiments/e5_cache_pressure/pressure_judge.py <dir with the K and S records> --json judge_k_and_s.json
# pool planning with the simulator
python experiments/e5_cache_pressure/plan_pools.py --arch hy --workload synth --seeds 1 2 3 --n-sessions 16 \
  --turns 12 --p0 2000 --step 250 --n-out 256 --mode closed --inflight 16
```

数据和模型的准备见 `scripts/setup/`。Python 脚本从仓库根目录运行，仓库根目录要在 `PYTHONPATH` 上。

## 文件

- `pressure_replay.py`：并发回放客户端，带对照请求、`/metrics` 采样和对账。
- `pressure_sessions.py`：合成会话、真实会话的渲染、缓存无限大时的命中预测。
- `lru_sim.py`：vLLM 块池和空闲队列的逐块模拟器（M0、M1 两套规则）。
- `plan_pools.py`：用模拟器在同样的会话和发送顺序上定池的大小。
- `pressure_table.py`：每格的保留率等汇总，README 的表来自它。
- `pressure_judge.py`：对照、停止条件的判定，可选地和模拟器逐请求对照。
- `run_e5.sh`：GPU 服务器上的启动脚本（闸门、按 PID 停服务）。
- `results/`：`table_s.json`、`table_i.json`、`table_l.json`、`table_s_extra.json`（S 档补测）、`table_l_extra.json`（L 档 `C1@9990`）、`judge_k_and_s.json`（K、S 档的判定）。

---

# E5: prefix cache under concurrency and a tight KV pool

## Question

E1 to E4 ran one session at a time and never forced the cache to evict. E5 asks how many prefix-cache hits the hybrid model (Qwen3.5-4B, `--prefix-match-unit 16`) keeps, compared with the full-attention Qwen3-4B, when the KV pool cannot hold all active sessions. It also checks whether the early release of the prompt-end state described in upstream issue #61281 shows on a real GPU.

## Setup

- vLLM `1479deb0`, one L40 (250 W power cap), `--max-model-len 16384 --max-num-seqs 32 --kv-cache-metrics`, pool size fixed with `--kv-cache-memory-bytes`.
- Configs: F0 Qwen3-4B (64-token blocks, 9 MiB each), C1 Qwen3.5-4B with unit 16 (576-token blocks, 18 MiB each), C0 Qwen3.5-4B defaults (a few cells only).
- Requests are token-id `/v1/completions` calls with a fixed output length. The metric is retention: cached tokens divided by the hit an unlimited cache would give (`pressure_sessions.infinite_cache_prediction`), over the requests after the first of each session. Also recorded: share of requests with zero hit, preemptions, cache usage.
- Stages, seeds 1, 2, 3: K (controls: ample pools F0 28,103 / C1 8,640 MiB, minimum pools 4,608 / 1,440 MiB); S (saturation: 16 workers, each sends one synthetic session of 12 turns, prompts from 2,000 to 4,750 tokens, no shared prefixes, outputs 16 and 256; pools at 1.0, 1.05, 1.1, 1.5 times the measured peak W, later also 1.75, 2.0, 2.25; W = 1,249 blocks (11,241 MiB) for F0 and 192 blocks (3,456 MiB) for C1); I (idle eviction: 12 live SWE-smith sessions, at most 2 requests in flight, 36 sessions per seed, thinking on, output 16; pools at 1.3, 1/1.25, 1/2, 1/3 of the simulator's smallest lossless pool F\*); L (real saturation: 8 workers, 24 SWE-smith sessions per seed, output 16; pools at 1.0, 1.15, 1.5, 2.0, 3.0 times the measured peak W_L, 10,737 MiB for F0 and 3,330 MiB for C1).
- Controls: in the ample pool at least 99.5 % of the requests hit the prediction; in the minimum pool retention is below 0.25; a deliberately too small pool (F0, 1,024 MiB) must refuse to start; per-request sums equal the `/metrics` counters in runs without preemption.
- Predictions, criteria and stop conditions were written down before each stage. `lru_sim.py` simulates vLLM's free queue block by block under two readings of the source: M0 (the state block is freed when the request ends) and M1 (it returns to the free queue when the prompt is done, as #61281 describes).

## Results

All controls passed (ample pools: 100 % of the requests hit the prediction; minimum pools: retention 0.015 for F0 and 0.000 for C1; the 1,024 MiB server refused to start; no request errors).

| Stage, pool | F0 | C1 |
|---|---|---|
| S, 1.0 W (out 16 / 256) | 1.000 / 1.000 | 0.000 / 0.000 |
| S, 1.05 W | 1.000 / 1.000 | 0.003 / 0.004 |
| S, 1.1 W | 1.000 / 1.000 | 0.040 / 0.040 |
| S, 1.5 W | 1.000 / 1.000 | 0.574 / 0.492 |
| S, 1.75 / 2.0 W | not run | 0.992 / 1.000 (both outputs) |
| I, 1.3 F\* / F\*/1.25 / F\*/2 / F\*/3 | 1.000 / 0.995 / 0.865 / 0.671 | 0.975 / 0.840 / 0.640 / 0.452 |
| I, same bytes (11,705 MiB) | 0.865 | 0.9985 |
| L, 1.0 / 1.15 / 1.5 / 2.0 / 3.0 W_L | 0.952 / 1.000 / 1.000 / 1.000 / 1.000 | 0.296 / 0.663 / 1.003 / 1.003 / 1.003 |

- S: at 1.0 W every C1 request after the first hit nothing, while cache usage peaked at only 64 % (output 16). C0 kept 0.087 / 0.074, so unit 16 is not the cause. The simulator predicted 0.815 under M1 and 1.000 under M0. S1 (F0 loses nothing) held, S2 (C1 lossless at 1.5 W) failed, S3 (C1 below 0.95 at 1.0 W) held; stop condition 1 failed, so stage L was run.
- I: I1 failed, I2 failed (retention gaps of 15.5 and 22.5 points, recompute median ratios 1.37 and 2.10), I3 (C1 not below F0 at the same bytes) held, I4 (monotone) held. C0 at 5,141 MiB: 0.945.
- L: L1, L3, L4 held, L2 (C1 below 0.5 at 1.0 and 1.15 W_L) failed. Median TTFT at 1.0 W_L: F0 332 ms, C1 4,475 ms. Values above 1 come from system prompts shared between sessions.
- Reading: tightened by the same factor relative to its own peak, the hybrid model loses far more, and it loses whole requests. Per byte it is never worse than full attention in any measured cell: on S, F0 needs 11.0 GiB while C1 keeps 99.2 % at 6,048 MiB; on L, F0 needs 12.1 GiB and C1 4.9 GiB. The cause is consistent with #61281 but not verified. E7 round R2 measures the eviction policy of PR #66881 on the same requests.

## Limitations

- One model size (4B), one GPU, `--max-num-seqs 32`; S uses synthetic sessions, I and L SWE-smith only.
- The simulator overestimates the hybrid model's block use by about a quarter (S peak: 240 simulated, 192 measured). The C1 pools of stage I were set from the simulator's F\* times 0.8 and may be too tight, which may explain part of the I2 failure.
- C1's smallest lossless pool on synthetic sessions lies between 1.5 and 1.75 W; it was not narrowed further.
- The S cells at 1.75 to 2.25 W and C1 at 3.0 W_L were run later with another job on the GPU; only hit counts are reported from them.
- L numbers count all requests with k > 0 (`pressure_table.py`). With `pressure_judge.py`'s window, which drops each worker's first session, 1.0 and 1.15 W_L give F0 0.972, 1.000 and C1 0.376, 0.779; the verdicts do not change.
- Per-request raw records are not published; they are available on request.

## Reproduce

See the commands in the Chinese section above (`run_e5.sh` with the `E5_*` variables of each stage, then `pressure_table.py`, `pressure_judge.py`, `plan_pools.py` on the CPU). Run Python scripts from the repository root with the root on `PYTHONPATH`; data and models are prepared with `scripts/setup/`.

## Files

`pressure_replay.py` concurrent replay client; `pressure_sessions.py` sessions and the unlimited-cache prediction; `lru_sim.py` block-pool simulator; `plan_pools.py` pool sizing with the simulator; `pressure_table.py` per-cell summary (source of the tables above); `pressure_judge.py` controls and stop conditions; `run_e5.sh` launcher; `results/` the summaries (`table_*.json`, `judge_k_and_s.json`).
