# E10：缓存吃紧时混合模型为什么整段丢命中 / E10: why the hybrid model loses whole prompts under cache pressure

## 问题

E5 量到：并发高、缓存贴着峰值占用时，全注意力模型（Qwen3-4B）的前缀命中一个不丢，混合模型（Qwen3.5-4B，`--prefix-match-unit 16`）整段丢失，池加到 1.5 倍还丢一半。E10 问三件事：命中为 0 的请求缺的是状态块还是注意力块；一个请求 prompt 末尾那份状态是不是在请求结束前就被放回空闲队列（issue #61281 描述的路径）；修它的 PR #61282（状态一直持有到请求结束，结束时排在注意力块后面释放）能不能把命中保住。

## 做法

- vLLM main `1479deb0`，一张 L40，Qwen3.5-4B，Triton/FLA 后端，块大小 576，`--max-num-seqs 32`。负载和 E5 相同：S 档是 16 个合成会话同时跑（12 轮，prompt 从 2,000 涨到 4,750 个 token），L 档是 SWE-smith 的真实 Agent 会话（8 个 worker）。3 个种子。
- **记录钩子**（`hook/`，经 `PYTHONPATH` 加载进服务进程，只记录、不改行为）：每个请求被接纳时，用只读的 `find_longest_cache_hit_per_group` 分别查注意力组和状态组各自能命中到哪里；请求结束时看它 prompt 末尾那份状态还在不在、引用计数是多少；每次淘汰记下属于哪一组。
- **分类**：命中少于无限缓存预测的请求分三类：只缺状态（注意力组够到上一轮的存档点、状态组够不到）、两样都缺、只缺注意力。
- **CPU 回放**（`cpu_replay.py`）：在 CPU 上直接驱动 vLLM 自己的 `AsyncScheduler` 和 `KVCacheManager`，模型输出用假的 token，调度器配置和缓存分组从真实服务里 pickle 下来。用来在 GPU 出数之前做预测。
- **修法**：把 #61282 移植到 `1479deb0`（两个文件三方合并，一处冲突，解法在 `port_pr61282.py`），新环境只替换这两个文件（`scripts/setup/setup_env_pr61282.sh`）。先过正确性对照（命中和不命中时贪心解码的输出逐 token 相同），再跑同样的负载。
- 每一步的预测和判据在开跑前写下。对照：钩子不改变行为（主线的保留率落在 E5 的范围里）；每次运行开头的对照请求（全新的 prompt、重发同一个 prompt）每次都分对；钩子查到的命中和客户端收到的逐个相同。

## 结果

**合成会话（S 档）**，斜线两边是每个请求输出 16 和 256 个 token：

| 池（相对峰值占用 192 块） | 主线保留率 | 主线里命中为 0 的请求「只缺状态」 | #61282 移植版 |
|---|---|---|---|
| 1.0 倍（192 块） | 0.000 / 0.000 | 68% / 91% | 0.287 / 0.266 |
| 1.5 倍（288 块） | 0.598 / 0.395 | 100% / 100% | **0.997 / 0.997** |
| 充足（480 块） | 1.000 / 1.000 | 没有未命中 | 1.000 / 1.000 |

**真实会话（L 档，SWE-smith）**：

| 池（相对峰值占用） | 主线 | 主线里没命中满的请求「只缺状态」 | #61282 移植版 | 每轮重算中位数（主线 → 修法） |
|---|---|---|---|---|
| 1.0 倍（3,330 MiB） | 0.275 | 84% | **0.850** | 7,261 → 469 |
| 1.15 倍（3,830 MiB） | 0.647 | 88% | **0.959** | 835 → 420 |
| 3.0 倍（9,990 MiB） | 1.003 | 没有未命中 | 1.003 | 406 → 406 |

- 缺的是 prompt 末尾那一份状态，不是注意力块。每个请求结束的那一刻，那份状态在所有档上 100% 已经不被它持有；请求结束时三个状态组排在注意力组前面释放，所以它最先被淘汰。
- 修掉以后，合成会话留一半余量就不再丢命中，真实会话贴着峰值也保住 85%。合成会话贴着峰值时剩下的未命中九成是注意力块和状态一起没了，是真的容量不够。
- 修法的代价：每个运行中的请求多持一个块，真实会话上缓存占用峰值从 75% 升到 96–99%，出现了 12 次和 2 次抢占（主线 0 次）。
- CPU 回放和 GPU 对得上：主线 16 格里最大差 0.108，大部分格子到小数点后三位相同；移植版的预测（GPU 出数之前做的）和实测差不到 0.003。之前手写的块级模拟器预测错，是因为它的块数规则和真实代码不一样（多算了四分之一的块）。
- 预测错的地方：以为贴着峰值时主要是恶性循环（注意力块也丢），实际大多仍只缺状态；以为修法贴着峰值时能保住三成以上，实际 27%–29%；以为修法以后剩下的未命中不再是「只缺状态」，真实会话上还有 15% 和 30%（17 个和 9 个请求），原因没有查。

## 局限

- 机制不是这里发现的：两条路径是 issue #61281 写出来的，修法是 #61282 的作者写的。这里做的是在真实 GPU 和真实负载上逐请求定位，并验证修法。
- 测的是移植版，不是作者的原样分支（它需要 rebase）。一个模型、两种负载、各一种并发设置。

## 复现

```bash
# 服务器上：两组（主线、修法）各跑一遍，钩子加载进服务进程
E10_ARMS="main pr61282" E10_ENV_main=<venv of 1479deb0> E10_ENV_pr61282=<venv with the port> \
E5_STAGE=S E5_WORKLOAD=synth E5_CELLS="C1@8640 C1@3456 C1@5184" E5_SEEDS="1 2 3" E5_MODE=closed \
E5_INFLIGHT=16 E5_N_SESSIONS=16 E5_TURNS=12 E5_P0=2000 E5_STEP=250 E5_OUT_LENS="16 256" E5_MAX_NUM_SEQS=32 \
  bash experiments/e10_state_eviction/run_e10.sh <run_dir>
python experiments/e10_state_eviction/analyze_e10.py <run_dir>/main --json main.json
# CPU 回放：scheduler_init_<pid>.pkl 是钩子在服务启动时存下的
bash experiments/e10_state_eviction/run_cpu_replay.sh <python> <scheduler_init.pkl> <out_dir>
```

## 文件

| 文件 | 内容 |
|---|---|
| `hook/e10_hook.py`、`hook/sitecustomize.py` | 记录钩子 |
| `run_e10.sh` | 每组跑一遍 `experiments/e5_cache_pressure/run_e5.sh`，加载钩子 |
| `analyze_e10.py` | 把钩子的记录和客户端记录配对（合成会话按会话号，真实会话按 prompt 长度和时间窗），分类、统计 |
| `cpu_replay.py`、`run_cpu_replay.sh` | CPU 回放 |
| `compare_cpu_e5.py` | CPU 回放和 E5 实测逐格对比 |
| `fetch_sources.sh`、`port_pr61282.py`、`port_pr61282.sha256` | 取三个版本的源码、移植 #61282、两个文件的哈希 |
| `results/` | 每一步的汇总：`step1_cpu`、`step3_cpu_prediction`（CPU 回放）、`step2_main`、`step3_pr61282`（合成会话）、`step4_L`（真实会话）、`k4_correctness`（移植版的正确性对照） |

---

## Question

E5 found that under concurrency, with the KV pool at the peak usage, the full-attention model (Qwen3-4B) keeps every prefix hit while the hybrid model (Qwen3.5-4B, `--prefix-match-unit 16`) loses whole prompts, and still loses half at 1.5x the peak. E10 asks: is it the state block or the attention blocks that the misses lack; is a request's prompt-boundary state returned to the free queue before the request ends (the path described in issue #61281); and does PR #61282 (hold that state until the request ends and free it after the attention blocks) keep the hits.

## Setup

- vLLM main `1479deb0`, one L40, Qwen3.5-4B, Triton/FLA backend, block size 576, `--max-num-seqs 32`. The workloads of E5: stage S is 16 concurrent synthetic sessions (12 turns, prompts from 2,000 to 4,750 tokens); stage L is recorded SWE-smith agent sessions with 8 workers. Three seeds.
- **Logging hook** (`hook/`, loaded into the server through `PYTHONPATH`; it only logs): at admission, the read-only `find_longest_cache_hit_per_group` gives how far the attention group and the Mamba groups each reach; at free, whether the state at the prompt checkpoint is still there and its ref count; every eviction with its group.
- **Classes** for requests that hit less than the infinite-cache prediction: state only (attention reaches the previous turn's checkpoint, state does not), both, attention only.
- **CPU replay** (`cpu_replay.py`) drives vLLM's own `AsyncScheduler` and `KVCacheManager` on the CPU with fake model outputs; the scheduler config and the KV cache groups are pickled from the real server. It predicted the fix's numbers before the GPU runs.
- **Fix**: PR #61282 ported onto `1479deb0` (three-way merge of two files, one conflict resolved in `port_pr61282.py`); the new environment differs from main in these two files only. Greedy outputs with and without a cache hit were identical before the measurements.
- Predictions were written before each step. Controls: the hook does not change behavior; the control requests at the start of every run are classified right; the hook's hit equals the client's for every request.

## Results

Synthetic sessions (stage S), output 16 / 256 tokens:

| Pool (vs. peak usage, 192 blocks) | Main | Zero-hit requests on main that lack only the state | PR #61282 (port) |
|---|---|---|---|
| 1.0x (192 blocks) | 0.000 / 0.000 | 68% / 91% | 0.287 / 0.266 |
| 1.5x (288 blocks) | 0.598 / 0.395 | 100% / 100% | **0.997 / 0.997** |
| Ample (480 blocks) | 1.000 / 1.000 | no misses | 1.000 / 1.000 |

Recorded sessions (stage L, SWE-smith):

| Pool (vs. peak usage) | Main | Short requests on main that lack only the state | PR #61282 (port) | Median recompute per turn (main → fix) |
|---|---|---|---|---|
| 1.0x (3,330 MiB) | 0.275 | 84% | **0.850** | 7,261 → 469 |
| 1.15x (3,830 MiB) | 0.647 | 88% | **0.959** | 835 → 420 |
| 3.0x (9,990 MiB) | 1.003 | no misses | 1.003 | 406 → 406 |

- The misses lack the one state block at the prompt boundary, not the attention blocks. When a request ends, that state is no longer held by it in 100% of requests in every cell, and the three Mamba groups are freed before the attention group, so the state is evicted first.
- With the fix, synthetic sessions keep their hits at 1.5x, and recorded sessions keep 85% even at the peak. On synthetic sessions at the peak, 90% of the remaining misses lost both the attention blocks and the state: real capacity pressure.
- Cost of the fix: each running request holds one more block; on recorded sessions peak usage rises from 75% to 96–99%, with 12 and 2 preemptions (0 on main).
- The CPU replay matches the GPU: largest gap 0.108 over 16 cells on main, most cells equal to three decimals; its prediction for the port, made before the GPU run, is within 0.003.
- Wrong predictions: at the peak the misses still mostly lack only the state (not a cascade); the fix keeps 27–29% there on synthetic sessions, not 30% or more; on recorded sessions 15% and 30% of the remaining misses (17 and 9 requests) still lack only the state, cause not examined.

## Limitations

- The mechanism is not ours: issue #61281 describes both paths and PR #61282's author wrote the fix. This experiment locates the misses request by request on a GPU and on real traffic, and verifies the fix.
- We measured a port, not the author's branch (which needs a rebase). One model, two workloads, one concurrency setting each.

## Reproduce and files

See the commands and the file table in the Chinese part above.
