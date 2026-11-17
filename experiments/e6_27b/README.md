# E6：换成 27B（FP8）模型 / E6: The same comparison on Qwen3.5-27B (FP8)

## 问题

4B（E2）和 9B（E3）上，`--prefix-match-unit 16` 多停的那一步分别是 34.5 和 38.2 毫秒，净效果都是变慢。4B、9B 都是 32 层，27B 是 64 层（48 层线性注意力加 16 层全注意力，读自各模型的 `config.json`）。如果那一步的代价跟着「发起多少个 kernel」走，它在 27B 上会明显变大。少重算的收益也随模型变大。E6 问：再大一档的模型上，净效果会不会转成变快。

## 做法

- 模型 `Qwen/Qwen3.5-27B-FP8`（官方 FP8 版，revision `97f5941b…`）。bf16 原版 55.6 GB，一张 48 GB 的 L40 放不下。L40 没有原生 FP8 计算，vLLM 用「只量化权重」的跑法（`MarlinFP8ScaledMMLinearKernel`），权重占 27.9 GiB。
- 启动日志：块大小 832 个 token（4B、9B 是 576），GDN prefill 后端是 Triton/FLA。
- 和 E2、E3 相同的四遍设计（C0a、C1a、C1b、C0b）、种子 4、5、6、客户端和判定脚本。改动：每个种子 15 / 5 / 3 个会话（E3 的一半，因为 27B 的 prefill 更慢），KV 缓存 10 GiB，服务加 `--max-num-seqs 64`（默认的 256 要 256 个线性注意力状态块，显存放不下），客户端和 `locate_overhead.py` 的块大小参数 832。服务绑在 GPU 所在 NUMA 节点的 CPU 核上。
- 单张 L40，vLLM `1479deb0`，功耗上限 250 W。

每个实验开跑前都先写下预测和判定标准，结果出来后不改预测。

## 结果

口径核对：6,348 个计分请求，0 报错，prompt 长度 0 个不一致，72 组 `/metrics` 对账相等。四遍的块大小都是 832，后端都是 Triton/FLA，服务日志里没有 CUDA 错误。

`judge.py` 的总判定是 `incomplete`：第 1 个请求、SWE-rebench 这一格只有 18 对，少于脚本要求的 20 对（会话数减半造成）。下面按事先写的 W0–W4 各自的判据写。单位毫秒，逐请求配对的中位数，「设参数减默认」为负表示设了参数更快：

| 格 | 对数 | 设参数减默认 | 变慢的占比 | 同配置两遍之差（默认 / 粒度 16） | 人为加的 30 | 每轮少重算（token） |
|---|---|---|---|---|---|---|
| 第 1 个请求，WildChat | 88 | +63.0 | 89.8% | −5.3 / −0.8 | 35.3 | 0 |
| 第 1 个请求，SWE-smith | 30 | +41.7 | 100% | −2.4 / −1.1 | 32.4 | 0 |
| 第 1 个请求，SWE-rebench | 18 | +52.7 | 88.9% | −7.2 / −2.9 | 37.2 | 0 |
| 之后，WildChat，思考开 | 225 | −135.9 | 22.7% | −0.4 / −2.6 | 30.4 | 288 |
| 之后，WildChat，思考关 | 225 | −97.0 | 30.7% | +1.5 / −0.6 | 28.5 | 256 |
| 之后，SWE-smith，思考开 | 243 | −241.1 | 13.2% | −0.2 / −3.5 | 30.2 | 416 |
| 之后，SWE-smith，思考关 | 187 | −203.1 | 23.5% | −9.1 / +0.8 | 39.1 | 384 |
| 之后，SWE-rebench，思考开 | 319 | −204.0 | 14.1% | −6.1 / −4.7 | 36.1 | 384 |
| 之后，SWE-rebench，思考关 | 318 | −204.7 | 13.5% | −3.6 / −1.9 | 33.6 | 384 |

| 预测 | 结果 | 数字 |
|---|---|---|
| W0 对照 | 通过 | 同一配置两遍之差都在 ±10 毫秒以内（最大 −9.1），人为加的 30 毫秒量出 28.5–39.1 |
| W1 多停一步的代价随层数涨（至少 9B 的 1.5 倍） | 成立 | 不多停的 10 个冷请求 −1.3 毫秒，多停一步的 102 个 +59.8 毫秒，是 9B 的 1.56 倍 |
| W2 每个 token 更贵 | 成立 | 默认配置的斜率 0.69–0.78 毫秒 / token（9B 是 0.152–0.172） |
| W3 打平点 | 方向成立，数值低于预计的 100–250 | 59.8 ÷ 0.69–0.78 = 77–86 个 token |
| W4 Agent 四格净效果转正 | 成立 | −241、−203、−204、−205 毫秒 |

读法：这个参数在 27B 上是净赚的。之后的请求里 69%–87% 变快，中位数快 97–241 毫秒，只有每个会话的第 1 个请求（没有可省的重算）慢 42–63 毫秒。那一步的代价跟着层数涨，但没有翻倍（对 9B 是 1.56 倍）。收益涨得快得多，一是每个 token 的 prefill 代价约是 9B 的 4.5 倍，二是 832 的块更大，默认配置每轮多重算得更多，设了参数省下 256–416 个 token（4B、9B 是 176–288 个）。

## 局限

- 条件和 4B、9B 不同：FP8（只量化权重）、功耗上限 250 W、绑核、`--max-num-seqs 64`、KV 10 GiB。「每个 token 贵 4.5 倍」里模型变大、FP8 跑法和功耗上限各占多少分不开，三个模型的绝对数不能连成一条线。能直接比的只有同一批里默认和粒度 16 的差。
- 会话数是 E3 的一半，有一格只有 18 对。
- bf16 的 27B 没有测。只有 L40 和 Triton 后端。

## 复现

```bash
python scripts/setup/fetch_assets.py --manifest scripts/setup/manifests/qwen3.5-27b-fp8.json --root $HP_ROOT
export HP_ROOT HP_ENV=$HP_ROOT/envs/vllm-main HP_QWEN35_DIR=$HP_ROOT/models/Qwen3.5-27B-FP8 HP_QWEN3_DIR=$HP_ROOT/models/Qwen3-4B GATE_GPU_UUID=<gpu-uuid>
taskset -c <GPU 所在 NUMA 节点的核> experiments/e6_27b/run.sh <run_dir>     # 约 3.5 小时
python experiments/e2_ttft_cost/judge.py <run_dir>/e1 --json judge.json
python experiments/e2_ttft_cost/locate_overhead.py <run_dir>/e1 --block 832 --json locate.json
```

逐请求的原始记录没有公开，需要的话可以索取。

## 文件

- `run.sh`：E6 的参数（会话数、KV 10 GiB、块大小 832、`--max-num-seqs 64`、端口 8063），调用 `scripts/run_replay.sh`。
- `results/judge.json`、`judge.txt`、`locate.json`：`experiments/e2_ttft_cost/judge.py` 和 `locate_overhead.py --block 832` 在正式批次上的输出。

---

## Question

On 4B (E2) and 9B (E3) the extra prefill step of `--prefix-match-unit 16` costs 34.5 and 38.2 ms and the net effect is a slowdown. 4B and 9B have 32 layers, 27B has 64 (48 linear-attention plus 16 full-attention layers, read from each `config.json`). If the step cost follows the number of kernel launches, it should grow clearly on 27B; the saved recompute is also worth more. E6 asks whether the net effect turns into a speed-up one size up.

## Setup

`Qwen/Qwen3.5-27B-FP8` (official FP8 build, revision `97f5941b…`); the bf16 weights (55.6 GB) do not fit a 48 GB L40. The L40 has no native FP8 compute, so vLLM runs weight-only FP8 (`MarlinFP8ScaledMMLinearKernel`); the weights take 27.9 GiB. Block size 832 tokens (576 on 4B and 9B), Triton/FLA GDN prefill backend. Same four-pass design (C0a, C1a, C1b, C0b), seeds 4-6, client and judge scripts as E2 and E3. Changes: 15 / 5 / 3 sessions per seed (half of E3, since 27B prefill is slower), a 10 GiB KV cache, `--max-num-seqs 64` (the default 256 needs 256 linear-attention state blocks, which do not fit), block-size argument 832 for the client and `locate_overhead.py`, the server pinned to the CPU cores of the GPU's NUMA node. One L40, vLLM `1479deb0`, 250 W power limit.

Predictions and judgement criteria were written down before each run and not changed afterwards.

## Results

Bookkeeping: 6,348 scored requests, 0 errors, 0 prompt-length mismatches, `/metrics` reconciled in all 72 runs, block size 832 and the Triton/FLA backend in all four passes, no CUDA errors in the server logs.

`judge.py` reports `incomplete`: the first-request SWE-rebench cell has 18 pairs, fewer than the 20 the script requires (a consequence of halving the sessions). The table above gives every cell (ms, paired medians; negative means faster with the flag); W0-W4 are judged by their own criteria. W0 passed (all same-config differences within ±10 ms, largest −9.1; injected 30 ms measured as 28.5-39.1). W1 held: 10 cold requests without an extra step −1.3 ms, 102 with one +59.8 ms, 1.56 times the 9B value. W2 held: default-config slopes 0.69-0.78 ms per token (9B: 0.152-0.172). W3: direction held, value below the predicted 100-250 (59.8 / 0.69-0.78 = 77-86 tokens). W4 held: the four agent cells are −241, −203, −204, −205 ms.

Reading: on 27B the flag is a net gain. 69-87% of later requests get faster, by a median 97-241 ms; only the first request of each session (nothing to save) is 42-63 ms slower. The step cost grows with the layer count but does not double (1.56 times 9B). The gain grows much faster: the per-token prefill cost is about 4.5 times the 9B value, and the larger 832-token block makes the default config recompute more per turn, so the flag saves 256-416 tokens (176-288 on 4B and 9B).

## Limits

Conditions differ from 4B and 9B: weight-only FP8, 250 W, CPU pinning, `--max-num-seqs 64`, 10 GiB KV cache. How much of the 4.5 times per-token cost comes from model size, the FP8 path or the power limit cannot be separated, so the absolute numbers of the three models do not form a curve; only the within-run difference between default and unit 16 is comparable. Half the sessions of E3, one cell with 18 pairs. bf16 27B not tested; L40 and Triton only.

## Reproduce

Commands above. Raw per-request records are not published; they are available on request.

## Files

`run.sh` E6 settings for `scripts/run_replay.sh` (sessions, 10 GiB KV, block 832, `--max-num-seqs 64`, port 8063); `results/` the outputs of `experiments/e2_ttft_cost/judge.py` and `locate_overhead.py --block 832` on the formal run.
