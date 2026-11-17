# E1：混合模型少命中多少，现有参数补回多少 / E1: How many prefix hits a hybrid model loses, and what the existing flags recover

## 问题

在真实的多轮对话和编码 Agent 轨迹上，vLLM 里的 Qwen3.5-4B（Gated DeltaNet 加全注意力的混合模型）比同规模的全注意力模型 Qwen3-4B 少命中多少前缀、每轮多重算多少 token？vLLM 现有的两个参数（`--prefix-match-unit`、`--prefix-cache-retention-interval`）各补回多少？另外检验一个离线记账模型（E0，只用 CPU）能不能逐请求预测出 vLLM 的命中。

## 做法

- 硬件和软件：单张 NVIDIA L40（48 GB），vLLM main `1479deb0`，bf16，KV 缓存固定 24 GiB，不开投机解码，请求一律贪心解码。
- 数据：WildChat-1M（第 8 个分片，用户轮数不少于 3），SWE-smith 轨迹（`tool` 第 4 个分片），SWE-rebench OpenHands 轨迹。一个会话里每条 assistant 消息对应一个请求，prompt 是它之前的全部消息套上模型自己的对话模板。`max-model-len` 32768，会话在第一个超长的请求处截断。
- 服务端配置（`scripts/run_replay.sh`）：F0 是 Qwen3-4B 默认配置，F1 是 Qwen3-4B 套用 Qwen3.5 的模板（只报告）。C0 是 Qwen3.5-4B 默认配置，C1 加 `--prefix-match-unit 16`，C2 加 `--prefix-cache-retention-interval 576`，C3 是 C1 加 C2，C4 是 unit 16 加每块都保留。
- 回放：R（历史用数据集记录的消息，输出上限 16 个 token 后丢弃，7 组都跑）和 O（历史换成模型自己的输出，`max_tokens` 512，F0、F1、C1、C4）。每个种子抽 WildChat 100 个、SWE-smith 30 个、SWE-rebench 20 个会话（O 回放 10 / 6 / 6），种子 1、2、3，思考开和关各一遍。逐会话发送，F0、C0、C3 另跑 8 个会话轮转。
- 对照：每次回放前重置前缀缓存，先发三条对照请求（新 prompt 必须不命中，原样重发必须命中，无关 prompt 必须不命中），任何一条不符合就停掉整个批次。本地渲染的 prompt 长度和服务端的 `prompt_tokens` 逐请求核对。`/metrics` 的命中计数器增量和逐请求求和核对。
- 判死标准 1（事先定下）：F0 对 C1、C4 中较好的一组，每个数据集、思考开和关各算一次。R 回放的命中率差小于 10 个百分点，并且 R 和 O 回放下每轮重算中位数之比都小于 1.5，才算「现有参数已经补回」。
- E0（`account.py`）：同样三份数据，每个种子 300 / 100 / 100 个会话，按 vLLM 的存档规则逐请求推算各配置应当命中多少（`hplab/cache_model.py`）。

每个实验开跑前都先写下预测和判定标准，结果出来后不改预测。

## 结果

口径核对：7 组共 114,681 个会话请求，0 报错。本地渲染的 prompt 长度和服务端 0 个不相等。252 组回放的 `/metrics` 增量都等于逐请求求和。对照请求 252 组都符合预期。

两个块大小：混合模型的缓存块是 576 个 token，全注意力模型是 64 个（`vllm/config/cache.py:185` 的 `DEFAULT_BLOCK_SIZE = 64`）。F0 的命中 100% 等于 `floor(min(D, P − 1) / 64) × 64`，D 是和更早 prompt 的最长公共前缀。

R 回放、逐会话，每格是「命中率 / 每轮重算中位数（token）」，3 个种子合并，每个会话的第 1 个请求不计入：

| 数据集 | 思考 | F0（全注意力） | C0（混合，默认） | C1（混合，unit 16） |
|---|---|---|---|---|
| WildChat | 开 | 71.40% / 333 | 56.09% / 532 | 72.34% / 293 |
| WildChat | 关 | 71.19% / 337 | 56.01% / 534 | 70.31% / 315 |
| SWE-smith | 开 | 93.91% / 410 | 92.02% / 709 | 93.89% / 418 |
| SWE-smith | 关 | 93.89% / 414 | 92.00% / 711 | 93.72% / 446 |
| SWE-rebench | 开 | 96.18% / 462 | 94.80% / 774 | 96.16% / 460 |
| SWE-rebench | 关 | 96.16% / 466 | 94.81% / 772 | 96.16% / 462 |

- 默认配置下，混合模型每轮重算是全注意力的 1.58–1.73 倍，WildChat 上命中率低 15.2–15.3 个百分点。
- 判死标准 1 触发。C1 和 C4 的重算中位数都是 397，按约定取 C1。六格的命中率差在 −0.95 到 0.88 个百分点之间，R 回放的重算之比 0.88–1.08，O 回放 0.54–1.09。
- 预测：H3（C0 命中上一个请求 prompt 末尾的 576 存档点）成立，99.875% 相等，加上回退规则后 100%。H4（C1 命中 `floor((P_prev − 1) / 16) × 16`）成立，C1 是 100%，C3、C4 是 99.67%。H5（模板造成的周期性未命中）成立，WildChat 上思考开 6.86%（预测 6.25%）、思考关 20.80%（预测 18.75%）的请求落在不可用的余数上，它们 100% 没对上上一轮的存档。H2 不成立，原因是全注意力的块是 64 而不是预测时写的 16。H6 部分成立，C1 的命中从不进到上一轮回答里，C4 有 13.1%–17.6% 的请求进到了。
- 保留间隔单独设几乎没有效果。C2 和 C0、C3 和 C1 的命中率差都不超过 0.04 个百分点。
- 8 个会话轮转时，F0 在 SWE-rebench 上命中率降 0.21–0.26 个百分点（24 GiB 只装得下 174,720 个 token 的全注意力 KV），混合模型六格不变。
- E0 的离线记账和实测一致。E0 按块大小 528 和 16 推算，默认配置下 WildChat 的差距是 16.3 / 16.2 个百分点，实测是 15.3 / 15.2（实际块大小是 576 和 64）。C0 的命中和按 576 推算的值逐请求相等（上面 H3 的 100%）。用 576 重算 E0 的 80,321 条记录，其余字段和原记录 0 条不同。
- 事后比较（看到结果之后才做，不在事先的判定里）：同一请求配对，C1 比 C0 的首个 token 慢。每个会话的第 1 个请求慢 28–33 毫秒，之后的请求每轮少重算 144–288 个 token，仍慢 27–32 毫秒。这一条由 E2 用事先写好的预测复核。

## 局限

- 只有一种 GPU、一个 4B 模型、一次一个请求（另加一档 8 个会话轮转）、输出上限 16。
- SWE-rebench 有 95% 以上的轨迹在 32,768 个 token 处截断，只覆盖轨迹的前半段。
- O 回放每格的请求少（WildChat 每格 114 个），各模型带回的是自己的输出，请求不是同一批。
- 全注意力模型没有测「也设 `--prefix-match-unit 16`」的一组。
- SWE-smith 上约 11% 的请求没有 TTFT（`qwen3_xml` 解析器吞掉了没写完的工具调用，见 E2），命中和重算不受影响。
- C4 的命中能进到回答里的原因是推断，没有读源码确认。混合模型和全注意力模型之间的 TTFT 不宜直接比较。

## 复现

```bash
# 环境、数据、模型（scripts/setup/）
bash scripts/setup/setup_env.sh $HP_ROOT
python scripts/setup/fetch_assets.py --manifest scripts/setup/manifests/datasets.json --root $HP_ROOT
python scripts/setup/fetch_assets.py --manifest scripts/setup/manifests/qwen3-4b.json --root $HP_ROOT
# Qwen3.5-4B 另行下载（Qwen/Qwen3.5-4B），它的模板应和 refs/ 里的一致
export HP_ROOT HP_ENV=$HP_ROOT/envs/vllm-main HP_QWEN35_DIR=<Qwen3.5-4B> HP_QWEN3_DIR=$HP_ROOT/models/Qwen3-4B GATE_GPU_UUID=<gpu-uuid>
experiments/e1_prefix_hits/run_account.sh <run_dir_e0>     # E0，只用 CPU
experiments/e1_prefix_hits/run.sh <run_dir>                 # E1，约 25 小时
# 判定和汇总（本机即可）
python experiments/e1_prefix_hits/judge.py <run_dir>/e1 --json judge.json
python experiments/e1_prefix_hits/posthoc.py <run_dir>/e1 --e0b <run_dir>/e0b --e0 <run_dir_e0>/e0 --json posthoc.json
python experiments/e1_prefix_hits/analyze_account.py <run_dir_e0>/e0 --json account_analysis.json
```

逐请求的原始记录没有公开（WildChat 来自真实用户），需要的话可以索取。

## 文件

- `account.py`：离线记账（E0），渲染每个请求并按 vLLM 的规则推算各配置的命中。
- `run_account.sh`：在服务器上对三份数据并行跑 `account.py`。
- `replay.py`：回放客户端，逐请求记录服务端的命中、TTFT 和离线预测。
- `run.sh`：E1 的参数，调用 `scripts/run_replay.sh`。
- `judge.py`：事先定下的判定（H2–H9、判死标准 1）。
- `posthoc.py`：事后比较（64 对齐、配对 TTFT、E0 用 576 重算）。
- `analyze_account.py`：汇总 E0 的记账记录。
- `results/judge.json`、`judge.txt`、`posthoc.json`、`account_analysis.json`：上面三个脚本在正式批次上的输出。

---

## Question

On real multi-turn chats and coding-agent trajectories, how many prefix-cache hits does Qwen3.5-4B (Gated DeltaNet plus full attention) lose against the full-attention Qwen3-4B in vLLM, and how many more tokens does it recompute per turn? How much do the two existing flags (`--prefix-match-unit`, `--prefix-cache-retention-interval`) recover? And can an offline accounting model (E0, CPU only) predict vLLM's hits request by request?

## Setup

- One NVIDIA L40 (48 GB), vLLM main `1479deb0`, bf16, KV cache fixed at 24 GiB, no speculative decoding, greedy decoding.
- Data: WildChat-1M (shard 8, at least 3 user turns), SWE-smith trajectories (`tool` shard 4), SWE-rebench OpenHands trajectories. Each assistant message of a session is one request whose prompt is everything before it under the model's own chat template. `max-model-len` 32768; a session ends at its first over-long request.
- Server configs (`scripts/run_replay.sh`): F0 Qwen3-4B defaults, F1 Qwen3-4B with the Qwen3.5 template (report only), C0 Qwen3.5-4B defaults, C1 `--prefix-match-unit 16`, C2 `--prefix-cache-retention-interval 576`, C3 = C1 + C2, C4 unit 16 plus every block retained.
- Replay: R (recorded history, output capped at 16 tokens and dropped, all 7 configs) and O (the model's own outputs carried back, `max_tokens` 512, F0 F1 C1 C4). Per seed 100 / 30 / 20 sessions (O: 10 / 6 / 6), seeds 1-3, thinking on and off, one session at a time; F0, C0, C3 also 8 sessions round-robin.
- Controls: before each replay the prefix cache is reset and three control requests are sent (fresh prompt must miss, resent prompt must hit, unrelated prompt must miss); any failure stops the batch. Local prompt lengths are checked against the server's `prompt_tokens`, and `/metrics` hit counters against per-request sums.
- Kill criterion 1 (fixed in advance): F0 against the better of C1 and C4, per dataset and thinking setting. The flags count as having closed the gap when the R-replay hit-rate gap is below 10 points and the recompute-median ratio is below 1.5 in both R and O replay.
- E0 (`account.py`): the same three datasets, 300 / 100 / 100 sessions per seed, predicted hits per request from vLLM's checkpoint rules (`hplab/cache_model.py`).

Predictions and judgement criteria were written down before each run and not changed afterwards.

## Results

Bookkeeping: 114,681 session requests in 7 configs, 0 errors, 0 prompt-length mismatches, `/metrics` deltas equal to per-request sums in all 252 replay runs, all control requests as expected.

Block sizes: 576 tokens for the hybrid model, 64 for the full-attention model (`vllm/config/cache.py:185`, `DEFAULT_BLOCK_SIZE = 64`). F0 hits equal `floor(min(D, P − 1) / 64) × 64` for 100% of requests.

The table above (R replay, one session at a time, hit rate / recompute median in tokens, seeds pooled, first request of each session excluded) gives the measured numbers.

- With defaults the hybrid model recomputes 1.58-1.73 times as many tokens per turn, and its hit rate on WildChat is 15.2-15.3 points lower.
- Kill criterion 1 triggered. C1 and C4 both have a recompute median of 397, so C1 is used. Hit-rate gaps are −0.95 to 0.88 points, recompute ratios 0.88-1.08 in R replay and 0.54-1.09 in O replay.
- Predictions: H3 (C0 hits the previous request's 576-token checkpoint) held, 99.875% (100% with the fallback rule). H4 (C1 hits `floor((P_prev − 1) / 16) × 16`) held, C1 100%, C3 and C4 99.67%. H5 (periodic misses caused by the chat template) held: 6.86% (predicted 6.25%) of WildChat requests with thinking on and 20.80% (predicted 18.75%) with thinking off fall on unusable residues, and all of them miss the previous checkpoint. H2 failed because the full-attention block is 64, not the 16 assumed. H6 held in part: C1 never hits into the previous reply, C4 does for 13.1-17.6% of requests.
- The retention interval alone has almost no effect: C2 vs C0 and C3 vs C1 differ by at most 0.04 points.
- With 8 sessions round-robin, F0 loses 0.21-0.26 points on SWE-rebench (24 GiB holds 174,720 full-attention tokens); the hybrid configs are unchanged.
- E0 agrees with the measurement. C0 hits equal the value computed with block 576 for every request (the 100% of H3 above); the default-config WildChat gap it predicted with blocks of 528 and 16 was 16.3 / 16.2 points (measured 15.3 / 15.2; the real blocks are 576 and 64). Recomputing its 80,321 records with block 576 changed no other field.
- Post hoc (chosen after seeing the results, not a pre-registered test): paired on the same request, C1 delivers the first token later than C0, by 28-33 ms on the first request of a session and by 27-32 ms on later requests despite recomputing 144-288 fewer tokens. E2 re-tests this with predictions written beforehand.

## Limits

One GPU type, one 4B model, one request at a time (plus one 8-session round-robin level), output cap 16. Over 95% of SWE-rebench trajectories are truncated at 32,768 tokens. O replay has few requests per cell and each model carries back its own outputs. Full attention with `--prefix-match-unit 16` was not tested. About 11% of SWE-smith requests have no TTFT (the `qwen3_xml` parser drops unfinished tool calls, see E2); hits and recompute are unaffected. Why C4 hits reach into replies is inferred, not confirmed from the source. TTFT across the two different models is not directly comparable.

## Reproduce

See the commands above. Raw per-request records are not published (WildChat comes from real users); they are available on request.

## Files

`account.py` offline accounting (E0); `run_account.sh` runs it on the server; `replay.py` replay client; `run.sh` E1 settings for `scripts/run_replay.sh`; `judge.py` pre-registered verdicts; `posthoc.py` post-hoc comparisons; `analyze_account.py` summary of the E0 records; `results/` the outputs of the three analysis scripts on the formal run.
