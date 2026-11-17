# E3：换成 9B 模型 / E3: The same comparison on Qwen3.5-9B

## 问题

E2 在 Qwen3.5-4B 上查明，`--prefix-match-unit 16` 让 prefill 多停一步，代价约 34.5 毫秒，比少重算省下的时间多。模型变大以后，每个 token 的 prefill 更贵，省下的重算更值钱，而多停一步里的固定部分未必变。E3 问：多停一步的代价、每个 token 的代价和两者相除的「打平点」在 9B 上各是多少，净效果有没有转成持平或变快。

## 做法

- 模型 Qwen3.5-9B（bf16，revision `c2022362…`，32 层，隐藏维度 4,096；4B 是 32 层、2,560），块大小同样是 576，GDN prefill 后端同样是 Triton，对话模板和 4B 逐字节相同。
- 和 E2 完全相同的四遍设计（C0a、C1a、C1b、C0b）、种子（4、5、6）、会话数（30 / 10 / 6）、客户端和判定脚本（`experiments/e2_ttft_cost/judge.py`、`locate_overhead.py --block 576`）。同样的种子抽到同样的对话，两个模型发的是同一批请求。
- 只改一处：KV 缓存 16 GiB（9B 的权重约 18 GiB）。逐会话发送，缓存用不完。单张 L40，vLLM `1479deb0`，功耗上限 300 W。
- `summarize.py` 从 E2 和 E3 的判定输出里算 W1–W4。

每个实验开跑前都先写下预测和判定标准，结果出来后不改预测。

## 结果

口径核对：四遍各 3,433 个会话请求、3,161 个计分请求，0 报错，prompt 长度 0 个不相等，72 组 `/metrics` 对账相等，对照请求都符合预期。命中率和重算量和 E2 逐项相同。

`judge.py` 的总判定是**部分**（`mixed`）：9 格里 4 格明显更慢，没有一格算不慢。对照通过，但比 E2 松，同一配置两遍之差最大 8.1 毫秒（判定线 10）。之后的请求（单位毫秒，逐请求配对的中位数）：

| 格 | 主效应（9B） | 粒度 16 更慢的比例 | 每轮少重算（token） | E2（4B）同一格 |
|---|---|---|---|---|
| WildChat，思考开 | 26.0 | 80.2% | 192 | 31.9 |
| WildChat，思考关 | 27.6 | 76.4% | 176 | 33.0 |
| SWE-smith，思考开 | 13.4 | 65.3% | 288 | 27.5 |
| SWE-smith，思考关 | 15.8 | 65.0% | 256 | 31.6 |
| SWE-rebench，思考开 | 12.6 | 66.8% | 256 | 27.4 |
| SWE-rebench，思考关 | 8.7 | 58.9% | 256 | 28.2 |

| 预测 | 结果 | 数字 |
|---|---|---|
| W0 对照 | 通过 | 两个零对照最大 8.1 毫秒，人为加的 30 毫秒量出 26.6–33.3 |
| W1 多停一步的代价还在 | 成立 | 不多停的 22 个冷请求差 1.7 毫秒，多停一步的 194 个差 38.2 毫秒（4B 是 0.3 和 34.5） |
| W2 每个 token 更贵 | 成立 | 默认配置的斜率 0.152–0.172 毫秒 / token（4B 是 0.080–0.083） |
| W3 打平点下降 | 成立 | 38.2 ÷ 斜率 = 222–252 个 token（4B 是 413–429） |
| W4 净效果比 4B 小 | 成立 | 六格都低于 4B 同一格 |

读法：多停一步的代价基本不随模型变大（34.5 到 38.2 毫秒），每个 token 的代价约翻倍。9B 上每轮少重算的 176–288 个 token 落在打平点附近，所以净效果接近打平，但六格都还是变慢（对话 26–28 毫秒，Agent 9–16 毫秒）。

## 局限

- 只多了一个模型大小，两个点不能当成一条随模型大小变化的曲线。更大的模型见 E6。
- SWE-rebench 思考关那一格的主效应（8.7 毫秒）和两遍之间的波动（最大 8.1 毫秒）是同一量级。
- 只有 L40 和 Triton 后端，一次一个请求，输出上限 16。
- 第 1 个请求里有 56 个命中过缓存，W1 只用没有命中的 216 个。

## 复现

```bash
python scripts/setup/fetch_assets.py --manifest scripts/setup/manifests/qwen3.5-9b.json --root $HP_ROOT
export HP_ROOT HP_ENV=$HP_ROOT/envs/vllm-main HP_QWEN35_DIR=$HP_ROOT/models/Qwen3.5-9B HP_QWEN3_DIR=$HP_ROOT/models/Qwen3-4B GATE_GPU_UUID=<gpu-uuid>
experiments/e3_9b/run.sh <run_dir>                            # 约 3 小时
python experiments/e2_ttft_cost/judge.py <run_dir>/e1 --json experiments/e3_9b/results/judge.json
python experiments/e2_ttft_cost/locate_overhead.py <run_dir>/e1 --block 576 --json experiments/e3_9b/results/locate.json
python experiments/e3_9b/summarize.py --small experiments/e2_ttft_cost/results --large experiments/e3_9b/results --json summary.json
```

逐请求的原始记录没有公开，需要的话可以索取。

## 文件

- `run.sh`：E3 的参数，调用 `scripts/run_replay.sh`。
- `summarize.py`：从两个模型的 `judge.json`、`locate.json` 算 W1–W4。
- `results/judge.json`、`judge.txt`、`locate.json`：E2 的两个脚本在 9B 正式批次上的输出。`results/summary.json`：`summarize.py` 的输出。

---

## Question

E2 showed on Qwen3.5-4B that `--prefix-match-unit 16` adds one prefill step costing about 34.5 ms, more than the saved recompute is worth. A larger model makes each prefill token more expensive, so saved recompute is worth more, while the fixed part of the extra step may not grow. E3 asks what the step cost, the per-token cost and their ratio (the break-even) are on 9B, and whether the net effect turns even or faster.

## Setup

Qwen3.5-9B (bf16, revision `c2022362…`, 32 layers, hidden size 4,096; 4B has 32 layers and 2,560), block size 576, Triton GDN prefill backend, chat template byte-identical to 4B. Exactly the four-pass design of E2 (C0a, C1a, C1b, C0b), seeds 4-6, 30 / 10 / 6 sessions, client and judge scripts (`experiments/e2_ttft_cost/judge.py`, `locate_overhead.py --block 576`); the same seeds draw the same conversations, so both models get the same requests. Only change: a 16 GiB KV cache (the 9B weights take about 18 GiB). One L40, vLLM `1479deb0`, 300 W power limit. `summarize.py` computes W1-W4 from the judge outputs of E2 and E3.

Predictions and judgement criteria were written down before each run and not changed afterwards.

## Results

Bookkeeping: 3,433 session requests and 3,161 scored requests per pass, 0 errors, 0 prompt-length mismatches, `/metrics` reconciled in all 72 runs, all controls as expected; hit rates and recompute are identical to E2.

Overall verdict **mixed**: 4 of 9 cells clearly slower, none counted as not slower. Controls passed but looser than in E2 (largest same-config difference 8.1 ms, threshold 10). The later-request cells are in the first table. W0 passed (injected 30 ms measured as 26.6-33.3). W1 held: 22 cold requests without an extra step differ by 1.7 ms, the 194 with one by 38.2 ms (4B: 0.3 and 34.5). W2 held: default-config slopes 0.152-0.172 ms per token (4B: 0.080-0.083). W3 held: break-even 222-252 tokens (4B: 413-429). W4 held: all six later-request cells are below the 4B value.

Reading: the extra step barely grows with model size (34.5 to 38.2 ms) while the per-token cost about doubles. The 176-288 tokens saved per turn sit near the 9B break-even, so the net effect is close to even but still slower in all six cells (26-28 ms on chat, 9-16 ms on the agent datasets).

## Limits

One more model size only; two points are not a curve (a larger model: E6). The SWE-rebench thinking-off cell (8.7 ms) is of the same size as the pass-to-pass variation (up to 8.1 ms). L40 and Triton only, one request at a time, output cap 16. 56 first requests hit the cache, so W1 uses the other 216.

## Reproduce

Commands above. Raw per-request records are not published; they are available on request.

## Files

`run.sh` E3 settings for `scripts/run_replay.sh`; `summarize.py` W1-W4 from the judge and locate outputs of two models; `results/` the outputs of the E2 scripts on the 9B run and of `summarize.py`.
