# E2：补回命中以后首个 token 为什么变慢 / E2: Why the first token gets slower after the hits are recovered

## 问题

E1 事后看到，Qwen3.5-4B 设了 `--prefix-match-unit 16` 以后命中更多，首个 token 却慢约 30 毫秒。E2 用事先写好的预测和对照复核这一点，再查它是不是一个和重算量无关的固定开销、少重算省下的时间能不能抵掉它，以及开销来自哪里。

## 做法

- 和 E1 相同的服务端命令、客户端、数据和环境（单张 L40，vLLM `1479deb0`，Qwen3.5-4B，KV 缓存 24 GiB，R 回放，逐会话，输出上限 16，思考开和关）。新种子 4、5、6，每个种子 30 / 10 / 6 个会话。
- 四遍：C0a、C1a、C1b、C0b（C0 默认，C1 加 `--prefix-match-unit 16`），每遍重新起服务，同一批请求每个请求发四次。首尾是 C0，中间是 C1，机器状态随时间匀速变化时两边受到的影响相同。
- 统计量（`judge.py`）：主效应 `(C1a + C1b) / 2 − (C0a + C0b) / 2`，逐请求配对后取中位数。已知为零的对照是 `C0b − C0a` 和 `C1b − C1a`（中位数要在 ±10 毫秒以内）。已知不为零的对照是人为加 30 毫秒（`(C0a + 30) − C0b`，必须量得出来）。对照不过就不下结论。
- 分格：每个会话的第 1 个请求按数据集分 3 格，之后的请求按数据集和思考分 6 格，每格至少 20 对。
- 定位（`locate_overhead.py`）：按 vLLM 调度器 `_mamba_block_aligned_split` 的规则推算每个请求的 prefill 要分几步，再按「多停几步」分组看 TTFT 的差。
- 诊断：最后一遍之后，把 SWE-smith 上流里没有内容增量的请求不带流式重发一次并取回生成的 token，查 TTFT 缺失的原因。

每个实验开跑前都先写下预测和判定标准，结果出来后不改预测。

## 结果

口径核对：四遍各 3,433 个会话请求、3,161 个计分请求，0 报错，prompt 长度 0 个不相等，72 组 `/metrics` 对账相等，对照请求都符合预期。

判定是**确认**（`judge.py` 输出 `confirmed`）。单位毫秒，逐请求配对的中位数：

| 格 | 配对数 | 主效应 | C1 更慢的比例 | `C0b − C0a` | `C1b − C1a` | 人为加 30 | 每轮少重算（token） |
|---|---|---|---|---|---|---|---|
| 第 1 个请求，WildChat | 176 | 34.6 | 96.0% | 0.3 | 0.4 | 29.7 | 0 |
| 第 1 个请求，SWE-smith | 60 | 33.0 | 98.3% | 0.2 | 0.9 | 29.8 | 0 |
| 第 1 个请求，SWE-rebench | 36 | 23.2 | 97.2% | 1.6 | 0.0 | 28.4 | 0 |
| 之后，WildChat，思考开 | 420 | 31.9 | 95.7% | 1.1 | 1.5 | 28.9 | 192 |
| 之后，WildChat，思考关 | 420 | 33.0 | 97.4% | 0.4 | 0.2 | 29.6 | 176 |
| 之后，SWE-smith，思考开 | 543 | 27.5 | 96.1% | 0.0 | 0.9 | 30.0 | 288 |
| 之后，SWE-smith，思考关 | 443 | 31.6 | 96.6% | −0.3 | 0.5 | 30.3 | 256 |
| 之后，SWE-rebench，思考开 | 600 | 27.4 | 95.8% | 0.5 | −0.2 | 29.5 | 256 |
| 之后，SWE-rebench，思考关 | 599 | 28.2 | 95.8% | −0.6 | 0.5 | 30.6 | 256 |

- V0（对照）通过，两个已知为零的对照都在 ±1.6 毫秒以内。V1（第 1 个请求有固定开销）成立。V2（之后的请求少重算 176–288 个 token 仍然更慢）成立。
- TTFT 对重算量的斜率四遍都是 0.080–0.090 毫秒 / token。截距（WildChat / SWE-smith / SWE-rebench）C0a 是 83.6 / 159.9 / 244.9 毫秒，C0b 是 83.9 / 159.0 / 244.2，C1a 是 123.0 / 210.1 / 296.7，C1b 是 123.8 / 210.3 / 297.2。
- 定位：按调度器规则，设了 unit 16 以后 prefill 要在 prompt 末尾 `floor((P − 1) / 16) × 16` 处多停一步，因为 Triton 的 GDN prefill 核不能在最后一段里顺带保存状态。没有命中缓存的第 1 个请求里，按规则不多停的 22 个（prompt 11–16 个 token）差 0.34 毫秒，多停一步的 194 个差 34.5 毫秒（M1 成立）。扣掉重算量后，多停 1 步慢 33.8–37.6 毫秒，多停 2 步慢 58.3–73.4 毫秒（M3 成立）。把 TTFT 差对步数差和重算差做线性回归（M2）不成立，每个 token 的系数低于预测，两个自变量相互纠缠。
- 诊断：SWE-smith 上没有 TTFT 的请求，模型的回答一上来就是工具调用的标记（`<tool_call>` 或 `<function=…>`），16 个 token 时还没写完。`qwen3_xml` 解析器把这段从内容里拿掉，又没有解析成工具调用，所以流里一个内容增量也没有。客户端另记的 `first_chunk_ms` 在全部请求上都有值，和 `ttft_ms` 的差中位 0.25 毫秒。

## 局限

- 只有 L40 一种显卡和 Triton 后端。按 vLLM 源码，prefill 后端是 FlashInfer 时（Hopper、部分 Blackwell）存档点在最后一段内部保存，不需要多停，那种情况没有测。
- 一次一个请求，输出上限 16，一个 4B 模型。更大的模型见 E3、E6。
- 第 1 个请求里有 56 个命中过缓存（同一数据集的会话开头相同），M1 只用没有命中的 216 个。
- 每一步的时间具体花在哪由 E4 用 Nsight Systems 拆开。

## 复现

```bash
export HP_ROOT HP_ENV=$HP_ROOT/envs/vllm-main HP_QWEN35_DIR=<Qwen3.5-4B> HP_QWEN3_DIR=$HP_ROOT/models/Qwen3-4B GATE_GPU_UUID=<gpu-uuid>
experiments/e2_ttft_cost/run.sh <run_dir>                     # 约 2 小时
python experiments/e2_ttft_cost/judge.py <run_dir>/e1 --json judge.json
python experiments/e2_ttft_cost/locate_overhead.py <run_dir>/e1 --block 576 --json locate.json
```

`judge.py` 和 `locate_overhead.py` 也用于 E3、E6、E7、E8 的四遍对比。逐请求的原始记录没有公开，需要的话可以索取。

## 文件

- `run.sh`：E2 的参数，调用 `scripts/run_replay.sh`。
- `judge.py`：四遍对比的判定（主效应、两个零对照、人为加 30 毫秒的对照）。
- `locate_overhead.py`：按调度器规则推算 prefill 步数，按多停几步分组。
- `results/judge.json`、`judge.txt`、`locate.json`：两个脚本在正式批次上的输出。

---

## Question

In E1, seen after the fact: with `--prefix-match-unit 16` Qwen3.5-4B hits more but delivers the first token about 30 ms later. E2 re-tests this with predictions and controls written beforehand, asks whether it is a fixed cost independent of recompute, whether the saved recompute pays for it, and where it comes from.

## Setup

Same server command, client, data and environment as E1 (one L40, vLLM `1479deb0`, Qwen3.5-4B, 24 GiB KV cache, R replay one session at a time, output cap 16, thinking on and off), new seeds 4-6 with 30 / 10 / 6 sessions per seed. Four passes C0a, C1a, C1b, C0b (C0 defaults, C1 `--prefix-match-unit 16`), a fresh server each, the same requests in each, so every request has four TTFTs; the ABBA order cancels a steady drift. `judge.py` takes the paired median of `(C1a + C1b) / 2 − (C0a + C0b) / 2`, with two known-zero controls (`C0b − C0a`, `C1b − C1a`, within ±10 ms) and a known-positive control (30 ms added to C0a must be detected); no verdict if a control fails. Cells: first requests per dataset (3), later requests per dataset and thinking (6), at least 20 pairs each. `locate_overhead.py` counts prefill steps from the rules of vLLM's `_mamba_block_aligned_split` and groups requests by the extra steps. A diagnostic run re-sent SWE-smith requests whose stream had no content delta, without streaming, and fetched the generated tokens.

Predictions and judgement criteria were written down before each run and not changed afterwards.

## Results

Bookkeeping: 3,433 session requests and 3,161 scored requests per pass, 0 errors, 0 prompt-length mismatches, `/metrics` reconciled in all 72 runs, all controls as expected.

Verdict **confirmed** (table above; ms, paired medians). V0 (controls) passed, both known-zero controls within ±1.6 ms. V1 (fixed cost on first requests) and V2 (later requests slower despite 176-288 fewer recomputed tokens) held. TTFT slopes are 0.080-0.090 ms per token in every pass; intercepts (WildChat / SWE-smith / SWE-rebench) are 83.6 / 159.9 / 244.9 ms for C0a, 83.9 / 159.0 / 244.2 for C0b, 123.0 / 210.1 / 296.7 for C1a and 123.8 / 210.3 / 297.2 for C1b.

Location: with unit 16 the scheduler must stop the prefill once more at `floor((P − 1) / 16) × 16`, because the Triton GDN prefill kernel cannot save the state inside the final chunk. Among cold first requests, the 22 that need no extra step (prompts of 11-16 tokens) differ by 0.34 ms and the 194 with one extra step by 34.5 ms (M1 held). After removing the recompute term, one extra step costs 33.8-37.6 ms and two cost 58.3-73.4 ms (M3 held). The linear regression on step and recompute differences (M2) failed: the per-token coefficient came out below the prediction because the two regressors are entangled.

Diagnostic: the SWE-smith requests without TTFT start their answer with a tool-call marker (`<tool_call>` or `<function=…>`) that is unfinished at 16 tokens; the `qwen3_xml` parser removes it from the content without producing a tool call, so the stream carries no content delta. The `first_chunk_ms` the client also records exists for every request and differs from `ttft_ms` by a median 0.25 ms.

## Limits

One GPU type and the Triton backend only; with the FlashInfer prefill backend (Hopper, some Blackwell GPUs) vLLM saves the checkpoint inside the final chunk and needs no extra step, which was not tested. One request at a time, output cap 16, one 4B model (larger models: E3, E6). 56 first requests hit the cache (sessions of a dataset share an opening), so M1 uses the other 216. Where the time of the extra step goes is taken apart in E4.

## Reproduce

Commands above. `judge.py` and `locate_overhead.py` are also used for the four-pass runs of E3, E6, E7 and E8. Raw per-request records are not published; they are available on request.

## Files

`run.sh` E2 settings for `scripts/run_replay.sh`; `judge.py` four-pass verdict with controls; `locate_overhead.py` prefill step counts from the scheduler rules; `results/` their outputs on the formal run.
