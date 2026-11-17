# E4：用 Nsight Systems 拆解 prefill 多停的一步和两种注意力结构的需求 / E4: Profiling the extra prefill step and the needs of two attention designs

## 问题

E2 发现 `--prefix-match-unit 16` 让 Qwen3.5 每个请求的首 token 慢约 30 毫秒，原因是 prefill 在 prompt 末尾多停一个调度步。E4 回答四个子问题。

- Q1：多停的那一步里时间花在哪。
- Q2：decode 和长 prompt 的 prefill 分别受访存还是算力限制。
- Q2b：上下文变长时两种结构的 decode 各怎么变。
- Q3：两张卡做张量并行时通信占多少。

## 做法

- 硬件和软件：单张 L40（48 GB）。Q1 到 Q2b 时功耗上限 300 W，Q3 时 250 W（单卡基线在同一批里重量）。vLLM main，commit `1479deb0`，Triton GDN 后端。工具是 Nsight Systems 2025.1.3（CUDA 和 NVTX 追踪，`--cuda-graph-trace=node` 展开 CUDA graph）和 NVML。服务器上普通用户读不到硬件计数器，所以单个 kernel 的计数器数据在一台 RTX 5060 笔记本上用 Nsight Compute 2026.4.0 补抓，只用来检验方向。
- 模型：Qwen3.5-4B 和 Qwen3.5-9B（混合：Gated DeltaNet 加全注意力）对 Qwen3-4B（全注意力）。
- 每个配置跑两遍，一遍不带 profiler，量请求耗时；一遍在 Nsight Systems 下，按请求之间的停顿把时间线切成请求，按 vLLM 每步打的 NVTX 标记切成调度步。耗时比较只用不带 profiler 的那一遍。
- Q1：冷请求（命中为 0），prompt 长度 16、560、580、2,000、8,000，每档 5 次，默认配置对粒度 16。长度 16 和 580 按调度规则不该多停，作为对照。
- Q2：先做缩放实验，只改一步处理的 token 数，量出矩阵乘的平台和拐点（`q2_scaling.py`，权重换 fp32 作对照）。再量 prefill（512、2,000、8,000 个 token）和同时 1、16、64 个请求的 decode。手算运算量和字节数，配时间线里各类 kernel 的时间。
- Q2b：先发一遍把 prompt 写进前缀缓存，再原样重发，时间线上只剩 decode 步。上下文 16 到 2,305 个 token，同时 1、16、64 个请求，每格 3 次。
- Q3：两卡张量并行，关掉 vLLM 自带的 all-reduce（`--disable-custom-all-reduce`），通信全走 NCCL。时间线按进程分开，把名字带 `nccl` 的 kernel 的时间除以这张卡上全部 kernel 的时间。
- 每个子问题的预测和判定标准都在开跑前写定。

## 结果

| 子问题 | 主要数字 | 预测 |
|---|---|---|
| Q1 | prompt 560 个 token 时多停的一步让请求多用 45.2 毫秒，其中 GPU 在跑 15.1 毫秒，空着等 CPU 30.1 毫秒。9B 上 GPU 在跑 26.9 毫秒，总量 42.2 毫秒。多停的每一档多 658 个 kernel，不多停的档多 0 个 | 步数预测成立。「多停一次慢 25–45 毫秒」只在 560 成立：2,000 时 23.7 毫秒，8,000 时 11.3 毫秒。全注意力的对照无法判定，vLLM 不接受对它设这个参数 |
| Q2 | 平台 6.21 毫秒一遍（691 GB/s），拐点每步 217 个 token。单请求 decode 一步 12.2–12.3 毫秒，读权重达到平台的 89%–93%。fp32 权重的平台高 1.90 倍 | 平台和拐点、fp32 两条成立。「并发到 64 每步不超过 1.5 倍」在混合模型上不成立（2.1 倍），手算漏了状态读写 |
| Q2b | 同时 64 个请求：上下文 16 时全注意力每秒 3,793 个 token、混合 2,348 个。上下文 2,305 时全注意力 1,372 个、混合 1,899 个。从 577 到 2,305，全注意力每步多 22.5 毫秒，混合多 5.1 毫秒 | 5 条全部成立。时钟一条也成立：缩放实验中每步 1,024 个 token 以上时 SM 时钟平均 1,441.3 MHz，16 个以下时 2,490 MHz |
| Q3 | 单请求 decode 时通信占 GPU 时间约 10%，两卡每步是单卡的 0.61 和 0.64 倍。prefill 时通信占 47%–54%。同时 64 个请求时占 29% 和 38%，两卡是单卡的 0.72 和 0.86 倍 | 两条都不成立：通信在 decode 里占得比 prefill 少，两卡比单卡快得比预计多 |

读法：

1. 多出的那一步是一次完整的前向，时间大半花在 GPU 等 CPU 发起下一个 kernel。所以模型从 4B 换到 9B，这一步只从 34.5 变到 38.2 毫秒（E3）。
2. 逐个生成 token 受访存限制，读长 prompt 受算力限制，分界在一步约 217 个 token。
3. 混合模型每个请求每步要读写约 103 MB 的状态，和上下文长度无关。全注意力要读的 KV 随上下文变长，每个 token 147 KB。所以高并发 decode 下两者的交叉点在约 1,000 个 token 的上下文。

## 局限

- L40 上没有硬件计数器，缓存命中率和 kernel 内部的等待看不到。「读权重的速率」按每个权重读一遍计算，这个假设只在 RTX 5060 上用计数器核对过。
- 拐点和算力上限在 300 W 下量得，只对这张卡和这一种矩阵形状成立。持续算力受限的负载会降频，短时间量到的峰值不能当分母。
- 上下文只测到 2,305 个 token。一个 vLLM 版本，Triton 后端。FlashInfer 后端没有多停的那一步，没有测。
- 两卡量的是 NCCL 这条路，不是 vLLM 两卡时默认的自带 all-reduce。连接方式只有一种（PCIe、跨 NUMA）。NCCL kernel 在 kernel 里等对方，所以通信占比是上限。
- Qwen3-4B 两卡带 profiler 的那一遍在最后一组出 CUDA 错误，同时 64 个请求那一格只有 2 次重复。原因后来在 E9 查明，是旧版 Nsight Systems 追踪 CUDA graph 节点时引起的（见 `../e9_two_gpu_debug/`）。
- 逐请求记录和时间线没有公开，需要可以索取。

## 复现

在 GPU 服务器上，从仓库根目录运行。共同的环境变量：`HP_ROOT=$HP_ROOT HP_ENV=<venv> GATE_GPU_UUID=<gpu-uuid> HP_QWEN35_4B_DIR=<dir> HP_QWEN35_9B_DIR=<dir> HP_QWEN3_DIR=<dir> PROBE_NSYS=<nsys>`。

```bash
# Q1
PROBE_CONFIGS="h4_C0 h4_C1 fa_F0 fa_F16 h9_C0 h9_C1" PROBE_LENGTHS="16 560 580 2000 8000" PROBE_REPS=5 \
  scripts/run_probe.sh $RUN
python experiments/e4_profiling/summarize_q1.py $RUN/probe --json q1_summary.json
python experiments/e4_profiling/posthoc_q1.py $RUN/probe --json q1_posthoc.json
# Q2
PROBE_CONFIGS="h4_C0 fa_F0" PROBE_LENGTHS="512 2000 8000" PROBE_REPS=3 PROBE_OUT_LEN=1 PROBE_NVML=1 PROBE_SCALING=1 \
  PROBE_CLIENT_ARGS="--pattern groups --batches 1 16 64 --decode-tokens 128" \
  PROBE_ANALYZE=experiments/e4_profiling/analyze_q2.py scripts/run_probe.sh $RUN
python experiments/e4_profiling/summarize_q2.py $RUN/probe --json q2_summary.json
# Q2b
PROBE_CONFIGS="h4_C0 fa_F0" PROBE_LENGTHS=8000 PROBE_REPS=3 PROBE_OUT_LEN=1 PROBE_NVML=1 PROBE_SCALING=1 \
  PROBE_SCALING_ARGS="--dtypes bf16" PROBE_ANALYZE=experiments/e4_profiling/analyze_q2.py \
  PROBE_CLIENT_ARGS="--pattern long --contexts 16 577 1153 2305 --batches 1 16 64 --decode-tokens 128" \
  scripts/run_probe.sh $RUN
python experiments/e4_profiling/summarize_q2b.py $RUN/probe --json q2b_summary.json
# Q3 (two GPUs; use Nsight Systems 2026.6.1.251 or later, see E9)
GATE_GPU2_UUID=<gpu-uuid-2> PROBE_CONFIGS="h4_C0 fa_F0 h4_T2n fa_T2n" PROBE_LENGTHS="512 2000 8000" PROBE_REPS=3 \
  PROBE_OUT_LEN=1 PROBE_NVML=1 PROBE_GATE_ANY_USER=1 PROBE_STARTUP_TRIES=36 PROBE_STOP_ON_FAIL=1 \
  PROBE_CLIENT_ARGS="--pattern groups --batches 1 16 64 --decode-tokens 128" \
  PROBE_ANALYZE=experiments/e4_profiling/analyze_q3.py scripts/run_probe.sh $RUN
python experiments/e4_profiling/summarize_q3.py $RUN/probe --json q3_summary.json
# number table and figures from the summaries in results/
R=experiments/e4_profiling/results
python experiments/e4_profiling/make_report.py --q1 $R/q1_posthoc.json --q2 $R/q2_summary.json \
  --q2b $R/q2b_summary.json --q3 $R/q3_summary.json --ncu $R/ncu_medians.json --out-dir experiments/e4_profiling
```

单个 kernel 的计数器（Windows，管理员 PowerShell）：`tools/run_ncu_local.ps1`，再对每个报告 `ncu --import <case>.ncu-repz --csv --page raw > <case>.raw.csv`，最后 `python tools/ncu_extract.py decode.raw.csv prefill.raw.csv attention.raw.csv --medians ncu_medians.json`。

## 文件

- `probe_client.py`：三种请求模式（`cold` 冷请求，`groups` prefill 加并发 decode，`long` 长上下文 decode）。
- `analyze_q1.py`：Q1 时间线按请求和调度步拆开。`analyze_q2.py`：Q2、Q2b 按组合计。`analyze_q3.py`：Q3 按组、按进程合计，单列 NCCL。
- `summarize_q1.py`、`summarize_q2.py`、`summarize_q2b.py`、`summarize_q3.py`：各子问题的汇总和判定。
- `posthoc_q1.py`：按整个请求合计（看到结果后写的，不是判定；图 1 和 Q1 的数来自它）。
- `q2_scaling.py`：缩放实验。`nvml_sampler.py`：不带 profiler 时的 NVML 采样。
- `make_report.py`：生成 `e4_numbers.json` 和 `figures/`。
- `tools/`：笔记本上的 Nsight Compute 采集（`run_ncu_local.ps1`、`ncu_probe.py`）和提取（`ncu_extract.py`）。
- `results/`：四次采集的汇总文件和 Nsight Compute 的中位数。
- 启动脚本是仓库的 `scripts/run_probe.sh`。

---

## Question

E2 found that `--prefix-match-unit 16` makes the time to first token of Qwen3.5 about 30 ms longer, because prefill stops one extra scheduler step before the end of the prompt. E4 answers four sub-questions.

- Q1: where the time of that extra step goes.
- Q2: whether decode and long-prompt prefill are limited by memory bandwidth or by arithmetic.
- Q2b: how decode of the two designs changes with context length.
- Q3: how much of the GPU time goes to communication with tensor parallelism over two GPUs.

## Setup

- Hardware and software: one L40 (48 GB). Power limit 300 W for Q1 to Q2b, 250 W for Q3 (the one-GPU baseline was measured again in the same batch). vLLM main, commit `1479deb0`, Triton GDN backend. Tools: Nsight Systems 2025.1.3 (CUDA and NVTX tracing, `--cuda-graph-trace=node` to expand CUDA graphs) and NVML. Normal users cannot read hardware counters on the server, so per-kernel counters were taken on an RTX 5060 laptop with Nsight Compute 2026.4.0, only to check the direction of a conclusion.
- Models: Qwen3.5-4B and Qwen3.5-9B (hybrid: Gated DeltaNet plus full attention) against Qwen3-4B (full attention).
- Each config runs twice: once without a profiler for request timings, once under Nsight Systems. The trace is cut into requests at the pauses between them and into scheduler steps at the NVTX marker vLLM emits every step. Latency comparisons use only the run without a profiler.
- Q1: cold requests (no cache hit), prompts of 16, 560, 580, 2,000 and 8,000 tokens, 5 each, default against unit 16. Lengths 16 and 580 should not take an extra step under the scheduler rule and serve as controls.
- Q2: a scaling sweep that changes only the tokens per step measures the plateau and the knee of the matrix multiply (`q2_scaling.py`, with fp32 weights as a control). Then prefill (512, 2,000, 8,000 tokens) and decode with 1, 16 and 64 concurrent requests. Operations and bytes counted by hand, combined with the kernel time per class from the trace.
- Q2b: each group is sent once to fill the prefix cache and then again, so the trace holds only decode steps. Contexts of 16 to 2,305 tokens, 1, 16 and 64 concurrent requests, 3 repetitions per cell.
- Q3: tensor parallelism over two GPUs with vLLM's own all-reduce switched off (`--disable-custom-all-reduce`), so all communication goes through NCCL. The trace is split per process, and the time of kernels named `nccl` is divided by the time of all kernels on that GPU.
- Predictions and criteria for every sub-question were written down before the runs.

## Results

| Sub-question | Main numbers | Predictions |
|---|---|---|
| Q1 | At a 560-token prompt the extra step makes the request 45.2 ms longer: 15.1 ms with the GPU busy, 30.1 ms with the GPU idle and waiting for the CPU. On the 9B model the GPU is busy for 26.9 ms, 42.2 ms in total. Every length with an extra step has 658 more kernels, the others 0 more | Step counts held. "One extra step costs 25-45 ms" held only at 560: 23.7 ms at 2,000 and 11.3 ms at 8,000. The full-attention control could not be judged, because vLLM rejects the flag for that model |
| Q2 | Plateau 6.21 ms per pass (691 GB/s), knee at 217 tokens per step. Decode of one request takes 12.2-12.3 ms per step and reads the weights at 89%-93% of the plateau. The fp32 plateau is 1.90 x higher | Plateau and knee, and fp32, held. "Batch 64 costs at most 1.5 x per step" failed on the hybrid model (2.1 x): the hand estimate left out the state reads and writes |
| Q2b | 64 concurrent requests: at context 16, full attention 3,793 and hybrid 2,348 tokens/s. At context 2,305, full attention 1,372 and hybrid 1,899 tokens/s. From 577 to 2,305 tokens, the step grows by 22.5 ms for full attention and 5.1 ms for the hybrid | All 5 held. The clock prediction also held: in the scaling sweep the SM clock averages 1,441.3 MHz at 1,024 tokens per step and above, against 2,490 MHz at 16 and below |
| Q3 | Decode of one request: communication takes about 10% of GPU time, and a step on two GPUs takes 0.61 and 0.64 x the one-GPU time. Prefill: 47%-54%. With 64 requests: 29% and 38%, and 0.72 and 0.86 x | Both failed: communication takes less of decode than of prefill, and two GPUs speed up decode more than predicted |

How to read it:

1. The extra step is a full forward pass, and most of its time the GPU waits for the CPU to launch the next kernel. That is why it grows only from 34.5 to 38.2 ms from the 4B to the 9B model (E3).
2. Generating one token at a time is memory bound and reading a long prompt is compute bound. The boundary is about 217 tokens per step.
3. The hybrid model reads and writes about 103 MB of state per request per step, independent of the context. Full attention reads a KV cache that grows by 147 KB per token. So under high-concurrency decode the two cross at a context of about 1,000 tokens.

## Limitations

- No hardware counters on the L40, so cache hit rates and waits inside a kernel are not visible. "Weight read rate" assumes every weight is read once per pass, which was checked with counters only on the RTX 5060.
- The knee and the compute ceiling were measured at 300 W and hold only for this GPU and this matrix shape. Sustained compute-bound load lowers the clock, so a short peak measurement is not the right denominator.
- Contexts up to 2,305 tokens. One vLLM version, the Triton backend. The FlashInfer backend has no extra step and was not measured.
- The two-GPU numbers are for the NCCL path, not for vLLM's default own all-reduce. One interconnect (PCIe across NUMA nodes). An NCCL kernel waits for its peer inside the kernel, so the communication share is an upper bound.
- The two-GPU Qwen3-4B run under the profiler hit a CUDA fault in its last group, so the 64-request cell has 2 repetitions. E9 later traced the fault to the old Nsight Systems version tracing CUDA graph nodes (see `../e9_two_gpu_debug/`).
- Per-request records and traces are not published. They are available on request.

## Reproduce

On the GPU server, from the repository root, with the environment variables listed in the Chinese section and the commands in the code block above. The per-kernel counters on Windows: `tools/run_ncu_local.ps1` in an administrator PowerShell, then `ncu --import <case>.ncu-repz --csv --page raw > <case>.raw.csv` per report, then `tools/ncu_extract.py ... --medians ncu_medians.json`.

## Files

- `probe_client.py`: three request patterns (`cold`, `groups` for prefill plus concurrent decode, `long` for decode at long context).
- `analyze_q1.py`: the Q1 trace per request and scheduler step. `analyze_q2.py`: Q2 and Q2b totals per group. `analyze_q3.py`: Q3 totals per group and process, NCCL separately.
- `summarize_q1.py`, `summarize_q2.py`, `summarize_q2b.py`, `summarize_q3.py`: summaries and verdicts.
- `posthoc_q1.py`: totals over whole requests (written after the results, not a verdict; figure 1 and the Q1 numbers come from it).
- `q2_scaling.py`: the scaling sweep. `nvml_sampler.py`: NVML samples during the runs without a profiler.
- `make_report.py`: writes `e4_numbers.json` and `figures/`.
- `tools/`: Nsight Compute on a laptop (`run_ncu_local.ps1`, `ncu_probe.py`) and extraction (`ncu_extract.py`).
- `results/`: the summary files of the four collections and the Nsight Compute medians.
- The launcher is `scripts/run_probe.sh` at the repository root.
