# E9：两卡张量并行时的 CUDA 故障排查 / E9: Debugging a CUDA fault with two-GPU tensor parallelism

## 问题

E4 的 Q3 在两张 L40 上做张量并行时出过两次故障。第一次用 vLLM 默认配置（自带的 all-reduce），服务没有起来，内核日志报 Xid 62，第二张卡要重启机器才恢复。第二次只走 NCCL（`--disable-custom-all-reduce`），8 遍里前 7 遍正常，最后一遍带 Nsight Systems 的采集里，一个 worker 报 `unspecified launch failure`，内核日志是 Xid 13。两张卡各接一个 CPU（PCIe，跨 NUMA 节点），IOMMU 开着。E9 要回答：故障是不是这种布局造成的，如果不是，原因在哪。

## 做法

一共 10 步，每一步先写下预测和读法，结果出来以后决定下一步。模型都是 Qwen3-4B。vLLM 里的一组是「1 个 512 个 token 的 prefill，再同时 64 个请求各生成 128 个 token」，一遍 30 组。第 1 到 4 步和另一个用户的训练任务共用两张卡，只看出不出错，不读耗时。第 5 到 10 步卡上没有别的任务。每一步都带一个已知会失败的对照：第 1、7 步故意发错数据，看检查能不能发现，之后各步保留原样的出错设置，它必须照旧出错，这一步的「不出错」才算数。

| 步 | 做了什么 | 结果 |
|---|---|---|
| 1 | 不经过 vLLM，两个进程反复做 all-reduce 和 all-gather，逐元素校验。四组：故意发错（对照）、关 P2P、不设、强制 P2P（`NCCL_P2P_LEVEL=SYS`），每组 240 秒 | 对照查出了发错的 2 个元素。三组各约 27,000 次通信、约 580 GB 数据，0 个校验错。NCCL 默认不在这两张卡之间用 P2P，走主机共享内存（`SHM/direct`），只有强制时才走 `P2P/CUMEM` |
| 2 | vLLM 里重复 E4 出错的那一档：不带 profiler 一遍，带 Nsight Systems 一遍 | 不带 profiler 30 个 decode 组正常。带 profiler 的在第 14 个 decode 组出错 |
| 3 | 两张卡对调顺序再跑 | 不带 profiler 正常，带 profiler 在第 8 个 decode 组出错 |
| 4 | 单卡带 profiler，两张卡各一遍 | 各 30 组正常 |
| 5 | 把带 profiler 的设置拆成四样分别试：只设两个环境变量、只追踪 CUDA 调用、不展开 CUDA graph、带着展开 graph 的选项启动但不采集 | 四种各 30 个 decode 组正常。原样的设置在第 3 个 decode 组出错。触发条件是「追踪 CUDA graph 的节点」加「正在采集」 |
| 6 | 用 `taskset` 把整个服务绑在 NUMA 节点 0、节点 1，或者不绑 | 三遍都出错（第 5、8、15 个 decode 组）。不绑的那一遍报错的是第一张卡上的 worker，所以故障不跟着某一张卡 |
| 7 | 写一段不含 vLLM 的复现程序：两个进程，CUDA graph 里有 NCCL 的 all-reduce，每遍 7,680 步。第一版每一步整卡同步 | 不带 profiler、不展开 graph、原样设置都正常，没有复现 |
| 7b | 照 vLLM 的异步输出加上「另一个线程在第二个 CUDA stream 上等事件」 | 原样设置下两个进程都报 `unspecified launch failure`，不带 profiler 和不展开 graph 都正常。复现成功 |
| 7c | 原样设置再加 `--cuda-event-trace=false` | 复现程序上两遍正常，原样设置两遍停住 |
| 8 | 回到 vLLM 验证 `--cuda-event-trace=false` | 仍在第 3 个 decode 组出错。这个开关对 vLLM 不管用 |
| 9 | 复现程序上换新版 Nsight Systems 2026.6.1.251，和旧版交替跑 | 新版两遍各 7,680 步正常，旧版照旧出错 |
| 10 | vLLM 里换新版 | 新版两遍各 30 个 decode 组正常，旧版在第 13 个 decode 组出错 |

## 结果

故障来自旧版 Nsight Systems（2025.1.3，CUDA 12.9 工具包自带）在两卡运行时追踪 CUDA graph 节点并正在采集。换成 2026.6.1.251 以后，同样的选项和负载不再出错。按旧版在 vLLM 里 74 个 decode 组出错 9 次的比例，新版 60 组都不出错的偶然概率不到千分之一。

| 条件 | 旧版 2025.1.3 | 新版 2026.6.1.251 |
|---|---|---|
| vLLM 两卡，展开 graph 加采集 | 9 遍都出错 | 2 遍各 30 个 decode 组，0 次出错 |
| 复现程序（异步输出），展开 graph 加采集 | 4 遍都出错 | 2 遍各 7,680 步，0 次出错 |
| vLLM 两卡，不带 profiler | 63 个 decode 组，0 次出错 | 没有测 |
| vLLM 单卡，展开 graph 加采集 | 两张卡各 30 组，0 次出错 | 没有测 |

排除掉的解释：P2P（故障时根本不走 P2P，强制 P2P 的纯通信测试也没有出错）；卡槽和 CPU 的布局（绑在哪个 NUMA 节点都出错）；某一张卡（两张卡上都出过错，单卡不出错）；vLLM 的代码（不含 vLLM 的复现程序也出错）。所以「两张卡在不同卡槽、各接一个 CPU」不是原因，这种布局唯一确定的后果是 NCCL 默认不用 P2P。

在这台机器上采两卡时间线的做法：用新版 Nsight Systems。只有旧版时去掉 `--cuda-graph-trace=node`，代价是看不到 graph 里面的 kernel。

## 局限

- 旧版里的机制没有查清。只知道触发条件，不知道是哪一层出的错，也没有在 NVIDIA 的发布说明里找到对应条目。
- 第一次故障（vLLM 自带 all-reduce，没有带 profiler，Xid 62）不在这个解释里。它的条件不同，之后没有再跑。
- 各次出错的内核日志普通账号读不到，没有取。
- 只测了 Qwen3-4B 一个模型、一台机器、一个驱动版本。
- 结论没有报告给上游。逐组记录和服务日志没有公开，需要可以索取。

## 复现

在有两张 GPU 的服务器上，从仓库根目录运行。共同的环境变量：`HP_ROOT`、`HP_ENV`（装了 vLLM 的 venv）。

```bash
# step 1: communication test
E9_GPU_A=<gpu-uuid-0> E9_GPU_B=<gpu-uuid-1> experiments/e9_two_gpu_debug/run_comm_test.sh $RUN
python experiments/e9_two_gpu_debug/summarize.py comm $RUN --json step1_comm_summary.json --redact-uuids
# steps 2-6, 8, 10: vLLM with the profiling launcher; the modes are documented in scripts/run_probe.sh
GATE_GPU_UUID=<gpu-uuid-0> GATE_GPU2_UUID=<gpu-uuid-1> HP_QWEN3_DIR=<dir> HP_QWEN35_4B_DIR=<dir> HP_QWEN35_9B_DIR=<dir> \
  PROBE_CONFIGS=fa_T2n PROBE_MODES="envonly nsyscuda nsysnograph nsysidle nsys" PROBE_LENGTHS=512 PROBE_REPS=30 \
  PROBE_OUT_LEN=1 PROBE_STARTUP_TRIES=36 PROBE_STOP_ON_FAIL=0 PROBE_SERVE_EXTRA="--gpu-memory-utilization 0.6" \
  PROBE_CLIENT_ARGS="--pattern groups --batches 64 --decode-tokens 128" \
  PROBE_ANALYZE=experiments/e4_profiling/analyze_q3.py PROBE_NSYS=<nsys-2025.1.3> PROBE_NSYS_NEW=<nsys-2026.6.1> \
  scripts/run_probe.sh $RUN
python experiments/e9_two_gpu_debug/summarize.py groups $RUN/probe/fa_T2n_*.jsonl
# steps 7-9: the reproducer
E9_GPU_A=<gpu-uuid-0> E9_GPU_B=<gpu-uuid-1> E9_NSYS=<nsys-2025.1.3> E9_NSYS_NEW=<nsys-2026.6.1> \
  E9_MODES="control plain nsysnograph nsys nsysnoevent nsysnew" E9_REPRO_ARGS="--async-output 1" \
  experiments/e9_two_gpu_debug/run_graph_repro.sh $RUN
python experiments/e9_two_gpu_debug/summarize.py repro $RUN/summary.json
python experiments/e9_two_gpu_debug/trace_counts.py $RUN/nsys/trace.sqlite   # after nsys export --type sqlite
```

各步实际用的模式：第 2、3 步 `plain nsys`（第 3 步配置 `fa_T2ns`），第 4 步 `plain nsys`（配置 `fa_F0`，各一张卡），第 5 步见上，第 6 步 `nsysc1 nsysc0 nsys`，第 8 步 `nsysnoevent nsys`，第 10 步 `nsysnew nsys`。复现程序第 7 步 `control plain nsysnograph nsys`（不带 `--async-output`），7b 同样四种带 `--async-output 1`，7c `nsysnoevent@1 nsys@1 nsysnoevent@2 nsys@2`，第 9 步 `nsysnew@1 nsys@1 nsysnew@2`。

## 文件

- `comm_test.py`：两卡通信测试，逐元素校验。`run_comm_test.sh`：按顺序跑四组。
- `graph_repro.py`：不含 vLLM 的复现程序。`run_graph_repro.sh`：按模式依次运行它，带或不带 profiler。
- `summarize.py`：三种表（`comm` 通信测试，`repro` 复现程序，`groups` vLLM 每一遍的组数和没完成的组）。
- `trace_counts.py`：数一份时间线里每个进程的 kernel、NCCL kernel 和带 graph 节点号的 kernel。
- `results/step1_comm_summary.json`：第 1 步四组的汇总（GPU UUID 已替换）。
- `results/step7_repro_sync.json`、`step7b_repro_async.json`、`step7c_repro_event_trace_off.json`、`step9_repro_nsys_versions.json`：复现程序各次运行的汇总。
- `results/vllm_runs.txt`：vLLM 各遍的组数和出错的组（`summarize.py groups` 的输出）。

---

## Question

Q3 of E4 ran tensor parallelism over two L40 GPUs and failed twice. The first time, with vLLM's default (its own all-reduce kernel), the server did not come up, the kernel log reported Xid 62, and the second GPU needed a reboot. The second time, with NCCL only (`--disable-custom-all-reduce`), 7 of 8 passes were fine, and in the last pass, under a Nsight Systems collection, a worker reported `unspecified launch failure` (Xid 13 in the kernel log). Each GPU hangs off a different CPU (PCIe across NUMA nodes) and the IOMMU is on. E9 asks whether this layout causes the fault, and if not, what does.

## Setup

Ten steps. Before each one the prediction and its reading were written down, and the result decided the next step. All runs use Qwen3-4B. In vLLM a group is one 512-token prefill followed by 64 concurrent requests that generate 128 tokens each, 30 groups per pass. Steps 1 to 4 shared the GPUs with another user's training job and were judged only on whether they failed, never on timing. Steps 5 to 10 had the GPUs to themselves. Every step had a known-bad control: steps 1 and 7 sent wrong data on purpose and the checks had to find it, and the later steps kept the original failing setting, which had to fail again for a "no fault" in the same step to count.

| Step | What | Result |
|---|---|---|
| 1 | No vLLM: two processes repeat all-reduce and all-gather and check every element. Four groups: wrong data on purpose (control), P2P off, nothing set, P2P forced (`NCCL_P2P_LEVEL=SYS`), 240 s each | The control found both wrong elements. The three groups ran about 27,000 collectives and about 580 GB each with 0 wrong elements. NCCL does not use P2P between these GPUs by default and goes through host shared memory (`SHM/direct`); only when forced does it use `P2P/CUMEM` |
| 2 | The failing case of E4 in vLLM, once without a profiler and once under Nsight Systems | Without a profiler 30 decode groups were fine. Under the profiler the 14th decode group failed |
| 3 | The same with the GPUs in the other order | Fine without a profiler, failed in the 8th decode group with it |
| 4 | One GPU under the profiler, each GPU once | 30 groups each, all fine |
| 5 | The profiler setting taken apart: only the two environment variables, only CUDA API tracing, no graph-node tracing, graph-node tracing option but no collection | Each 30 decode groups fine. The original setting failed in the 3rd decode group. Trigger: tracing CUDA graph nodes while collecting |
| 6 | The whole server bound with `taskset` to NUMA node 0, node 1, or not bound | All three failed (decode groups 5, 8, 15). In the unbound pass the worker on the first GPU reported the error, so the fault does not follow one GPU |
| 7 | A reproducer without vLLM: two processes, NCCL all-reduce inside a CUDA graph, 7,680 steps per pass. First version: device-wide synchronize every step | No fault in any mode |
| 7b | Add vLLM's asynchronous output: a second thread waits for an event on a second CUDA stream | Under the original setting both processes reported `unspecified launch failure`; without a profiler and without graph-node tracing it was fine. Reproduced |
| 7c | The original setting plus `--cuda-event-trace=false` | Two passes fine in the reproducer, the original setting hung twice |
| 8 | `--cuda-event-trace=false` back in vLLM | Still failed in the 3rd decode group. The switch does not help vLLM |
| 9 | Nsight Systems 2026.6.1.251 in the reproducer, alternating with the old version | New version: two passes of 7,680 steps fine. Old version failed again |
| 10 | The new version in vLLM | New version: two passes of 30 decode groups fine. Old version failed in the 13th decode group |

## Results

The fault comes from the old Nsight Systems (2025.1.3, shipped with the CUDA 12.9 toolkit) tracing CUDA graph nodes during a collection on two GPUs. With 2026.6.1.251 the same options and load do not fail. Given 9 failures in 74 decode groups with the old version in vLLM, 60 fault-free groups with the new version by chance have a probability below one in a thousand.

| Condition | Old 2025.1.3 | New 2026.6.1.251 |
|---|---|---|
| vLLM on two GPUs, graph-node tracing while collecting | all 9 passes failed | 2 passes of 30 decode groups, 0 failures |
| Reproducer (asynchronous output), graph-node tracing while collecting | all 4 passes failed | 2 passes of 7,680 steps, 0 failures |
| vLLM on two GPUs, no profiler | 63 decode groups, 0 failures | not measured |
| vLLM on one GPU, graph-node tracing while collecting | 30 groups on each GPU, 0 failures | not measured |

Ruled out: P2P (the failing runs do not use P2P at all, and the forced-P2P communication test did not fail); the slot and CPU layout (failures with the server on either NUMA node); one particular GPU (failures on both, none on one GPU alone); vLLM's code (the reproducer without vLLM fails too). So "the two GPUs sit in different slots on different CPUs" is not the cause; the only definite effect of that layout is that NCCL does not use P2P by default.

How to collect two-GPU traces on this machine: use the new Nsight Systems. With only the old version, drop `--cuda-graph-trace=node`, at the cost of not seeing the kernels inside CUDA graphs.

## Limitations

- The mechanism inside the old version is unknown. The trigger is known, not the layer that fails, and no matching entry was found in NVIDIA's release notes.
- The first fault (vLLM's own all-reduce, no profiler, Xid 62) is not explained by this. Its conditions differ and it was not run again.
- The kernel logs of the failures were not readable for a normal user and were not collected.
- One model (Qwen3-4B), one machine, one driver version.
- Not reported upstream. Per-group records and server logs are not published; they are available on request.

## Reproduce

See the code block and the list of modes per step in the Chinese section.

## Files

- `comm_test.py`: the two-GPU communication test with element-wise checks. `run_comm_test.sh`: runs the four groups in order.
- `graph_repro.py`: the reproducer without vLLM. `run_graph_repro.sh`: runs it in several modes, with or without the profiler.
- `summarize.py`: three tables (`comm`, `repro`, and `groups` for the group counts and incomplete groups of a vLLM pass).
- `trace_counts.py`: counts kernels, NCCL kernels and kernels with a graph node id per process in a trace.
- `results/step1_comm_summary.json`: step 1, four groups (GPU UUIDs replaced).
- `results/step7_repro_sync.json`, `step7b_repro_async.json`, `step7c_repro_event_trace_off.json`, `step9_repro_nsys_versions.json`: reproducer runs.
- `results/vllm_runs.txt`: group counts and failing groups of the vLLM passes (output of `summarize.py groups`).
