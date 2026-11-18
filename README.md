# hybrid-prefix-lab

**中文** | [English](README.en.md)

Qwen3.5 这类混合注意力模型（Gated DeltaNet 线性注意力加少量全注意力层）在 [vLLM](https://github.com/vllm-project/vllm) 上做前缀缓存，会比普通的全注意力模型少命中多少？vLLM 现有的启动参数能补回多少？补回来以后，首个 token 的等待是变短还是变长，时间花在了哪里？

这个仓库是回答这些问题的测量代码和结果汇总。负载是真实的多轮对话（WildChat）和两套编码 Agent 的轨迹（SWE-smith、SWE-rebench），对照模型是同规模的全注意力模型 Qwen3-4B。测量报告已经发在 vLLM 的 [issue #67674](https://github.com/vllm-project/vllm/issues/67674)。

## 结论

单张 NVIDIA L40，vLLM main（commit `1479deb0`），Qwen3.5-4B 对 Qwen3-4B，请求逐会话发送。E1 一共回放了 114,681 个请求，0 报错。

- **默认配置下差距不小。** 混合模型每轮要重算的 token 是全注意力模型的 1.6 到 1.7 倍，多轮对话的命中率 56% 对 71%。原因是混合模型的缓存块是 576 个 token，全注意力是 64 个。
- **现有参数能把命中率拉平，但首个 token 反而变慢。** 设 `--prefix-match-unit 16` 以后，命中率差不到 1 个百分点，可是约 96% 的请求首个 token 慢约 30 毫秒，比少重算省下的时间还多。
- **这 30 毫秒是调度器多停的一步。** L40 上 Qwen3.5 的线性注意力层走 Triton 后端，它不能在一次前向里顺带保存状态，调度器只好在 prompt 末尾多切一刀。Nsight Systems 显示这一步多做了一次完整的前向（多 658 个 kernel），请求多出的 45.2 毫秒里 GPU 只忙了 15.1 毫秒，其余在等 CPU 发起下一个 kernel。
- **模型越大越划算。** 这一步在 9B 上是 38.2 毫秒，几乎不变，在 27B（FP8，64 层）上涨到 59.8 毫秒，但每个 token 的计算更贵，省下的更多：27B 上设这个参数以后，之后的请求快 97 到 241 毫秒。
- **社区的修法有用，要配对参数。** PR #65186（已合并）让 Triton 后端在 64 个 token 的格点上顺带保存状态。在这个 PR 的分支上（基点已含 #66938，合并进主线的就是这一版）配 `--prefix-match-unit 64`，Agent 负载上的请求不再变慢，对话负载仍慢 8 到 9 毫秒。另一个 PR #61952 在服务的默认并发上限下，让 prefill 一步不超过 512 个 token 的请求慢 67 到 275 毫秒，作者在 issue 里确认了这是缺陷。
- **缓存吃紧时混合模型丢命中是整段地丢。** 缓存池贴着峰值占用时，全注意力模型保住 95% 到 100% 的命中，混合模型只保住 0% 到 66%。混合模型要给到峰值占用的 1.5 到 2 倍才不丢。
- **整段丢的原因是一个实现上的缺陷，上游的修法有效。** 每个请求在 prompt 末尾存的那一份状态，在请求结束前就被放回空闲队列，请求结束时又排在自己的注意力块前面，于是最先被淘汰；它一丢，注意力块留着也命中不了。命中为 0 的请求里 68% 到 100% 只缺这一份状态。把上游的修法 PR #61282 移植过来以后，合成会话 1.5 倍池的保留率从 0.40–0.60 升到 0.997，真实 Agent 会话贴着峰值时从 0.275 升到 0.850。机制是 issue #61281 先写出来的，这里做的是在真实负载上逐请求定位并验证修法。

每个实验都先把预测和判定标准写下来再跑，判定性结论都带「已知合格」和「已知不合格」两类对照（例如同一配置跑两遍、人为加 30 毫秒看能不能量出来）。

## 实验一览

| 实验 | 问题 | 结果 |
|---|---|---|
| [E1](experiments/e1_prefix_hits) | 默认配置下混合模型少命中多少，现有参数能补回多少 | 重算量 1.6–1.7 倍；设 `--prefix-match-unit 16` 后命中率差不到 1 个百分点 |
| [E2](experiments/e2_ttft_cost) | 补回命中以后更快吗，慢在哪 | 约 96% 的请求慢约 30 毫秒；多停一次 +34.5 毫秒，不多停 +0.3 毫秒 |
| [E3](experiments/e3_9b) | 换成 9B 呢 | 那一步 38.2 毫秒，几乎不变；净效果缩小，仍然更慢 |
| [E4](experiments/e4_profiling) | 时间花在哪，计算受限还是访存受限，两卡通信占多少 | GPU 只忙三分之一；屋顶线拐点 217 个 token、平台 691 GB/s；两卡时通信占 GPU 时间约 10%（decode）到约一半（prefill） |
| [E5](experiments/e5_cache_pressure) | 并发高、缓存吃紧时混合模型有没有额外的问题 | 有，整段丢命中；要多给 1.5 到 2 倍的池 |
| [E6](experiments/e6_27b) | 27B 上那个参数还亏吗 | 不亏了，之后的请求快 97–241 毫秒 |
| [E7](experiments/e7_upstream_prs) | 社区的 PR #65186、#66881 有用吗 | #65186 配粒度 64 后 Agent 负载不再变慢；#66881 的按收益淘汰在缓存吃紧时保住更多命中 |
| [E8](experiments/e8_pr61952) | 应作者要求测 PR #61952 | 默认并发上限下慢 67–275 毫秒，作者确认是缺陷；`--max-num-seqs 8` 时不再变慢，多停的那一步从 29.7 降到 18.0 毫秒 |
| [E9](experiments/e9_two_gpu_debug) | 两卡采集时间线时出的 CUDA 错误是什么原因 | 旧版本 Nsight Systems（2025.1.3）追踪 CUDA graph 节点时造成，换 2026.6.1.251 后不再出现；不是 P2P，也不是两张卡分接两个 CPU |
| [E10](experiments/e10_state_eviction) | E5 里为什么整段丢命中，上游的修法有没有用 | 缺的是 prompt 末尾那份状态，它被提前放回、最先淘汰；PR #61282 移植后合成会话 1.5 倍池 0.40–0.60 → 0.997，真实会话贴着峰值 0.275 → 0.850 |

每个实验目录里的 README 是那个实验的结果汇总：做法、数字、哪些预测成立、局限、怎么复现。

## 目录

```
hplab/                  公共库
  client.py             和 vLLM 服务通信：流式请求、/metrics、清空前缀缓存
  prompts.py            精确长度的合成 prompt
  sessions.py           读三个数据集的多轮会话，用对话模板渲染成 prompt
  cache_model.py        按 vLLM 的规则算「应该命中多少」和「prefill 要分几步」
  stats.py              中位数、配对差的汇总、直线拟合
  nsys.py               读导出成 SQLite 的 Nsight Systems 时间线
  records.py            读四遍对比（C0a、C1a、C1b、C0b）的逐请求记录
scripts/
  serve_lib.sh          启动脚本共用的 shell 函数：GPU 闸门、按 PID 停进程
  run_replay.sh         多轮回放：每个配置起一次 vLLM 服务，回放所有会话（E1、E2、E3、E6、E7、E8）
  run_probe.sh          一次一个请求的探针和 profiling（E4、E7、E8、E9）
  setup/                建环境、下载模型和数据并按哈希校验
experiments/            每个实验一个目录，见上表
refs/                   官方对话模板（渲染 prompt 的输入）
tests/                  公共库、判定脚本和分析脚本的测试
```

## 怎么复现

需要一台带 NVIDIA GPU 的 Linux 服务器（这里用的是 L40，48 GB）和 vLLM。

```bash
pip install -e ".[replay,test]"
python -m pytest -q            # 不需要 GPU
```

1. 建环境、下载模型和数据集：见 [scripts/setup](scripts/setup)。被测的 vLLM 是 main 的 commit `1479deb0`，E7、E8 另有 PR 分支的环境。
2. 起服务并回放：`scripts/run_replay.sh` 和 `scripts/run_probe.sh` 的开头列出了全部环境变量；每个实验的 README 给出它用的那一组。
3. 判定和汇总：每个实验 README 的「复现」一节列出了读回放记录、给出判定和汇总表的命令。

逐请求的原始记录没有放进仓库，需要的话可以在 issue 里联系。

## 范围

- 一种显卡（L40）和 Triton 后端。按 vLLM 源码，H100 这类走 FlashInfer 后端的卡不需要多停那一步，没有实测。
- 模型以 4B 为主，另有 9B 和 27B（FP8）各一组四遍对比。
- E1 到 E4 的请求逐会话发送（E1 另有一档 8 个会话轮转），并发和缓存吃紧在 E5 单独测。
- 机器的功耗上限中途从 300 W 改成了 250 W。跨日期的比较都在同一批里重量了基线。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。`experiments/e7_upstream_prs/patches/` 里的补丁来自 vLLM 的 PR，许可同 vLLM（Apache 2.0），见 [NOTICE](NOTICE)。
