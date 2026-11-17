# refs：对话模板原文 / chat templates

两份官方对话模板的原文，是实验的输入：`scripts/run_replay.sh` 的 F1 组用 Qwen3.5 的模板启动 Qwen3-4B，测试用它们检查渲染。许可 Apache-2.0。

| 文件 | 来源 | 版本 |
|---|---|---|
| `qwen3.5-4b_chat_template.jinja` | `Qwen/Qwen3.5-4B` 仓库的 `chat_template.jinja` | revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `qwen3-4b_chat_template.jinja` | `Qwen/Qwen3-4B` 仓库 `tokenizer_config.json` 里的 `chat_template` 字段 | revision `1cfa9a7208912126459214e8b04321603b3df60c` |

实验所用的 Qwen3.5-4B 模型目录里的 `chat_template.jinja` 和这里的文件逐字节相同（sha1 `f54c154d3de0a2c6c71a39ea77d8d819b74e54e3`，7,756 字节）。Qwen3.5-9B 的模板和 4B 的也逐字节相同。

---

The two official chat templates, verbatim. They are inputs: config F1 of `scripts/run_replay.sh` starts Qwen3-4B with the Qwen3.5 template, and the tests render with them. License Apache-2.0. Sources and revisions are in the table above. The `chat_template.jinja` in the Qwen3.5-4B model directory used for the runs is byte-identical to `qwen3.5-4b_chat_template.jinja` (sha1 `f54c154d3de0a2c6c71a39ea77d8d819b74e54e3`, 7,756 bytes); so is the Qwen3.5-9B template.
