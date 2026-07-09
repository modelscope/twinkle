# Auto-Research

Twinkle Auto 是一个基于终端的智能训练助手，支持通过**自然语言控制、监控和调试 ML 训练**。它将聊天驱动的 AI 代理与自动化健康监控器相结合，能够自主检测并修复训练故障。

## 架构概览

```
┌──────────────────────────────────────────────────────────┐
│ TwinkleAuto (asyncio 聊天循环)                            │
│                                                          │
│ 核心组件:                                                 │
│   AgentLoop  ─── LLM 工具调用循环                         │
│   TrainingMonitor ─── 定期健康检查与自动修复               │
│   LocalConnection ─── 基于文件系统的通信层                 │
│   SkillManager ─── 异步插件加载                           │
└──────────────────────────────────────────────────────────┘
```

## 安装与启动

Auto 是 `twinkle-client` 包的一部分：

```bash
pip install twinkle-client
```

### 命令行用法

```bash
# 基本启动（使用默认本地 Ollama 端点）
twinkle-auto

# 指定 LLM 后端
twinkle-auto --llm-base-url http://localhost:11434/v1 --llm-model qwen3.5

# 连接到已有训练运行
twinkle-auto --run-id my-grpo-run

# 使用远程 API（如 OpenAI 兼容接口）
twinkle-auto --llm-base-url https://api.example.com/v1 --llm-api-key sk-xxx --llm-model gpt-4o

# 启用调试日志
twinkle-auto --verbose
```

也可作为 Python 模块运行：

```bash
python -m twinkle_client.auto
```

### CLI 参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| `--run-id`, `-r` | `TWINKLE_AUTO_RUN_ID` | None | 连接到已有训练运行 |
| `--llm-base-url` | `TWINKLE_LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API 基础 URL |
| `--llm-model` | `TWINKLE_LLM_MODEL` | `qwen3.5` | LLM 模型名称 |
| `--llm-api-key` | `TWINKLE_LLM_API_KEY` | `not-needed` | LLM API 密钥 |
| `--verbose`, `-v` | `TWINKLE_AUTO_VERBOSE` | `False` | 启用 DEBUG 日志 |
| `--version`, `-V` | — | — | 显示版本并退出 |

## 聊天代理

Auto 的核心是一个 **LLM 驱动的工具调用代理**（`AgentLoop`），通过 OpenAI 兼容 API 处理自然语言命令。代理维护对话历史并自动修剪（保留最近 50 条消息），每次交互最多支持 10 轮工具调用。

### 你可以这样说

**训练生命周期：**
- *"列出我的训练运行"*
- *"用 Qwen3.5-4B 在 gsm8k 上启动一个新的 GRPO 训练"*
- *"暂停当前运行"*
- *"恢复训练"*
- *"停止训练"*

**服务器管理：**
- *"启动服务器，使用 Qwen3.5-4B 和一个 2 卡的 Qwen3.5-72B 采样器"*
- *"关闭服务器"*
- *"有多少 GPU 可用？"*

**监控与分析：**
- *"训练进展如何？"*
- *"显示 reward 相关的指标"*
- *"放大到 step 100-200"*
- *"重置图表视图"*

**搜索：**
- *"搜索数学数据集"*
- *"在 ModelScope 上查找 Qwen 模型"*

### 可用工具

代理内置 13 个工具：

| 工具 | 说明 |
|------|------|
| `list_training_runs` | 列出所有训练运行 |
| `get_training_status` | 获取详细状态和最近指标 |
| `start_server` | 启动 Ray 集群 + Twinkle Server（幂等） |
| `shutdown_server` | 关闭服务器并释放 GPU 资源 |
| `start_training` | 创建并启动新的训练运行 |
| `select_run` | 切换监控到另一个运行 |
| `pause_training` | 暂停训练（SIGKILL，服务器保留状态） |
| `resume_training` | 通过重新启动客户端脚本恢复训练 |
| `stop_training` | 停止训练（SIGTERM，保存检查点） |
| `update_script` | 更新训练脚本（带版本归档） |
| `list_supported_models` | 查询服务器支持的模型 |
| `search_datasets` | 在 ModelScope 搜索数据集 |
| `search_models` | 在 ModelScope 搜索模型 |
| `zoom_metrics` | 调整指标图表视图范围 |
| `select_metrics` | 选择显示哪些指标（最多 4 个） |
| `get_cluster_info` | 获取 GPU/集群资源信息 |

### 服务器启动

`start_server` 工具自动化一个多步骤流程：

1. **GPU 检测** — `nvidia-smi` 硬件扫描
2. **GPU 分配** — 在训练模型和采样器之间分配 GPU
3. **配置生成** — 自动创建 `server_config.yaml`
4. **Ray 集群启动** — 多节点 GPU 分区，隔离 `CUDA_VISIBLE_DEVICES`
5. **服务器启动** — 作为后台进程启动 Twinkle Server
6. **健康检查** — 轮询 `/api/v1/healthz` 直到就绪

支持多模型拓扑：1 个训练模型 + N 个采样器/教师模型。

### Skills 系统

Auto 支持从三个来源加载可扩展的技能插件：

1. **内置技能** — 包含在 `twinkle_client/skills/bundled/` 中
2. **用户本地技能** — `~/.cache/twinkle/auto/skills/local/`
3. **社区技能** — 从 ModelScope 获取（尽力而为，10 秒超时）

技能在启动后异步加载并注入代理的系统提示词中。代理在技能加载完成前即可使用。

## 训练监控器（自动修复）

`TrainingMonitor` 是一个后台服务，每 **30 秒**运行一次，收集当前训练运行的所有可用信号，并提交给 LLM 进行分析。

### 收集的信号

- **进程状态**：alive / dead / unknown
- **output.log 尾部**：最后 1500 个字符（优先提取 traceback）
- **指标**：最近条目 + 前半段 vs 后半段趋势分析
- **停滞时长**：自最后一次产生指标以来的秒数
- **当前 train.py**：完整脚本源码（用于精确修复）

### 决策框架

LLM 将每次检查分类为三种操作之一：

| 决策 | 触发条件 | 执行动作 |
|------|---------|---------|
| **LGTM** | 训练正常推进 | 无操作 |
| **WARNING** | Loss 平台期、reward hacking、KL 爆炸等 | 向用户报告观察结果 |
| **FIX** | 脚本崩溃、进程死亡并有 traceback | 自动修复并重启 |

### 自动修复流程

当需要 FIX 时：

1. LLM 输出诊断 + 完整修复脚本
2. 监控器将旧 `train.py` 归档为 `train_v{N}.py`
3. 将修复脚本写为新的 `train.py`
4. 通过 `resume_training` 重新启动训练
5. 重置停滞追踪

安全保障：
- 每个运行最多 **3 次自动修复尝试**（防止无限重试循环）
- 修复尝试按 `run_id` 追踪
- 快照去重避免对未变化状态的重复分析

## 基于文件的连接层

Auto 通过本地文件系统与训练进程通信：

```
~/.cache/twinkle/{run_id}/
├── meta.json       — 运行元数据（model_id、config、status、pid）
├── metrics.jsonl   — 每步一个 JSON 对象（增量）
├── output.log      — 训练的 stdout+stderr 合并输出
├── train.py        — 当前活动训练脚本
└── train_v{N}.py   — 归档的历史脚本版本
```

### 训练控制模型

在 Server 模式下，Twinkle Server 将所有模型/优化器状态保留在 GPU 内存中：

- **暂停** = 杀死客户端进程 (SIGKILL) — 服务器状态保留
- **恢复** = 重新启动客户端脚本 — 无缝继续训练
- **停止** = SIGTERM — 触发检查点保存后退出
- **关闭服务器** = 释放 GPU 资源，**销毁**模型状态

## TrainingRuntime（脚本集成）

训练脚本使用 `TrainingRuntime` 与 Auto 集成：

```python
from twinkle_client.auto.runtime import TrainingRuntime

rt = TrainingRuntime(run_id='my-grpo-run')
rt.start(model_id='Qwen/Qwen3.5-4B', config={'lr': 1e-5})
rt.register_graceful_shutdown(model, dataloader)

for step, batch in enumerate(dataloader):
    # ... 训练逻辑 ...
    rt.log_metrics(step=step, loss=loss, reward=reward, grad_norm=gn, lr=lr)
    rt.log(f'Completed step {step}, loss={loss:.4f}')

rt.finish()
```

### 核心方法

| 方法 | 说明 |
|------|------|
| `start(model_id, config, script_path)` | 初始化运行目录和元数据 |
| `log_metrics(**kwargs)` | 向 `metrics.jsonl` 写入指标条目 |
| `log(message)` | 打印日志消息（被捕获为 `output.log`） |
| `get_resume_info()` | 获取 `last_step` 用于从检查点恢复 |
| `finish(status)` | 标记训练完成，关闭文件 |
| `register_graceful_shutdown(model, dataloader)` | 注册 SIGTERM 处理器以保存检查点 |

### 断点续训支持

`TrainingRuntime` 自动将训练进度保存到 `meta.json`（每 5 秒节流写入一次）。脚本可以使用 `get_resume_info()` 从上次保存的步数恢复：

```python
rt = TrainingRuntime(run_id='my-run')
resume = rt.get_resume_info()
global_step = resume['last_step']

if global_step > 0:
    dataloader.skip_consumed_samples(global_step * BATCH_SIZE)
    print(f'从 step {global_step} 恢复训练')
```

### 优雅关停

调用 `register_graceful_shutdown()` 后，会安装一个 SIGTERM 处理器：

1. 保存模型检查点（LoRA 权重 + 优化器状态）
2. 保存数据加载器位置（`consumed_train_samples`）
3. 记录检查点路径
4. 标记训练为 `stopped` 并退出

## 日志记录

所有日志写入 `./auto.log`（当前工作目录）：

- 5MB 时轮转，保留 3 个备份
- **无控制台输出** — 所有输出写入日志文件
- 使用 `--verbose` 启用 DEBUG 级别日志
