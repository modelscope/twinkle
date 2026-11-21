# 可恢复的流式 GRPO Rollout

`StreamingDriver` 的 durable 模式用于解决一个训练语义问题：进程重启后，不能只恢复
“已经生成了多少条轨迹”，还必须保证完整 GRPO 组、behavior policy、batch lease 和
optimizer checkpoint 属于同一次可验证提交。

该能力默认关闭。未传入 `ResumeStore` 时，`StreamingDriver` 保持原有内存流式行为。

## 核心不变量

1. 一个 GRPO 组只使用一个不可变 behavior policy。
2. 恢复、成员补采和组替换都沿用首次准入时的 policy binding。
3. 一个 batch 在模型 checkpoint 持久化前保持 lease；恢复后重交付同一批轨迹。
4. checkpoint 提交后，同一 lease 不再交付，避免重复 optimizer step。
5. 无法证明安全恢复的环境或外部工具副作用必须 fail closed。

## GroupPolicyBinding

每个组在准入时固定以下信息：

```python
from twinkle_agentic.async_rl.rollout_resume import GroupPolicyBinding

binding = GroupPolicyBinding(
    state_key="run/group/17",
    policy_version=4,
    adapter_path="/checkpoints/policy-4",
)
```

- `state_key` 是跨进程稳定的逻辑组 ID，不使用随机 session ID。
- `policy_version` 是该组采样时的 policy 版本。
- `adapter_path` 是该版本的 adapter 快照；`None` 表示使用基础模型。

`adapter_path` 在一次 run 内必须指向不可变内容，不能原地覆盖或复用为另一个 policy；建议路径包含
run、policy version 或 checkpoint generation。当前 binding 校验身份和路径一致性，不对远端 adapter
内容额外计算哈希，因此 artifact 不可变性由 checkpoint 管理层保证。

新 driver 快照同时写入结构化 `binding` 和旧版扁平字段。读取旧快照时会自动升级；
binding 与扁平字段不一致时拒绝恢复。

## 自动持久化边界

启用 `resume_store` 后，driver 在以下稳定边界自动保存快照：

1. prompt 分配并完成成员提交；
2. 每个 rollout turn、ORM/PRM 打分提交及结果返回；
3. 完整组进入 ready 队列；
4. batch lease 写入后；
5. `commit_batch()` 完成 checkpoint 关联后；
6. 组淘汰后。

PRM 按 segment 保存打分输入；恢复时只补发缺失的分数，并继续遵守 `on_step` 屏障。
组进入 ready 或被淘汰的安全快照落盘后，其 prompt、episode object 和 LocalEnv workspace
快照会被回收；journal、driver 快照和未提交 lease 仍保留用于恢复。

调用方不需要在故障前手工调用 `state_dict()`。恢复入口为：

```python
driver = StreamingDriver(
    dataloader,
    rollout,
    batch_size=8,
    num_generations=4,
    checkpoint_manager=policy_index,
    resume_store=store,
    run_id="gsm8k-durable",
)

if store.has_object("driver", "gsm8k-durable"):
    driver.load_state_dict()
```

## Checkpoint 事务

训练侧必须按以下顺序提交：

```python
driver = StreamingDriver(..., resume_store=store, run_id="run")

rows, facts = driver._next_batch()
model.forward_backward(rows, ...)
model.clip_grad_and_step()

checkpoint = model.save(..., save_optimizer=True, async_save=False)
driver.commit_batch(
    checkpoint,
    manifest={
        "optimizer_step": optimizer_step,
        "policy_version": next_policy_version,
    },
)
```

`commit_batch(checkpoint_ref, manifest)` 是 batch lease 与训练 checkpoint 的唯一关联点。
未提交前崩溃，恢复上一模型 checkpoint 并重交付同一 lease；已提交后崩溃，恢复新
checkpoint 且不再交付该 lease。调用方必须先确保模型和 optimizer checkpoint 已完整持久化，再调用
`commit_batch()`；driver 不探测通用远端 checkpoint 的落盘状态，也不能修复一个提前提交的引用。

## ResumeStore 与 DurableJournal

`ResumeStore` 由两层组成：

- `DurableJournal`：带 hash chain 的 JSONL 事件日志，支持尾部截断修复、跨线程和跨进程
  文件锁，以及写入失败回滚；
- object store：保存 prompt、episode、lease 和 driver 快照，读取时校验 SHA-256。

故障边界：

- journal 最后一行未写完：仅移除没有终止换行的不完整尾部；
- 任意完整记录损坏或 SHA 校验失败（包括最后一条 commit）：拒绝恢复；
- 磁盘写入失败：回滚不确定的 append；
- object 文件不完整：拒绝解码；
- 旧 lease 已有 commit 记录：恢复时自动退休。

## 可恢复环境

环境恢复能力采用独立 `ResumableEnv` protocol，不向所有 `Env` 实现增加空方法：

```python
from twinkle_agentic.envs.resume import EnvResumeCapability

class MyDurableEnv:
    resume_capability = EnvResumeCapability.DURABLE_CHECKPOINT

    def checkpoint_episode(self, episode_id, state_key):
        ...

    def restore_episode(self, episode_id, state_key, state, trajectory=None):
        ...

    def discard_episode_state(self, state_key):
        ...
```

未声明恢复能力的 `AgentEnv`、`OpenEnv` 和 `OpenEnvClient` 在 episode 打开时 fail
closed。只恢复文本对话、但丢失 workspace 或工具状态的实现不允许静默继续。

## 外部工具副作用

每个工具轮次区分 intent 和 result：

1. 执行前写入 `TOOL_INTENT`；
2. 将工具 observation 写入 episode checkpoint；
3. checkpoint 持久化后写入 `TOOL_RESULT`。

恢复时如果存在 intent 但没有 checkpoint 能证明结果，整组状态被拒绝，不会再次执行非
幂等外部调用。已完成且被 episode checkpoint 覆盖的 effect 直接复用记录结果。这一策略是 fail
closed，不等于为任意外部系统提供 exactly-once 执行；要继续该 uncertain effect，业务侧必须提供
幂等键或结果查询/对账能力。

## 兼容性

- `resume_store=None`：行为与原有 `StreamingDriver` 相同；
- 一个 `run_id` 只对应一个活跃 driver 写者；journal 文件锁用于保护事件链，不允许多个 driver
  共享同一 run 并分别覆盖 driver/lease 快照；
- 旧 driver 快照：自动从 `version + adapter_path` 升级为 `GroupPolicyBinding`；
- 未设置 checkpoint manager：仍可保存组和 lease，但不执行 policy 版本淘汰；
- `ResumableEnv` 从 `twinkle_agentic.envs.resume` 或 `twinkle_agentic.envs` 导入；
- 旧的 `twinkle_agentic.envs.base` 导入路径保持兼容。

## 示例与测试

端到端示例：

```text
cookbook/rl/async_rl/durable_multi_turn_grpo.py
cookbook/rl/async_rl/durable_multi_turn_grpo.sh
```

示例使用官方 GSM8K、Qwen3.5、LoRA、真实多轮 rollout、GRPO 反向传播和 optimizer
checkpoint，并在每个 batch 后调用 `commit_batch()`。每个已提交 step 的训练指标、policy
version、driver 计数和累计训练耗时（不含进程停止期间）写入
`TWINKLE_RESUME_ROOT/training-metrics.jsonl`。指标同时随 checkpoint commit manifest 持久化；
如果进程在 commit 后、JSONL 追加前退出，重启会从 journal 回填该 step，保证曲线没有缺步。

聚焦测试：

```bash
PYTHONPATH="$PWD/src" python -m pytest \
  tests/twinkle_agentic/test_rollout_resume_store.py \
  tests/twinkle_agentic/test_localenv_resume.py \
  tests/twinkle_agentic/test_turn_ledger_resume.py \
  tests/twinkle_agentic/test_multi_turn_resume.py \
  tests/twinkle_agentic/test_group_policy_binding.py \
  tests/twinkle_agentic/test_rollout_seed_derivation.py \
  tests/twinkle_agentic/test_streaming_driver_resume.py \
  tests/twinkle_agentic/test_resume_boundaries.py \
  tests/twinkle_agentic/test_tail_prefetch.py \
  tests/twinkle_client/test_checkpoint_policy_history.py \
  -q
```

## 实验结论与边界

在单张 Ascend 910B4、Qwen3-0.6B、vLLM 0.23.0 和 vLLM-Ascend 0.23.0 上，正式 cookbook
的两步 GSM8K 训练、after-save 和 after-commit 进程崩溃恢复均已跑通。两种故障恢复后的
`history_steps` 与 `commit_steps` 都为 `[1, 2]`；after-commit 缺步指标可从 journal 回填。

真实 vLLM A/B 在第 2、5 步保存 checkpoint 后、commit 前退出进程，结果如下：

| 预算 | durable | cursor-restart |
|---|---|---|
| 固定 8 attempts | 8 optimizer steps，737 秒 | 6 optimizer steps，590 秒 |
| 固定 8 optimizer steps | 8 attempts，621 秒 | 10 attempts，623 秒 |

固定 step 下减少 20% 的 rollout attempts，但本轮墙钟时间基本相同，不能宣称耗时提升。
相同 64 条 held-out GSM8K 上，base、durable、cursor-restart 准确率分别为 45.3125%、
46.875%、45.3125%；durable 的差值为 +1.5625pp，但 McNemar 双侧 `p=1.0`，不构成
统计显著的质量提升证据。完整逐 step 曲线由四组 `training-metrics.jsonl` 生成。

### 长程多工具故障恢复对照

在相同单卡环境上使用 Qwen3-4B-Instruct-2507 进一步运行真实 vLLM 多工具 GRPO。每组实验包含
10 个 optimizer steps，每个 batch 有 2 个 prompt 组、每组 2 条采样；每条完整 episode 执行 6 个
订单工具阶段并生成最终回复。在 policy 0、3、6 的旧组仍在 rollout、且新 policy 已提交时终止进程，
三种实现均经历 3 次重启，最终 `history_steps` 都完整为 `[1, ..., 10]`。

实验使用真实模型生成和 `ToolManager` 调用，工具副作用写入持久化审计日志。`cursor-restart` 仅恢复
数据游标并重新运行未提交工作；`naive-episode-resume` 是反例基线，恢复 episode 但重新绑定最新 policy，
不属于推荐实现。

| 指标 | durable | cursor-restart | naive-episode-resume |
|---|---:|---:|---:|
| 墙钟时间（秒） | 943 | 970 | 910 |
| 实际 generation 次数 | 449 | 495 | 458 |
| 实际生成 response tokens | 13018 | 14455 | 13349 |
| 跨重启重复 generation 次数 | 1 | 326 | 0 |
| 跨重启重复 response tokens | 14 | 9557 | 0 |
| 跨重启重复工具调用 | 0 | 76 | 0 |
| 其中重复非幂等工具调用 | 0 | 32 | 0 |
| episode 中途切换 policy 的数量 | 0 | 不适用（整组重启） | 26 |
| 错误重绑定产生的 generation / tokens | 0 / 0 | 不适用 | 65 / 1868 |

durable 相对 cursor 的总生成 token 减少 9.941%，跨重启重复 token 减少 99.854%，并避免 76 次重复
工具调用。只有一次 14-token generation 在故障边界重做，不能把该结果写成绝对零重复采样。单组墙钟
差异不足以证明通用提速。

naive 的 26 条混采 episode 最终被 staleness 规则淘汰，交付 optimizer 的混采组为 0；因此本实验证明
durable 消除了恢复时的错误 policy 切换，但不宣称已经量化混采对模型质量的损害。训练曲线和逐次生成、
工具、lease、journal 原始记录均已留存。

### 8B、双卡与新版 CaMem 补充验证

在另一台单节点双卡 Ascend 910（64GB/卡）上，使用 Qwen3-8B、CANN 9.1.0、torch_npu 2.10.0.post7、
vLLM 0.33.0 和 vLLM-Ascend 0.32.0.dev0 补做验证：

| 场景 | 结果 |
|---|---|
| 单卡无故障 | 完成 1 个 optimizer step 和 IPC 权重同步；训练与 vLLM 合计占用约 47.4GB |
| 单卡安全边界退出 | 1 次重启后恢复 2 组、4 条成员，完成 `[1,2,3]`；跨重启重复工具调用为 0 |
| 单卡不确定工具窗口 | pending intent 触发 fail closed，未重复执行非幂等工具；不将其算作训练恢复成功 |
| 双卡无故障 | 双 rank 训练完成，2 个 vLLM worker 均完成 IPC 权重同步 |
| 双卡安全边界退出 | 1 次重启后恢复 2 组、4 条成员，完成 `[1,2,3]`；重复工具及中途 policy 切换均为 0 |
| 单卡 CaMem 两步 | 两轮 weights/KV sleep-wake 和第二轮 LoRA-only 同步完成，未出现旧版 `aclnnInplaceCopy` 错误 |

该结果只覆盖上述版本组合和单节点双卡，不回溯证明 vLLM/vLLM-Ascend 0.23.0，也不代表 FSDP2
或多节点已经验证。本地 overlay 的规模基准显示，journal 在 250、500、1000、2000 事件时的平均追加
延迟约为 1.89、3.53、6.71、13.16ms，确认全量重扫是优先优化项；64MiB LocalEnv workspace 快照约
85ms，在该本地盘上暂不是主要瓶颈。

尚未覆盖：

- 可写 NFS 或对象存储上的 lease、journal 和 adapter 故障矩阵；
- 双卡 native_fsdp 以外的 FSDP2、多节点和更大规模验证；
- vLLM-Ascend 0.23.0 在 CaMem sleep/wake 后激活多 policy LoRA 的稳定性；上述 4B 对照使用无 sleep
  colocate 路径，新版结果不能替代旧版复测；
- 无法 checkpoint 的远端 agent harness；
- 所有外部工具协议都具备幂等结果查询；
- 大规模长训练的持久化开销尚未形成独立基准。当前 journal 追加会重扫事件链，driver 在稳定边界
  保存完整状态，LocalEnv 每轮复制并校验 workspace；后续可分别评估增量扫描、按 episode 分片和
  增量 workspace 快照，不能用当前端到端墙钟结果替代这些专项评估。

因此该机制的主要结论是进程重启时保持 GRPO 组语义、事务一致性、固定预算下的有效训练量，
并减少重复工具副作用和重复采样，不宣称所有故障率下无条件提升模型质量或缩短墙钟时间。
