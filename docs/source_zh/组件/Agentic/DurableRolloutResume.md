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

新 driver 快照同时写入结构化 `binding` 和旧版扁平字段。读取旧快照时会自动升级；
binding 与扁平字段不一致时拒绝恢复。

## 自动持久化边界

启用 `resume_store` 后，driver 在以下稳定边界自动保存快照：

1. prompt 分配并完成成员提交；
2. 完整组进入 ready 队列；
3. batch lease 写入后；
4. `commit_batch()` 完成 checkpoint 关联后；
5. 组淘汰后。

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
checkpoint 且不再交付该 lease。

## ResumeStore 与 DurableJournal

`ResumeStore` 由两层组成：

- `DurableJournal`：带 hash chain 的 JSONL 事件日志，支持尾部截断修复、跨线程和跨进程
  文件锁，以及写入失败回滚；
- object store：保存 prompt、episode、lease 和 driver 快照，读取时校验 SHA-256。

故障边界：

- journal 最后一行未写完：移除不完整尾部；
- 中间记录损坏或 SHA 校验失败：拒绝恢复；
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
幂等外部调用。已完成且被 episode checkpoint 覆盖的 effect 直接复用记录结果。

## 兼容性

- `resume_store=None`：行为与原有 `StreamingDriver` 相同；
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
checkpoint，并在每个 batch 后调用 `commit_batch()`。

聚焦测试：

```bash
PYTHONPATH="$PWD/src" python -m pytest \
  tests/twinkle_agentic/test_rollout_resume_store.py \
  tests/twinkle_agentic/test_localenv_resume.py \
  tests/twinkle_agentic/test_turn_ledger_resume.py \
  tests/twinkle_agentic/test_multi_turn_resume.py \
  tests/twinkle_agentic/test_group_policy_binding.py \
  tests/twinkle_agentic/test_streaming_driver_resume.py \
  tests/twinkle_agentic/test_resume_boundaries.py \
  tests/twinkle_agentic/test_tail_prefetch.py \
  tests/twinkle_client/test_checkpoint_policy_history.py \
  -q
```

## 实验结论与边界

已验证：

- 真实 Qwen3-0.6B LoRA 在 Ascend 910B4 上完成 optimizer step 后强杀；
- checkpoint 临时目录、已发布未提交、已提交三种边界均可正确回退或前进；
- 已提交 batch 不重复训练，未提交 optimizer 更新不泄漏；
- 无故障时 baseline 与 durable 的训练信号和评测结果逐值一致；
- 50% 组丢失故障下，保留完整组的 durable 路径在官方 GSM8K 测试集上取得显著提升。

尚未覆盖：

- 可写 NFS 或对象存储上的 lease、journal 和 adapter 故障矩阵；
- 多卡 HCCL/FSDP2 和大模型验证；
- 无法 checkpoint 的远端 agent harness；
- 所有外部工具协议都具备幂等结果查询。

因此该机制不以“所有故障率下无条件提升”为结论，而是在完整组丢失或进程重启时保持
GRPO 组语义、事务一致性和固定训练预算下的有效训练量。
