# RolloutResume RFC：Group-Atomic Durable GRPO

- 状态：Gate0 通过；阶段 1-4 原型完成；A2 恢复与训练组收益探针通过
- 日期：2026-11-18
- 实施分支：`codex/rollout-resume-20261118`
- 上游基线：`origin/dev-swift-v5@faafaf4189b4a88a434add0108b80e11cae30ab8`
- 目标后端：Ascend 910B4、32 GB HBM、Qwen3-0.6B、LoRA

## 1. 结论先行

本提案补的是 **live rollout 的组级事务化恢复**，不是再实现一遍模型、Optimizer、Scheduler、RNG 或 DataLoader 续训。

当前实现已经把恢复从“文件能读回来”推进到以下可验证语义：

- 半完成 GRPO 组恢复后继续补齐，仍作为同一完整组进入训练；
- group 从准入到交付固定 behavior policy 的 `version + adapter_path`，恢复或补采不得改用当前新策略；
- ready group 超 staleness 时整组丢弃，禁止按成员混入新策略；
- batch lease 与训练 checkpoint reference 关联，已提交 lease 不再重新交付；
- `ClientCheckpointManager` 保留历史 adapter path，episode 每次采样显式选择其绑定路径；
- LocalEnv workspace、TurnLedger 与工具副作用边界已经可通过 A2 真机进程强杀恢复。

这些结果证明的是 **故障条件下的可训练组保留、采样真值一致和重复更新抑制**。它不会改变无故障训练曲线，也尚未证明最终模型 reward 一定提高；最终收益仍以故障注入 A/B 的训练组 yield、优化步效率和固定预算 reward 为准。

当前上游已具备：

- #340：模型、Optimizer、Scheduler、RNG、训练样本消费位置和训练批 pending 的恢复；
- #361/#375/#396：多轮 rollout、完整组交付、控制通道和尾批预取；
- #387：服务端任务 loader 与训练 checkpoint；
- DataLoader：`get_state()` / `load_state()` 与确定性 epoch/offset。

当前上游缺少：

- `StreamingDriver.state_dict()` 只有 `handed` 和 `prompts_drawn`，没有 `load_state_dict()`；
- prompt 已被抽取但尚未训练时，恢复会跳过整个预取窗口；
- `_Group`、`_Episode`、`_ready`、奖励等待和完整组 attempt 只在内存；
- `MultiTurnRollout` 的 `TurnLedger`、tool loop 计数和 live env session 只存在当前进程；
- 环境没有跨进程恢复契约；
- 已执行工具与 observation 落盘之间没有 exactly-once 边界；
- batch 在 yield 前就从 ready 队列删除并增加 `_handed`，训练提交前进程退出会产生“已交付但未训练”的语义窗口。

因此，只有 Gate0 能稳定复现“现有接口无法无损恢复”时，本方向才继续。若只能通过改一个游标或新增 `enable_resume` 开关解决，立即降级为普通 bugfix，不作为获奖主线。

## 2. 与已有 PR 的边界

| 已有能力 | 负责范围 | 本提案不重复实现 |
|---|---|---|
| #340 | 权重、Optimizer、Scheduler、RNG、训练数据游标 | 模型训练状态本身 |
| #361 | 客户端多轮 rollout 与完整组交付 | 普通 rollout 调度 |
| #375/#396 | streaming 控制、批量交接、尾批预取 | 并发控制和预取策略 |
| #387 | 服务端任务 loader / checkpoint | 服务端任务状态 |
| #397/#398 | 权重加载与 kernel 性能 | 采样器或 HCCL/NCCL 优化 |
| DataLoader | epoch、offset、随机种子恢复 | 平行数据游标实现 |

本提案只在既有 checkpoint manifest 中增加 rollout 补层，并与 #340 的训练 checkpoint 原子关联。

## 3. 术语与不变量

### 3.1 `state_key`

`state_key` 是内部稳定身份，不是用户配置项：

```text
state_key = run_id / group_key / episode_id / turn / attempt
```

- `run_id`：一次训练运行的稳定 ID；
- `group_key`：一个原始 prompt 的完整 GRPO 组；
- `episode_id`：组内成员身份；
- `turn`：turn boundary；
- `attempt`：质量再生或调度重试次数。

禁止用随机 session id 或单调 deadline 代替 `state_key`。

### 3.2 必须保持的语义

1. 一个 prompt 在一个 epoch 中只分配一次，除非整组被明确丢弃并记录原因。
2. 一个完整组只能使用同一 behavior policy 的 `version + adapter_path`；不得用新策略补旧组缺失成员，也不得只凭 version 数字相同就假定实际权重相同。
3. 已提交的 turn 必须保留原 `input_ids`、`labels`、`completion_mask`、`response_positions` 和 `rollout_logps`。
4. 工具副作用最多执行一次；无法证明幂等时，恢复必须 fail closed。
5. 未提交的训练 batch 在恢复后可以重新交付同一完整组，但不得跳号或重复计入已完成 batch。
6. checkpoint 完整标记落盘前，崩溃按“上一份完整 checkpoint”处理。

## 4. 当前丢失窗口审计

### 4.1 Prompt 窗口

`StreamingDriver._next_row()` 在训练前读取 dataloader；`_prompt_seq` 随分配递增。现有 `state_dict()` 返回该计数，但没有保存正在执行或 ready 中的 prompt 内容。

如果调用方把 DataLoader 恢复到 `prompts_drawn`，则尚未训练的 in-flight/ready prompt 会被跳过。源码已有 TODO 明确承认该风险。

### 4.2 完整组窗口

`_Group` 当前包含 `key`、原始 `row`、`attempt`、`source`、`members`、`baselines` 和 `missing`。`_Episode` 当前包含 episode/group/role/policy version、trajectory、`RolloutStatus`、ORM/PRM pending、PRM segment、held index、训练行、submit trajectory 和 no-slot 重试状态。

这些字段没有持久化；只保存最终 trajectory 会丢失分数、未完成 turn、baseline 和再生预算。

### 4.3 Rollout 窗口

`MultiTurnRollout._Episode` 保存 ledger、ToolManager、预算、malformed/stuck 计数、agent 请求队列和已冻结 segment。进程重启后，`_episodes` 中找不到 episode，`step()` 会直接报错。

`TurnLedger` 的 `_pif`、`_logprobs` 与 turn 计数是训练真值。不能从 transcript 重新编码恢复，因为 tokenizer 可能给同一文本生成不同 token id，旧 logprob 会错位。

### 4.4 环境窗口

| 环境 | 进程内 | 跨 Twinkle 进程 | 当前结论 |
|---|---|---|---|
| LocalEnv | workspace + running procs | 无 reattach API | 本 MVP 增加有限持久化 |
| 嵌入式 OpenEnv | Python 对象 | 对象丢失 | fail closed / drain |
| OpenEnvClient | live WebSocket | 无 session reattach token | fail closed / drain |
| AgentEnv | 服务端可暂停原 sandbox | 适配器无按 sandbox id attach | fail closed / drain |

现有 `Env.snapshot()` 是给模型查看的截断文本，不是 checkpoint，不能用于恢复。

## 5. 设计

### 5.1 Env 能力扩展点

不新增“是否启用恢复”的布尔配置。通过能力发现决定可恢复性：

```python
class EnvResumeCapability(Protocol):
    def checkpoint_episode(self, state_key: str, episode_id: str) -> dict: ...
    def restore_episode(self, state_key: str, state: dict) -> None: ...
    def discard_episode_state(self, state_key: str) -> None: ...
```

能力分三档：`none`、`live_attach`、`durable_checkpoint`。能力声明必须说明覆盖范围：workspace、running processes、interpreter state、open handles、provider session。未声明能力的环境不能静默按 `durable_checkpoint` 处理。

MVP 只实现 `LocalEnv` 的 workspace durable checkpoint：

- 只允许 turn boundary、且没有运行中子进程时写入；
- workspace 使用临时目录 + 原子 rename；
- 恢复时创建新的本地 workspace 并绑定原 `state_key`；
- 不尝试恢复 OS 进程、打开文件描述符或网络连接；
- 任一不满足即抛 `UnsafeCheckpoint`，整组 fail closed。

### 5.2 Rollout checkpoint

`MultiTurnRollout` 增加 episode checkpoint/restore：

- 保存 `trajectory` 的纯数据视图；
- 保存 ledger 的 `_pif`、`_logprobs`、`_turns`、sampling masks；
- 保存 tool loop 计数：budget、spent、followups、malformed、stuck、seen calls；
- 保存 `last_reply`、closed rows、stop/truncated/error 状态；
- 保存 policy version、当前 turn 和剩余超时，不保存 `Future`、线程、Queue、monotonic deadline 原值；
- 恢复时重建 ToolManager，并重新绑定已恢复 env session；
- 对非文本、不可序列化媒体或未恢复状态，明确拒绝 checkpoint。

### 5.3 Driver journal 与完整组

journal 记录状态转换，不记录模型权重：

```text
PROMPT_ASSIGNED
TURN_COMMITTED
TOOL_INTENT
TOOL_RESULT
SCORE_RESULT
GROUP_READY
GROUP_DROPPED
BATCH_LEASED
MODEL_CHECKPOINT_COMMITTED
```

元数据写 JSON，token/logprob 大数组写独立二进制文件；每次状态转换先 fsync 再释放外部副作用。最后一行损坏时可截断到最后一条完整记录。

保存内容：prompt journal 与 DataLoader state；group 固定 version/adapter path、attempt/source/members/baselines/missing；ready 顺序、baseline 与 ORM/PRM 分数；version 和计数器；live episode 的 rollout/env checkpoint；batch lease 与模型 checkpoint manifest 的关联。

不保存：Future、actor handle、线程、Queue、monotonic 时间或不可序列化回调。恢复后由新进程重新注入回调并校验版本。

### 5.4 Exactly-once 交付

训练 batch 采用 lease，并要求 commit 时提供非空 checkpoint reference：

1. 从 ready 组生成 batch，写入 `BATCH_LEASED`；
2. 训练完成后写 `MODEL_CHECKPOINT_COMMITTED`，payload 包含 checkpoint、batch、policy versions 和训练 manifest；
3. 两者都完整时，该组才标记 trained；
4. checkpoint 前崩溃，回退到上一份模型 checkpoint 并重新交付同一 lease；
5. checkpoint 完成后崩溃，不得再次训练该组。

当前通过稳定 lease id 和 journal 防止显式重复 commit。若训练器无法校验“模型 checkpoint 已持久化”和 commit 标记之间的原子关系，恢复仍只能保守回退，不能对外声称无条件 exactly-once；该上层事务接口仍需在真实训练 cookbook 中接入验证。

### 5.5 工具副作用

工具调用必须有 `effect_key = state_key + call_index + canonical_arguments_hash`。

- 执行前写 `TOOL_INTENT`；
- 执行后写 `TOOL_RESULT`；
- 恢复时若只有 intent、没有 result，且环境具备 workspace rollback，则回到 execution 前状态后重试；
- 无 rollback 或 tool 不声明幂等时，整组丢弃并显式记录；
- 不允许只重跑组内一个成员后与旧成员混合。

## 6. 分阶段实施

### 阶段 0：Gate0 反例与探针

固定 prompt 序列、多轮工具 episode、turn/score/ready/lease/checkpoint 故障注入、LocalEnv checkpoint 可行性探针，以及连续运行与“现有 API 恢复”对照。Gate0 不修改生产代码。

状态：已完成。已复现 in-flight prompt 跳过、ready 组不可恢复、live rollout 不可重连、工具副作用重放，并验证 workspace 复制/回灌可行。

### 阶段 1：journal 与 prompt exactly-once

journal 原子写、截断恢复和版本校验；prompt assigned/consumed 边界；DataLoader state 绑定；重启后不跳号、不重读已提交 prompt。

状态：原型完成，hash-chain JSONL、最后一条 torn record 修复、中间损坏 fail closed 已覆盖测试。

### 阶段 2：完整组与 ready 快照

group attempt、baseline、分数和 ready 顺序；旧组缺失成员禁止跨策略补采样；未交付 ready group 的完整恢复。

状态：原型完成。group 固定 `version + adapter_path`；ready 组 staleness 触发整组淘汰；半完成组恢复后继续同一策略。

### 阶段 3：rollout turn boundary

ledger 与 tool loop 状态导出/导入；LocalEnv workspace checkpoint；不支持环境 fail closed；工具 intent/result journal。

状态：原型完成，包含 token/labels/mask/logprob 真值、workspace 代际快照和工具 effect 边界。

### 阶段 4：batch lease 与训练 checkpoint 关联

lease/commit；回退到上一份完整模型 checkpoint；训练与 rollout journal 一致性检查。

状态：核心 lease/commit 完成，训练侧仍需在真实 cookbook 中把 `model.save(..., save_optimizer=True)` 与 `driver.commit_batch(checkpoint_ref, manifest)` 串接。

### 阶段 5：A2 实测

Qwen3-0.6B LoRA、单 NPU、至少 12 条多轮文本任务、`num_generations=2`、`batch_size=4`、`max_turns=3`、序列上限 1024；在固定点执行 `SIGKILL`；上报恢复 turn、避免重跑的 token 数、工具副作用次数和训练样本集合。

状态：机制探针完成，尚缺完整训练曲线。2026-11-18 在 A2 的 Ascend 910B4 上完成：

- Qwen3-0.6B 真实采样进程 `SIGKILL` 后，新进程恢复 24 个已采样 token，`restored_exact=true`；
- 第一轮 logprob 全部有限，恢复后继续采样 16 个 token；
- LocalEnv workspace 恢复后读到 `turn-1-durable`；
- A2 聚焦测试 `24 passed`；
- 训练组收益探针：普通游标重启 `0` 个可训练行，durable resume `2` 行（`1` 个同策略完整组），重复 commit 返回 `None`，commit 后 lease 为空。

## 7. Gate0 失败判据

出现任一项立即停止本方向：

1. 现有 API 无新增接口即可无损恢复 live episode/ready group；
2. LocalEnv 固定种子下无法稳定复现组丢失或工具重放；
3. 无法保留已发生 turn 的 token/label/logprob，只能重新采样；
4. 任一允许的路径出现工具副作用重复；
5. 只能靠拆组、混用 policy version 或跳 prompt 得到“成功”；
6. 实现退化成单个游标修复或 `enable_resume` 开关；
7. 与 #340 的训练 checkpoint 语义重复；
8. Qwen3-0.6B 无法稳定产生可观测的多轮 tool call；
9. 核心实现超过 2000 行，或必须修改 AgentENV/OpenEnv 服务端协议。

## 8. 代码预算

| 模块 | 预算 |
|---|---:|
| journal / schema / 原子写 | 280 |
| StreamingDriver 快照、lease、恢复 | 240 |
| rollout / ledger checkpoint | 420 |
| Env capability + LocalEnv workspace | 260 |
| tool effect journal / 接入 | 180 |
| 合计生产代码 | 1380 |

测试、cookbook 和文档另计；生产代码硬上限 2000 行。

## 9. 最终验收

连续运行与恢复运行必须比较：prompt 集合与顺序；group member 集合和 policy version；`input_ids`、`labels`、`completion_mask`；`response_positions`、`rollout_logps`；ORM/PRM/baseline 分数；工具 effect ledger 次数；已完成 optimizer step 和模型 checkpoint hash；避免重跑的 rollout token 数。

没有 A2 实测和上述一致性证据，不提交“提升训练效率”或“可获奖”结论。

## 10. 当前实测结论

| 指标 | 普通游标重启 | Durable GRPO | 结论 |
|---|---:|---:|---|
| 可训练行 | 0 | 2 | 半完成组被保留 |
| 可训练完整组 | 0 | 1 | 同 policy version |
| 恢复成员 | 0 | 1 | 只补未完成成员 |
| 重复 commit | 0 | 0 | 已提交 lease 不再交付 |
| NPU token 真值 | 丢失 | 24 个逐值保留 | 不重新编码 |
| 工具 observation | 丢失 | `turn-1-durable` | workspace 可恢复 |

因此当前方案已经不再是“只恢复日志”：它在故障场景下保留了原本会丢失的训练组，并阻止旧组跨策略补采。下一阶段唯一值得继续做的是把 commit 接口接入真实训练 cookbook，并在固定故障率下比较最终 reward、有效 optimizer step 和 wall-clock；在此之前不应宣称完整训练收益。
