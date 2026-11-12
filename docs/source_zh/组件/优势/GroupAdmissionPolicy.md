# GroupAdmissionPolicy

GRPO 组内奖励完全相同时，中心化后的 advantage 全为零；对于稠密或组合奖励，非零
差异也可能低于奖励的有效分辨率，标准化会把这种微小差异放大成较强的相对信号。
`GroupAdmissionPolicy` 将质量判定与采样基础设施解耦，只对完整 prompt 组执行
准入或拒绝。调用方可以在硬预算内补采有效组，避免无界重试。

本组件有意通过专用子模块导入，而不从 `twinkle.advantage` 根命名空间重新导出：它
适用于 GRPO、DAPO-style 等 group-relative 方法，并不适用于所有 advantage 估计器。

## 判定信号

策略检查两类可选条件：

- reward 条件检查奖励总体标准差、奖励极差，以及非平凡中心化 advantage 的数量；
  `min_reward_range` 表示由场景定义的有效奖励分辨率；
- 可选的近重复条件检查字符串表示是否过度相似。默认的 token n-gram
  Jaccard 只是零依赖的字面基线，不判断语义等价；有需要的场景可注入自定义 scorer。

所有阈值默认关闭，因此默认策略会放行全部合法组，不改变已有训练行为。

```python
from twinkle.advantage.group_admission import (
    GroupAdmissionConfig,
    GroupAdmissionPolicy,
)

policy = GroupAdmissionPolicy(
    GroupAdmissionConfig(
        min_reward_std=1e-6,
        min_reward_range=0.02,
        min_nontrivial_advantages=2,
        max_mean_pairwise_similarity=0.95,
    )
)

decision = policy.evaluate(
    rewards=[0.0, 1.0, 0.0, 1.0],
    completions=["答案 A", "答案 B", "答案 C", "答案 D"],
)
if decision.admitted:
    train_complete_group()
```

被拒绝的组会暴露一个互斥的 `primary_rejection_reason`：

- `exact_dead`：reward gate 拒绝了完全同分组；
- `near_tie`：reward 虽不完全相同，但未达到配置的离散度或有效分辨率；
- `redundant`：reward 信号有效，但近重复门拒绝该组。

外部前置条件失败具有最高优先级，统一标记为 `external`；各 Gate 的底层原因仍保留在
`decision.reasons` 中。

## 预算感知补采

`SamplingBudgetController` 使用有效组率的指数移动平均估算下一轮需要补采多少组，
同时支持额外组数、补采轮数、生成样本数和生成 token 数硬上限。

```python
from twinkle.advantage.group_admission import (
    SamplingBudgetConfig,
    SamplingBudgetController,
    SamplingBudgetState,
)

controller = SamplingBudgetController(
    SamplingBudgetConfig(
        max_extra_groups=16,
        max_extra_groups_per_round=4,
        max_resample_rounds=2,
        max_total_samples=128,
    )
)
state = SamplingBudgetState()

# decisions 中每个元素对应本轮的一个完整 GRPO 组。
controller.observe_round(
    state,
    decisions,
    num_generations=4,
    generated_tokens=round_output_tokens,
)
plan = controller.plan_resampling(
    state,
    target_groups=8,
    num_generations=4,
)
sample_more_complete_groups(plan.extra_groups)
```

补采必须使用与首批候选相同的 rollout policy 快照。控制器只负责规划补采量，
调用方仍需检查 policy version 和 staleness。

从 GSM8K rollout、整组准入、有界补采、GRPO advantage 到优化器更新的
端到端流程，可参考
[`cookbook/rl/grpo/group_admission.py`](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/grpo/group_admission.py)
。该示例基于官方 `short_math_grpo.py` 配置，首轮与补采组共用同一个
rollout policy 快照：

```bash
sh cookbook/rl/grpo/group_admission.sh --max-steps 2
```

真实 rollout 中 exact-dead 和 near-tie 的出现具有随机性；相关分支的确定性
覆盖仍由 `tests/advantage/test_group_admission.py` 保证。

可以使用 `group_admission_metrics(decisions)` 记录有效组率、reward/近重复条件的独立
拒绝数量，以及互斥的 `exact_dead`、`near_tie`、`redundant` 数量。比例由调用方
基于原始计数计算，以便分布式合并时使用统一分母。

## Challenger 接入

`twinkle_agentic.challenger.Challenger` 已会在已完成单元没有产生可训练组时持续补充新单元。
将策略传给 `AgenticChallenger`，即可在现有的 group-atomic 边界执行分辨率感知的准入：

```python
challenger = AgenticChallenger(
    backend,
    envs=envs,
    group_admission_policy=policy,
)
```

未传策略时，原有的同分组过滤保持不变。启用可选近重复门后，Challenger 使用每条
轨迹最后一个 assistant 文本作为表示。现有连续补充和 `max_empty_rounds` 语义均不
改变。需要样本数或 token 硬预算的其他采样循环仍可使用独立的
`SamplingBudgetController`；Challenger 不会隐式放宽 Gate。

## 非目标

预算耗尽时不会隐式放宽任何 Gate；调用方根据自身 batch 语义跳过任务或失败退出。
本组件也不修改 GRPO loss 和标准化 advantage 的计算。
