# StateGraphAdvantage

`StateGraphAdvantage` 在 episode-level GRPO advantage 之上，增加基于显式状态图的
turn-level advantage。它依赖阶段 A 产生的 `turn_credits`，不根据聊天文本推断状态，
也不会增加额外 policy forward。

## 使用示例

```python
from twinkle.advantage import StateGraphAdvantage

advantage_fn = StateGraphAdvantage(
    beta_episode=1.0,
    beta_graph=1.0,
    gamma=0.1,
    min_state_support=2,
)

advantages = advantage_fn(
    rewards,
    trajectories,
    num_generations=4,
)
```

每条 trajectory 需要包含：

- `turn_credit_status.available=True`；
- `turn_credits` 中每个 turn 的 `pre_state_key`、`post_state_key`；
- 左闭右开的 `action_token_start` / `action_token_end`；
- 明确的 `policy_version`；
- 至少一个带 `terminal_success=True` 的状态边。

## 计算方式

同一 prompt group 内，状态键构成图节点，turn 构成有向边。成功终态的距离为 0，
算法从成功态反向计算最短距离，并按相同 `pre_state_key` 对动作做归一化：

```text
R_t^G = 10 * gamma ^ distance(post_state_t)
A_t^G = Normalize(R_t^G | same prompt group, same pre_state_key)
A_t   = beta_episode * A_t^E + beta_graph * A_t^G
```

`A_t^E` 每条 episode 只计算一次，然后广播到该 episode 的可训练 token。
graph advantage 只写入 turn 的 action span，不覆盖 observation、bridge 或 padding。

## 严格回退

以下情况不会部分解释 graph advantage，而是保留原 GRPO 数值：

- 缺失状态键或 terminal success；
- 同一轮工具结果的状态或成功标记冲突；
- action span、observation span 越界；
- policy version 缺失、冲突或类型非法；
- 没有成功终态；
- 状态图节点或边超过预算；
- turn credit 不可用或为空。

回退原因记录在 `advantage_fn.last_diagnostics.fallback_counts`。相同状态的动作
少于 `min_state_support` 时，该状态组保持零 graph advantage。

## Gate 0

确定性探针位于 `cookbook/rl/graphcredit/gate0.py`：

```bash
PYTHONPATH=src python cookbook/rl/graphcredit/gate0.py
```

探针检查状态覆盖率、同状态支撑、动作排序、无成功终态负例、回退完整性、
CPU 计算耗时以及额外 policy forward 数量。
