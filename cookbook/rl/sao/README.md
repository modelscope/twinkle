# Twinkle 同步 SAO：实现审查、算法原理与使用报告

本文档是当前 `add-sao-sync` 分支的完整交付说明。内容包括：论文对照结论、与最新 PPO 的兼容性、每个算法组件的原理、训练时序、代码文件映射、运行命令、日志解释、实验口径和已知限制。

## 1. 先给结论

当前代码正确实现了 SAO 的主要**算法组件**：

1. Single-Rollout：每个 prompt 只生成一条轨迹。
2. Direct Double-Sided Importance Sampling（DIS）：直接用 rollout engine 保存的 token log-prob 作为 behavior probability。
3. 严格双侧 trust region：ratio 超出区间的 token 完全不参与 policy gradient。
4. 独立 value critic。
5. Faster Value Update：每次 actor update 前执行 `K=2` 次 critic update。
6. Frozen-Attention critic。
7. Skip-Observation token-level GAE。
8. 长度自适应 policy lambda。
9. critic 使用独立的 `lambda=1` 和 10-step warmup。

当前入口是**同步正确性基线**，不是论文完整的异步系统。它仍然遵循：

```text
收集完整 rollout batch -> 停止采样 -> 更新 critic/actor -> 再采下一批
```

因此它可以验证 SAO 公式、value 训练和单 rollout 流程，但不能验证论文所强调的 rollout/learner 并行、trajectory 到达即训练、policy lag 或异步吞吐收益。

## 2. 为什么要在最新 PPO 上重新检查

最新 `main` 中的 PPO 已经明确了以下接口语义：

- policy loss 使用 `old_logps` 接收 vLLM rollout log-prob。
- response-only ragged logp 会依据 `labels != -100` 自动对齐。
- PPO policy loss 默认使用 `loss_agg_mode='token-mean'`。
- PPO epoch 改用通用的 `training.num_train_epochs`。
- value model、GAE、Ray GPU 资源检测已合入 main。

SAO 重新基于该 main 检查后：

- 仍通过 `old_logps` 向公共 loss 接口传入 rollout logp。
- 删除了多余的 `rollout_logps=` loss 参数，只保留框架真正消费的 `old_logps=`。
- SAO loss 继续采用 token-mean，与最新 PPO 明确的聚合口径一致。
- 没有修改 PPO 训练代码，也没有改变 `PPOLoss` 行为。
- SAO 的 advantage 和 value loss 保持独立类，不会替换普通 PPO GAE/value loss。

## 3. 基础概念

### 3.1 Policy、rollout policy 和 critic

- Policy/Actor：生成 action token，并通过梯度更新提高高 advantage action 的概率。
- Rollout policy：采样时真正位于 vLLM 中的 actor 权重。同步版每个 rollout batch 前更新一次。
- Critic/Value model：预测从当前 token 状态继续执行可以获得的未来 return。

完整异步 SAO 中 rollout policy 可能落后 learner 多个版本。当前同步版在每批采样前同步 actor，因此 policy lag 很小，主要用于算法验证。

### 3.2 Trajectory 与 action token

当前 GSM8K 示例的一条轨迹为：

```text
题目 prompt -> 模型 completion -> 格式/答案 reward
```

模型只应该学习自己生成的 completion token：

```text
labels == -100  -> prompt、padding 或不训练位置
labels != -100  -> action token
```

多轮 Agent 轨迹通常是：

```text
action_0 -> observation_0 -> action_1 -> observation_1 -> ...
```

observation 来自环境或工具，不是模型 action，所以不应该计算 policy gradient。

### 3.3 Reward、value、return 与 advantage

- `reward`：环境给出的真实反馈。
- `V(s_t)`：critic 对状态 `s_t` 后续累计收益的预测。
- `return_t`：训练 critic 的监督目标。
- `advantage_t`：action 的实际效果相对 critic 预期好多少。

直观上：

```text
advantage > 0 -> 提高该 action token 的概率
advantage < 0 -> 降低该 action token 的概率
```

当前 GSM8K reward 加在 completion 最后一个 action token，GAE 再把信号反向传播到更早的 action token。

## 4. DIS policy loss

### 4.1 为什么保存 rollout logp

生成 action `a_t` 时，vLLM 保存：

```text
rollout_logp_t = log pi_rollout(a_t | s_t)
```

训练时当前 actor 对同一 token 重新前向：

```text
current_logp_t = log pi_theta(a_t | s_t)
```

importance ratio 为：

```text
r_t = exp(current_logp_t - rollout_logp_t)
```

ratio 接近 1 表示当前 actor 与生成这条数据的 behavior policy 接近；偏离很大表示数据对于当前 actor 已经比较 stale/off-policy。

### 4.2 SAO 与 PPO clipping 的区别

PPO 通常根据 advantage 符号选择一侧进行 clipping。SAO 使用无条件双侧信任区间：

```text
trusted = 1 - epsilon_low < r_t < 1 + epsilon_high
```

校准函数：

```text
f(r_t) = r_t, trusted
f(r_t) = 0,   otherwise
```

对应 token objective：

```text
f(r_t) * advantage_t * log pi_theta(a_t | s_t)
```

数学推理实验：

```text
epsilon_low  = 0.3
epsilon_high = 5.0
trust region = (0.7, 6.0)
```

Coding/SWE-Bench：

```text
epsilon_low  = 0.8
epsilon_high = 3.0
trust region = (0.2, 4.0)
```

边界是严格不等号；ratio 恰好位于边界也被屏蔽。`SAOMetric` 使用 `<=`/`>=` 统计 rejection，与 loss 的严格区间一致。

### 4.3 Importance weight 是否反向传播

当前默认 `detach_importance_weight=true`，将 ratio 当作 policy-gradient 权重，而不通过 ratio 自身再产生额外导数。论文没有明确说明 autograd 实现细节，所以这是显式工程选择，不应宣称是论文唯一实现。

可以做消融：

```bash
--no-detach-importance-weight
```

### 4.4 为什么强制原始采样分布

DIS 要求 rollout logp 与 learner current logp 描述同一种概率分布。温度、top-k、top-p 或 repetition penalty 会改变 sampler 实际行为分布，而 learner 默认计算原始模型分布。

当前脚本因此强制：

```text
temperature = 1.0
top_p = 1.0
top_k = -1
repetition_penalty = 1.0
```

若覆盖为其他值会直接报错，而不是静默计算错误 ratio。

## 5. Single-Rollout

SAO 设置：

```text
num_generations = 1
```

每个 prompt 只需要一条轨迹。GRPO 通常为一个 prompt 生成多条回答，用组内 reward 计算相对 advantage；SAO 使用 critic 提供 state-dependent baseline，所以不需要 prompt group。

论文对照实验中的总 trajectory batch 都是 128：

```text
SAO : 128 prompts x 1 rollout
GRPO: 16 prompts  x 8 rollouts
```

当前脚本若 `num_generations != 1` 会 fail fast。

## 6. Skip-Observation token-level GAE

### 6.1 普通 token GAE

TD residual：

```text
delta_t = reward_t + gamma * V(s_{t+1}) - V(s_t)
```

GAE：

```text
A_t = delta_t + gamma * lambda * A_{t+1}
```

### 6.2 为什么要跳过 observation

环境 observation 并非模型生成，直接在 observation token 上计算 value/advantage 会引入无意义的训练信号。当前实现先取所有 `action_mask=True` 的位置，然后只在这些位置建立 Bellman 链。

例如：

```text
action A -> observation tokens -> action B
```

计算 A 的 next value 时直接使用 B 的 value，跨过 observation：

```text
delta(A) = reward(A) + gamma * V(B) - V(A)
```

### 6.3 长度自适应 lambda

Actor 使用：

```text
lambda_policy = 1 - 1 / (alpha * action_length)
alpha = 1.5
```

实现按每条 trajectory 的有效 action token 数单独计算 lambda，并裁剪到 `[0,1]`。

Critic 使用固定：

```text
lambda_critic = 1.0
```

### 6.4 Terminal 与 truncated

- `terminated=True`：环境真的结束，末端 next value 为 0。
- `truncated=True`：因 token/turn 上限截断，必须提供 `bootstrap_value`。

当前单轮 vLLM completion 没有保存截断后的环境状态，因此同步 cookbook 会拒绝 `length/abort/error` response，不会把它们错误当作 terminal。

## 7. Critic 训练

### 7.1 Faster Value Update

每一次 actor optimizer step 前执行：

```text
K = 2 次 critic optimizer step
```

当前时序：

```text
1. critic 前向，保存 value 快照
2. 用快照计算 fixed returns
3. critic 对同一份 fixed returns 更新两次
4. 更新后的 critic 再次前向
5. 用新 value 计算 actor advantage
6. actor 更新一次
```

两次 critic update 共用 fixed returns，避免监督目标随着 critic 自己的每一步更新而漂移。

### 7.2 Frozen-Attention

`freeze_attention_for_value_training()` 在创建 optimizer 前遍历模型模块，定位 `self_attn`/`attn` 并设置：

```python
parameter.requires_grad = False
```

MLP/MoE、norm、embedding 和 scalar value head 保持其默认可训练状态。该实现对应论文的 Frozen-Attention 操作；它不是“只训练 value head”。如果没有找到 Attention 模块会报错，防止配置看似开启但实际上没有冻结。

### 7.3 Value loss

`SAOValueLoss` 是 action token 上的 MSE：

```text
mean((V_theta - fixed_return)^2 over action tokens)
```

它不使用 PPO value clipping，因为论文没有把 PPO clipped value objective 列为 SAO 组件。

### 7.4 Value 初始化的现实差距

论文强调扩大 value pretraining 数据，并用预训练结果初始化 value model。当前 cookbook 从同一个语言模型 backbone 创建 critic，但 scalar value head 是零初始化，没有提供论文规模的 value-pretraining checkpoint。

因此：

- 算法训练链路已实现。
- value cold-start 处理没有复现论文规模。
- 若要复现实验指标，需要先提供兼容的 pretrained value checkpoint 或增加独立 value pretraining 阶段。

## 8. 同步训练时序与 batch 语义

### 8.1 三种 batch 参数

- `batch_size`：一次 rollout 收集的 prompt/trajectory 数。
- `mini_batch_size`：每次 actor/critic optimizer step 消费的 trajectory 数。
- `micro_batch_size`：一次设备前向/反向的子批大小，用于梯度累积和显存控制。

例如：

```text
batch_size=128
mini_batch_size=128
micro_batch_size=1
```

表示生成 128 条轨迹，然后用 128 条共同完成一次 optimizer step；内部拆成 micro batch 以控制显存。

如果配置：

```text
batch_size=128
mini_batch_size=4
```

则一个 rollout batch 会产生 32 次 actor optimizer step，这不再是“global batch 128 的一次更新”。脚本会打印明确警告，并记录：

```text
train/actor_updates_per_rollout_batch
```

论文口径应设置 `batch_size == mini_batch_size == 128`。

### 8.2 每条数据是否重复训练

当前同步 SAO 不设置 PPO epoch，indices 只遍历一遍：

```text
一条 rollout trajectory -> 只参与一次 actor update
```

同一 trajectory 会用于两次 critic update，这是论文 K=2 的 TTUR 设计，不是 policy replay。

## 9. 源码逐段教程：具体添加了什么

这一节不再只讲概念，而是按代码真实调用顺序解释实现。建议一边打开源码，一边对照阅读。

### 9.1 注册新组件：为什么字符串 `SAOLoss` 能找到类

训练脚本没有直接实例化 loss，而是写：

```python
policy.set_loss('SAOLoss', ...)
critic.set_loss('SAOValueLoss')
```

因此首先要把新增类暴露给 Twinkle 的组件加载机制。`src/twinkle/loss/__init__.py` 新增：

```python
from .sao import SAOLoss
from .value import PPOValueLoss, SAOValueLoss

torch_loss_mapping = {
    ...
    'sao': SAOLoss,
    'sao_value': SAOValueLoss,
}
```

这里同时支持两种查找方式：类名 `SAOLoss` 和 mapping 名 `sao`。Advantage 与 metric 也在各自 `__init__.py` 导出：

```python
from .sao_gae import SAOGAEAdvantage
from .grpo import ..., SAOMetric
```

如果忘记这些导出，文件虽然存在，但 cookbook 中的 import 或字符串组件查找会失败。

### 9.2 `SAOLoss` 如何复用最新 PPO/GRPO 的公共对齐代码

文件：`src/twinkle/loss/sao.py`

```python
class SAOLoss(GRPOLoss):
```

SAO 没有重新实现完整的 `__call__`，而是继承 `GRPOLoss` 的公共流程：

```text
读取 labels
-> 构造 loss_mask = labels != -100
-> 取得当前模型 logps
-> 把 ragged old_logps 对齐到 action token
-> 对齐 advantages
-> 计算 log_ratio 与 ratio
-> 调用子类 _compute_per_token_loss
-> 调用子类 _aggregate_loss
```

这非常关键，因为 vLLM 返回的是 response-only ragged 数据：

```python
old_logps = [
    [logp_0, logp_1, ...],       # 第一条 completion
    [logp_0, logp_1, logp_2],   # 第二条 completion
]
```

而 learner 的 labels 通常是 padding 后的完整 prompt+response：

```text
labels.shape = [batch, full_sequence_length]
```

继承公共实现后，`_pad_and_align_to_batch` 会把 response logp scatter 到 `labels != -100` 的位置。SAO 只需要改“每个 token 的 loss 怎么算”。

构造函数：

```python
def __init__(
    self,
    epsilon_low: float = 0.3,
    epsilon_high: float = 5.0,
    detach_importance_weight: bool = True,
    **kwargs,
):
    if not 0.0 <= epsilon_low < 1.0:
        raise ValueError(...)
    if epsilon_high < 0.0:
        raise ValueError(...)
    super().__init__(epsilon=epsilon_low, epsilon_high=epsilon_high, **kwargs)
```

这里把 `epsilon_low` 同时传给父类的 `epsilon`，是因为父类 metric/loss 基础设施使用 `epsilon` 表示下侧范围，而 SAO 自己保留更明确的 `epsilon_low` 名称。

核心 token loss：

```python
trusted = (
    (ratio > 1.0 - self.epsilon_low)
    & (ratio < 1.0 + self.epsilon_high)
)

weight = torch.where(
    trusted,
    ratio,
    torch.zeros_like(ratio),
)

if self.detach_importance_weight:
    weight = weight.detach()

return -weight * advantages.detach() * per_token_logps.float()
```

逐行解释：

1. `trusted` 是 `[batch, seq_len]` 的 bool tensor。
2. 两侧都使用严格不等号，对应论文公式。
3. 越界 token 的 `weight=0`，不是把 ratio 改成边界值。
4. `advantages.detach()` 保证 policy loss 不会反向更新 critic 计算图。
5. 默认 `weight.detach()` 把 ratio 当作 importance weight。
6. 最前面的负号是因为框架执行梯度下降，而论文写的是最大化 objective。

聚合：

```python
def _aggregate_loss(self, per_token_loss, loss_mask, **kwargs):
    mask = loss_mask.to(per_token_loss.dtype)
    return (
        (per_token_loss * mask).sum()
        / mask.sum().clamp(min=1.0)
    )
```

分母是所有 action token 数，包括被 DIS 拒绝的 token。被拒绝 token 对分子贡献 0，但不会因为拒绝数量多而把剩余 token 的平均权重放大。

### 9.3 用一个具体数字走完 `SAOLoss`

假设四个 token 的 ratio 是：

```text
[0.7, 0.8, 2.0, 6.0]
```

配置：

```text
epsilon_low=0.3
epsilon_high=5.0
trusted interval=(0.7, 6.0)
```

得到：

```text
trusted = [False, True, True, False]
weight  = [0.0, 0.8, 2.0, 0.0]
```

若 advantage 都为 1：

```text
per_token_loss = [0, -0.8*logp_1, -2.0*logp_2, 0]
```

最后仍除以 4，而不是除以 2。这一行为由 `test_sao_ragged_alignment_and_token_mean_denominator` 固定下来。

### 9.4 `SAOGAEAdvantage` 的输入张量是什么

文件：`src/twinkle/advantage/sao_gae.py`

接口：

```python
advantages, returns = gae(
    rewards,
    values,
    action_masks=action_masks,
    terminated=terminated,
    truncated=truncated,
    bootstrap_values=bootstrap_values,
    effective_lengths=effective_lengths,
)
```

典型形状：

```text
rewards.shape      = [B, T]
values.shape       = [B, T]
action_masks.shape = [B, T]
terminated.shape   = [B]
truncated.shape    = [B]
advantages.shape   = [B, T]
returns.shape      = [B, T]
```

同步 GSM8K cookbook 在进入 GAE 前已经把 prompt 去掉，因此这里的 `T` 是本 batch padding 后的最大 action 长度。通用实现仍允许 mask 中夹 observation。

输入先统一成 tensor：

```python
rewards = torch.as_tensor(rewards, dtype=torch.float32)
values = torch.as_tensor(values, dtype=torch.float32, device=rewards.device)
action_masks = torch.as_tensor(
    action_masks,
    dtype=torch.bool,
    device=rewards.device,
)
```

然后严格验证三者形状相同，防止 reward/value/action 发生 token shift。

### 9.5 Skip-Observation 的核心代码

每条轨迹先取得 action 位置：

```python
positions = action_masks[batch_idx].nonzero(as_tuple=True)[0]
```

例如：

```text
action_masks = [True, False, False, True]
positions    = [0, 3]
```

反向循环不是遍历 `T-1 ... 0`，而是遍历 `positions`：

```python
for index in range(positions.numel() - 1, -1, -1):
    position = positions[index]

    if index + 1 < positions.numel():
        next_value = values[batch_idx, positions[index + 1]].detach()
    elif truncated:
        next_value = bootstrap_value
    else:
        next_value = 0

    delta = (
        rewards[batch_idx, position]
        + gamma * next_value
        - values[batch_idx, position]
    )

    next_advantage = (
        delta
        + gamma * lambda_value * next_advantage
    )
```

当 `position=0` 时，下一个 action value 直接取位置 3，中间 observation 位置 1、2 完全不进入 Bellman 递推。

### 9.6 Adaptive lambda 的实现

```python
lambda_value = self.gae_lambda

if lambda_value is None:
    lambda_value = (
        1.0
        - 1.0 / (self.alpha * float(lengths[batch_idx]))
    )
    lambda_value = min(1.0, max(0.0, lambda_value))
```

含义：

- Actor 创建 GAE 时传 `gae_lambda=None`，启用 adaptive lambda。
- Critic 创建 GAE 时传 `gae_lambda=1.0`，使用固定 lambda。
- `lengths[batch_idx]` 是每条轨迹自己的 action token 数，不是 batch 最大 padding 长度。

返回值：

```python
returns = advantages + values.detach()
returns = returns.masked_fill(~action_masks, 0.0)
```

advantage 标准化也只统计：

```python
valid = advantages[action_masks]
```

不会让 prompt、observation 或 padding 的 0 参与均值/方差。

### 9.7 `SAOValueLoss` 如何训练 critic

文件：`src/twinkle/loss/value.py`

```python
class SAOValueLoss(Loss):
    require_logps = False
    require_values = True
```

这两个标志会通知 Transformers forward：critic 不需要词表 logp，只需要 scalar value。

构造 mask：

```python
labels = torch.as_tensor(inputs.get('labels'))
mask = labels != self.ignore_index
```

对齐 ragged returns：

```python
returns = self._aligner._pad_and_align_to_batch(
    returns,
    mask,
    values.device,
    values.dtype,
)
```

损失：

```python
loss = (
    (values.float() - returns.detach().float()).square()
    * mask_f
).sum() / mask_f.sum().clamp(min=1.0)
```

与 `PPOValueLoss` 相比，它没有：

```text
old_values
clipped_values
max(unclipped_loss, clipped_loss)
```

因为当前 SAO 实现按照论文 value 训练描述使用普通 masked MSE。

### 9.8 Value model 本身增加了什么

文件：`src/twinkle/model/transformers/value_model.py`

PPO 已经提供了 `TransformersValueModel`：它把语言模型输出 head 替换成 scalar value head。

SAO 在这个公共 value model 上新增两个远程方法。

冻结 Attention：

```python
@remote_function(dispatch='all', collect='first', lazy_collect=False)
def freeze_attention_for_value_training(self):
    model = self.strategy.unwrap_model(self.model)
    attention_modules = []

    for module in model.modules():
        for attribute in ('self_attn', 'attn'):
            attention = getattr(module, attribute, None)
            if isinstance(attention, nn.Module):
                attention_modules.append(attention)

    if not attention_modules:
        raise ValueError(...)

    for attention in attention_modules:
        for parameter in attention.parameters():
            parameter.requires_grad = False
```

为什么使用 `remote_function(dispatch='all')`：critic 是 Ray/FSDP 多 worker 模型，每个 critic worker 都必须冻结本地 shard 对应的参数，不能只改 driver 上的代理对象。

为什么必须在 `set_optimizer` 前调用：optimizer 只应该收集仍然 `requires_grad=True` 的参数。

训练脚本顺序正是：

```python
critic = TransformersValueModel(...)

critic.freeze_attention_for_value_training()
critic.trainable_parameter_summary()

critic.set_optimizer('AdamW', lr=CRITIC_LR)
```

参数统计：

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

日志可以据此确认冻结确实生效。

### 9.9 `SAOMetric` 为什么不能直接使用 PPO clip metric

PPO 的 clip 统计通常依赖 advantage 符号。例如正 advantage 只关心 ratio 是否过高。SAO 的 trust region 与 advantage 符号无关，只要越过任一侧就拒绝。

因此新增：

```python
class SAOMetric(GRPOMetric):
    def _accumulate_clip(self, log_ratio, advantages, mask, mask_f):
        ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
        is_low = ratio <= 1 - self.epsilon
        is_high = ratio >= 1 + self.epsilon_high

        self.sum_clip_low += (is_low.float() * mask_f).sum()
        self.sum_clip_high += (is_high.float() * mask_f).sum()
```

这里把 `log_ratio` 限制到 `[-20,20]` 只用于 metric 数值稳定，不会改变实际 loss。

### 9.10 CLI 具体增加了哪些参数

`src/twinkle/cli/cli.py` 的 `LossArgs` 新增：

```python
epsilon_low: float = 0.3
detach_importance_weight: bool = True
```

对应命令：

```bash
--epsilon-low 0.3
--detach-importance-weight
--no-detach-importance-weight
```

`RLArgs` 新增：

```python
critic_updates_per_actor_update: int = 2
sao_alpha: float = 1.5
sao_policy_lambda_adaptive: bool = True
sao_critic_lambda: float = 1.0
freeze_critic_attention: bool = True
```

对应：

```bash
--critic-updates-per-actor-update 2
--sao-alpha 1.5
--sao-policy-lambda-adaptive
--no-sao-policy-lambda-adaptive
--sao-critic-lambda 1.0
--freeze-critic-attention
--no-freeze-critic-attention
```

当关闭 adaptive lambda 时，Actor 回退到公共参数：

```bash
--no-sao-policy-lambda-adaptive --gae-lambda 0.95
```

### 9.11 训练脚本从 rollout 中保存了什么

文件：`cookbook/rl/sao/sao_sync.py`

每个 rollout batch 前：

```python
checkpoint_manager.sync_weights(merge_and_sync=False)
sampler.reset_prefix_cache()
samples = sampler.sample(prompts, sampling_params)
```

这保证 sampler 使用刚发布的 actor LoRA 权重，并清理可能关联旧权重的 prefix cache。

解析 response：

```python
trajectories = []
rollout_logps = []
lengths = []

for response in samples:
    for sequence in response.sequences:
        trajectories.append(sequence.new_input_feature)
        rollout_logps.append([
            entry[0][1]
            for entry in sequence.logprobs
        ])
        lengths.append(len(sequence.tokens))
```

这里与 PPO cookbook 一致，不根据 `stop_reason` 过滤轨迹。达到 `max_tokens` 的序列也进入
当前 batch，并把采样边界之后的 value 按 0 处理。

`entry[0][1]` 是当前被采样 token 的 log-prob。进入列表前新增了三方检查：

```text
len(sequence.tokens)
== len(sequence.logprobs)
== count(labels != -100)
```

任何 token shift 都会直接报错。

### 9.12 Reward 如何变成 token reward

环境先返回 sequence reward：

```python
accuracy = GSM8KAccuracyReward()(trajectories)
formatting = GSM8KFormatReward()(trajectories)
rewards = [a + f for a, f in zip(accuracy, formatting)]
```

然后使用 PPO 已有公共工具：

```python
token_rewards = GAEAdvantage.build_token_rewards(
    rewards,
    lengths,
)
```

例如 completion 长度为 4、最终 reward 为 1：

```text
sequence reward = 1
token rewards   = [0, 0, 0, 1]
```

同步 SAO 没有额外 reference-model KL reward，因此这里不传 `old_logps/ref_logps/kl_coef`。

### 9.13 为什么先算 fixed returns

第一次 critic 前向：

```python
initial = critic.forward_only(inputs=trajectories)
initial_values = response_rows(initial['values'], trajectories)
```

`response_rows` 根据 labels mask 从完整序列 value 中只提取 action value，并检查 value 序列长度。

Critic GAE：

```python
_, fixed_returns = critic_gae(
    padded_rewards,
    pad_rows(initial_values, lengths),
    action_masks=masks,
    terminated=terminated,
    truncated=truncated,
    effective_lengths=lengths,
)
```

这里丢弃 critic advantage，只保存 returns。名字叫 `fixed_returns` 是为了强调：后面的两次 critic step 不能重新计算目标。

### 9.14 K=2 critic update 是怎么写的

```python
for _ in range(CRITIC_UPDATES):
    critic.forward_backward(
        inputs=mb_inputs,
        returns=mb_returns,
        micro_batch_size=MICRO_BATCH_SIZE,
    )
    critic.clip_grad_and_step()
```

当 `CRITIC_UPDATES=2`：

```text
forward/backward on fixed returns
-> clip critic grad + optimizer step
-> forward/backward on same fixed returns
-> clip critic grad + optimizer step
```

同一条数据用于两次 critic step 是论文 TTUR；它不代表 actor 也训练两次。

### 9.15 更新后的 critic 如何服务 actor

两次 critic step 后重新前向：

```python
updated = critic.forward_only(inputs=mb_inputs)
new_values = response_rows(updated['values'], mb_inputs)
```

再用 Actor GAE：

```python
advantages, _ = actor_gae(
    pad_rows(mb_rewards, mb_lengths),
    pad_rows(new_values, mb_lengths),
    action_masks=mb_masks,
    terminated=[True] * len(chosen),
    truncated=[False] * len(chosen),
    effective_lengths=mb_lengths,
)
```

区别：

```text
critic_gae: gae_lambda=1.0, normalize=False
actor_gae : adaptive lambda, normalize=CLI 配置
```

### 9.16 Actor update 如何连接到 `SAOLoss`

```python
policy.forward_backward(
    inputs=mb_inputs,
    old_logps=[rollout_logps[i] for i in chosen],
    advantages=mb_advantages,
    micro_batch_size=MICRO_BATCH_SIZE,
)
policy.clip_grad_and_step()
```

调用链为：

```text
TransformersModel.forward_backward
-> 当前 actor 前向，生成 current logps
-> SAOLoss.__call__
-> 对齐 old_logps/advantages
-> exp(current-old)
-> SAOLoss._compute_per_token_loss
-> backward
-> clip_grad_and_step
```

这里没有 PPO epoch 外层循环，所以每条 trajectory 对 actor 只使用一次。

### 9.17 完整伪代码

把实现压缩后就是：

```python
for rollout_batch in dataloader:
    sync_actor_to_vllm()

    trajectories, rollout_logps = rollout_once_per_prompt()
    rewards = environment_reward(trajectories)

    old_values = critic.forward(trajectories)
    fixed_returns = critic_gae(rewards, old_values)

    for mini_batch in rollout_batch:
        for _ in range(2):
            critic.train(mini_batch, fixed_returns)

        new_values = critic.forward(mini_batch)
        advantages = actor_gae(rewards, new_values)

        actor.train(
            mini_batch,
            old_logps=rollout_logps,
            advantages=advantages,
            loss=SAOLoss,
        )
```

### 9.18 测试代码具体证明了什么

`tests/loss/test_sao.py` 人工构造 ratio：

```python
logps = torch.tensor([
    [math.log(ratio) for ratio in ratios]
], requires_grad=True)
old_logps = torch.zeros_like(logps)
```

因为：

```text
exp(log(ratio) - 0) = ratio
```

所以可以精确测试 0.7、0.7001、5.999、6.0 的边界梯度。

`tests/advantage/test_sao_gae.py` 用：

```python
action_masks = [[True, False, True]]
```

并故意在 observation 位置放入很大的 reward/value：

```text
reward observation = 99
value observation  = 42
```

最终结果完全不受 99/42 影响，从而证明 GAE 确实跳过 observation。

Value model 测试确认：

```text
Attention requires_grad=False
MLP requires_grad=True
value head requires_grad=True
```

## 10. 代码文件映射

| 文件 | 职责 |
|---|---|
| `src/twinkle/loss/sao.py` | DIS trust mask、importance-weighted actor loss、token mean |
| `src/twinkle/advantage/sao_gae.py` | Skip-Observation GAE、adaptive lambda、terminal/truncated |
| `src/twinkle/loss/value.py` | `SAOValueLoss` masked MSE |
| `src/twinkle/model/transformers/value_model.py` | Frozen-Attention 与参数统计 |
| `src/twinkle/metric/grpo.py` | `SAOMetric` 双侧 rejection、ratio、KL |
| `src/twinkle/cli/cli.py` | SAO loss/GAE/critic 参数 |
| `cookbook/rl/sao/sao_sync.py` | 同步训练主流程 |
| `cookbook/rl/sao/sao_sync.sh` | 默认启动配置 |
| `tests/loss/test_sao.py` | DIS 边界、梯度、ragged 对齐、value loss |
| `tests/advantage/test_sao_gae.py` | observation skip、bootstrap、batch 隔离、adaptive lambda |

## 11. 论文参数对照

### 11.1 论文明确披露

| 参数 | 论文值 | 当前支持 |
|---|---:|---|
| reasoning backbone | Qwen3-30B-A3B-Thinking-2507 | 可通过 `--model-id` 替换 |
| global batch | 128 | `--batch-size 128 --mini-batch-size 128` |
| group size | 1 | 强制 `--num-generations 1` |
| max length/context | 128K | `--max-tokens 131072`，硬件需支持 |
| actor LR | 1e-6 | 默认脚本已设置 |
| critic LR | 5e-6 | 默认脚本已设置 |
| reasoning epsilon | 0.3 / 5.0 | 默认脚本已设置 |
| coding epsilon | 0.8 / 3.0 | 可覆盖 |
| policy lambda | `1 - 1/(alpha*l)` | 已实现 |
| alpha | 1.5 | 已实现 |
| critic lambda | 1.0 | 已实现 |
| critic warmup | 10 steps | 已配置 |
| critic updates K | 2 | 已配置 |
| Frozen-Attention | 开启 | 默认开启 |
| evaluation sampling | top-p=1, temperature=1 | 当前训练为 ratio 正确性也强制该分布 |

### 11.2 当前 cookbook 的工程默认

以下不是论文声明：

- 默认模型 `Qwen3.5-4B`。
- 数据集 GSM8K。
- 4 policy + 4 critic + 4 sampler GPU。
- LoRA `r=32, alpha=64, dropout=0.05`。
- actor 使用 cosine scheduler。
- critic warmup 后使用 Twinkle `LinearWarmupScheduler` 的线性衰减。
- 默认 smoke-friendly batch=4、max_tokens=1024、max_steps=200。

所以直接执行默认脚本是工程验证，不是论文复现。

## 12. 环境检查

建议在 WSL 打开仓库：

```bash
cd /mnt/c/Users/xxyyrr/Desktop/上班/twinkle
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

确认 GPU 与 Ray：

```bash
nvidia-smi
python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

Ray 输出必须包含足够的 `GPU` 资源。三组 GPU 数之和为 `nproc_per_node`：

```text
NUM_GPUS = model_gpus + critic_model_gpus + sampler_gpus
```

## 13. 运行测试

```bash
pytest \
  tests/loss/test_sao.py \
  tests/advantage/test_sao_gae.py \
  tests/model/test_value_model.py \
  tests/loss/test_ppo.py \
  tests/advantage/test_gae.py \
  tests/cli/test_cli.py -v
```

测试覆盖：

- ratio 信任区间严格边界。
- 越界 token 零梯度。
- 默认 detached ratio 梯度。
- ragged rollout logp 与 labels 对齐。
- token-mean 分母包含被拒绝 token。
- masked value MSE。
- GAE 跨 observation。
- terminal/truncated bootstrap。
- batch 中 trajectory 不串联。
- 每条 trajectory 独立 adaptive lambda。
- Frozen-Attention 参数状态。
- PPO/普通 GAE 回归。

CPU 单测不启动 vLLM/Ray 多 GPU，不能替代训练机 smoke test。

## 14. 小规模 smoke test

```bash
cd cookbook/rl/sao
sh sao_sync.sh \
  --model-gpus 1 \
  --critic-model-gpus 1 \
  --sampler-gpus 1 \
  --batch-size 2 \
  --mini-batch-size 2 \
  --micro-batch-size 1 \
  --max-tokens 128 \
  --max-steps 2 \
  --save-steps 1
```

这要求三个独立模型副本：policy、critic、vLLM sampler。三张 GPU 能否容纳默认 4B 模型取决于显存、FSDP 和 vLLM 配置，不能只按 LoRA 参数量估算。

## 15. 默认同步训练

```bash
cd cookbook/rl/sao
sh sao_sync.sh
```

默认值适合先检查链路：

```text
model = Qwen3.5-4B
rollout batch = 4
actor optimizer batch = 4
max new tokens = 1024
actor steps = 200
```

## 16. 论文口径配置示例

### 16.1 推理任务

```bash
sh sao_sync.sh \
  --model-id ms://Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --batch-size 128 \
  --mini-batch-size 128 \
  --micro-batch-size 1 \
  --max-tokens 131072 \
  --lr 1e-6 \
  --critic-learning-rate 5e-6 \
  --epsilon-low 0.3 \
  --epsilon-high 5.0 \
  --critic-updates-per-actor-update 2 \
  --sao-alpha 1.5 \
  --sao-critic-lambda 1.0
```

必须根据 30B MoE 模型实际显存重新设置三组 GPU 数。GSM8K 仍不是论文 TIR Agent 数据，因此这个命令只对齐模型和公开超参数，没有对齐完整任务环境。

### 16.2 Coding 信任区间

```bash
sh sao_sync.sh --epsilon-low 0.8 --epsilon-high 3.0
```

要复现 SWE-Bench，还需 OpenHands scaffold、最多 300 turns、128K context、对应 coding reward 和评测器；当前 GSM8K cookbook 不包含这些组件。

## 17. 日志怎么读

### 17.1 Reward

- completion length
- total reward
- format reward
- accuracy reward

### 17.2 Policy

- `train/mean_old_logp`
- `train/mean_new_logp`
- `train/logp_diff_mean`
- `train/approx_kl`
- `train/token_ratio_max`
- `train/clip_ratio_low/high`

SAO 中 clip ratio 实际表示被 trust region 拒绝的 token 比例，而不是 PPO 把 ratio 截到边界后的比例。

### 17.3 Critic

critic metric 现在以 `critic/` 前缀加入日志，可用于观察 value loss 和梯度趋势。论文报告 Frozen-Attention 能降低 critic gradient norm，因此正式实验应同时记录 critic grad norm。

### 17.4 训练时序

- `train/critic_updates_per_actor_update`
- `train/actor_updates_per_rollout_batch`

论文式全局 batch 128 应看到后者为 1。

## 18. 常见问题

### 18.1 ratio 一开始就远离 1

依次检查：

1. vLLM 权重是否在 rollout 前同步。
2. rollout logp 数是否等于 action label 数。
3. temperature/top-k/top-p/repetition penalty 是否改变了行为分布。
4. template 是否对 labels 做了额外截断或 token shift。
5. LoRA adapter 是否在 sampler 中正确加载。

当前代码新增了 tokens/logprobs/action labels 三者相等检查，会在对齐错误时直接停止。

### 18.2 大量 rollout 被丢弃

与仓库 PPO cookbook 保持一致，当前同步 SAO 基线接收 sampler 返回的全部轨迹，包括
`stop_reason=length` 的回复，并把采样序列边界当作本次 GAE 的终点，即边界之后的 value
按 0 处理。这样 rollout batch 不会因为过滤 length 轨迹而变小，分布式 mini-batch 的切分
也与 PPO 一致。

这是同步示例采用的工程简化，不是严格的 truncated-state bootstrap。如果后续长轨迹中
`stop_reason=length` 占比较高，可以再利用 value model 在最后有效输入位置输出的边界 value
实现 bootstrap；当前论文没有明确描述它的截断处理，因此本基线暂不额外引入该行为。

### 18.3 critic loss 或 grad norm 很大

检查：

- reward 尺度。
- fixed returns 是否只在 action token 上训练。
- critic LR 是否为 5e-6。
- Frozen-Attention 是否真的找到模块。
- value model 是否缺少预训练导致 cold start。

### 18.4 显存不足

这是三个模型副本，不是一个 LoRA 模型。可以：

- 增加每组 GPU。
- 减小 micro batch。
- 先减小 max tokens 做 smoke test。
- 调低 vLLM `gpu_memory_utilization`。

不要通过把 mini batch 改小而仍声称 global optimizer batch=128；那会改变 optimizer step 数和实验口径。

## 19. Checkpoint

训练期间：

```text
sao-policy-checkpoint-<actor_step>
sao-critic-checkpoint-<actor_step>
```

完成后：

```text
sao-policy-final
sao-critic-final
```

policy 和 critic 必须成对保存和恢复，否则 critic 分布与 actor 训练进度不一致。

## 20. 推荐实验顺序

1. CPU 单测。
2. 128 token、2 step 的 GPU smoke test。
3. 默认 Qwen3.5-4B/GSM8K 同步实验。
4. 检查 ratio、rejection、critic loss、reward 与显存。
5. 切换论文模型和 global batch。
6. 接入 value-pretraining checkpoint。
7. 接入真实 TIR/OpenHands Agent 环境。
8. 同步版本正确后，再从最新 main 单独开发异步 actor-learner pipeline。

## 21. 当前实现边界

- 同步 barrier，不是完整异步 SAO。
- GSM8K 单轮示例没有 observation；Skip-Observation GAE 的实现由单测覆盖，但 cookbook 没有真实多轮工具轨迹。
- 非 terminal 截断轨迹被拒绝。
- 没有论文规模的 value pretraining。
- 默认模型、数据、资源和 scheduler 是工程选择。
- 没有在当前本地机器完成真实多 GPU/vLLM 训练。
- 因此当前结果应表述为“同步 SAO 算法基线实现并通过单元回归”，不应表述为“完整复现论文异步系统及指标”。
