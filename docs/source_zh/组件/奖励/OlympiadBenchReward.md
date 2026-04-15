# OlympiadBench 奖励

用于评估 OlympiadBench 数学和物理竞赛问题的奖励函数族。

## OlympiadBenchAccuracyReward

评估答案正确性，支持 LaTeX 归一化、数值容差和部分匹配。

```python
from twinkle.reward import OlympiadBenchAccuracyReward

reward_fn = OlympiadBenchAccuracyReward()
rewards = reward_fn(generated_trajectories, ground_truth_trajectories)
# rewards: List[float], 1.0 表示正确, 0.0 表示错误
```

奖励函数的工作流程:
1. 从 `\boxed{...}` 中提取答案，支持嵌套大括号处理
2. 归一化预测和真实答案（LaTeX、单位、分数）
3. 通过归一化字符串匹配或带容差的数值比较进行判断

## OlympiadBenchFormatReward

验证模型输出的结构格式。

```python
from twinkle.reward import OlympiadBenchFormatReward

reward_fn = OlympiadBenchFormatReward()
rewards = reward_fn(trajectories, ground_truths)
# rewards: List[float], 基于格式质量的分数
```

评分标准:
- `\boxed{...}` 答案的存在性
- 答案位置（应出现在末尾附近）
- 答案的唯一性和一致性

## OlympiadBenchQualityReward

结合多个维度评估响应质量的复合奖励。

```python
from twinkle.reward import OlympiadBenchQualityReward

reward_fn = OlympiadBenchQualityReward()
rewards = reward_fn(trajectories, ground_truths)
```

质量维度:
- **推理结构**: 检测逐步推理模式
- **长度适当性**: 对过短或过长响应的平滑惩罚曲线
- **内容唯一性**: 惩罚响应中的重复内容

> 这些奖励可以单独使用或组合为复合奖励，用于竞赛级数学和物理问题的 GRPO 训练。
