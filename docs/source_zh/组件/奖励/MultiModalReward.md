# 多模态奖励

用于评估多模态视觉问答（VQA）任务的奖励函数。

## MultiModalAccuracyReward

评估多模态 VQA 答案的正确性，支持回退到符号数学验证。

```python
from twinkle.reward import MultiModalAccuracyReward

reward_fn = MultiModalAccuracyReward()
rewards = reward_fn(generated_trajectories, ground_truth_trajectories)
# rewards: List[float], 1.0 表示正确, 0.0 表示错误
```

奖励函数的工作流程:
1. 从补全文本中提取模型的答案
2. 使用精确字符串匹配与真实答案比较
3. 当字符串匹配失败时回退到 `math_verify` 进行符号表达式比较

> 专为 CLEVR 等视觉推理任务设计，答案可能是数字、布尔值或短文本。
