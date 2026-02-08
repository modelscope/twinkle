# Advantage

Advantage (优势函数) 是强化学习中用于计算动作相对于平均水平的优势值的组件。在 RLHF 训练中,优势函数用于指导策略优化。

## 基本接口

```python
class Advantage:
    
    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none'] = 'group',
                 **kwargs) -> 'torch.Tensor':
        """
        计算优势值
        
        Args:
            rewards: 奖励值列表或张量
            num_generations: 每个 prompt 生成的样本数量
            scale: 归一化方式
                - 'group': 对每组样本进行归一化 (GRPO)
                - 'batch': 对整个 batch 进行归一化
                - 'none': 不进行归一化
        
        Returns:
            优势值张量
        """
        ...
```

## 可用的优势函数

Twinkle 提供了两种优势函数实现:

### GRPOAdvantage

GRPO (Group Relative Policy Optimization) 优势函数通过减去组内均值来计算优势。

- 简单高效,适合大多数场景
- 减少方差,提高训练稳定性
- 在组内进行相对比较

详见: [GRPOAdvantage](GRPOAdvantage.md)

### RLOOAdvantage

RLOO (Reinforcement Learning with Leave-One-Out) 优势函数使用留一法计算基线。

- 理论上更优,减少偏差
- 需要更多样本(建议 8 个以上)
- 更准确的反事实基线估计

详见: [RLOOAdvantage](RLOOAdvantage.md)

## 如何选择

- **GRPO**: 适合样本数量较少(4 个左右)的场景,计算效率高
- **RLOO**: 适合样本数量较多(8 个以上)的场景,理论效果更好

> 优势函数的选择对 RLHF 训练效果有重要影响。建议根据计算资源和样本数量选择合适的方法。
