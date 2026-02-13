# Template

模板是用于将 Trajectory 转换为 InputFeature 的关键组件。

```python
class Template:

    def __init__(self,
                 model_id: str,
                 use_chat_template: bool = True,
                 max_length: Optional[int] = 8192,
                 truncation_strategy: Literal['raise', 'left', 'right', 'split'] = 'raise',
                 default_system: Optional[str] = None):
        ...

    def encode(self, trajectory: Trajectory, add_generation_prompt: bool = False) -> InputFeature:
        # 编码单条样本
        ...

    def batch_encode(self, trajectories: Union[Dict[str, Any], List[Trajectory]]) -> List[InputFeature]:
        # 批量编码样本
        ...

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        # 编码一条样本，并返回原样本
        # 一般用于在GRPO等RL算法中检查数据合理性
        ...

    def batch_check(self, trajectories: List[Trajectory]) -> List[Optional[Trajectory]]:
        # 批量检查样本
        ...

    def decode(self, token_ids: List[int], **kwargs) -> str:
        # 解码样本
        ...

    def batch_decode(self, token_ids: List[List[int]], **kwargs) -> List[str]:
        # 批量解码样本
        ...
```

- model_id: 包含tokenizer或者processor的模型id
- use_chat_template: 是否使用 chat_template。如果不使用，一般是预训练场景
- max_length: 单样本的最大长度
- truncation_strategy: 如果超过了最大长度，如何处理该样本
  - raise: 抛出异常。一般用于非常精确的数据集场景
  - left: 移除左边的 token，使其符合 max_length
  - right: 移除右边的 token，使其符合 max_length
  - default_system: 如果数据集没有 system，则使用默认 system

> Template 不支持使用函数来代替，因为其内部要支持的功能较多。如果需要编写新的 Template，请继承 `Template` 类。
> 一般来说，纯文本模型使用 Template 基类就足够了，在基类中我们使用了 tokenizer.apply_chat_template 来编码模型，对一般的纯文本模型是通用的。
