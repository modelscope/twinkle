# Template

模板编码轨迹时保留完整的 dict 载荷。`InputFeature` 是模型输入用途的类型指引，不承担字段删减：messages、tools、user_data 和原始 images/videos/audios 与数值 tensor 一同保留，非模型字段由模型的输入处理阶段筛选。

`encode_prompt` 直接返回完整 dict。多模态轨迹分别保留训练 `input_ids` 与 serving `engine_input_ids`；纯文本轨迹缺少 `engine_input_ids` 时复用 `input_ids`。`append_completion` 追加实际生成的 tokens，并返回带结构化 assistant 消息的新轨迹，不重编码已采样历史。

```python
class Template:

    def __init__(self,
                 model_id: str,
                 use_chat_template: bool = True,
                 max_length: Optional[int] = None,
                 truncation_strategy: Literal['raise', 'left', 'right', 'split', 'delete'] = 'raise',
                 default_system: Optional[str] = None):
        ...

    def encode(self, sample=None, add_generation_prompt=None, *, intent=None,
               engine='vllm', return_length=False, **kwargs):
        ...

    def batch_encode(self, trajectories, add_generation_prompt=None, *, intent=None,
                     engine='vllm', return_length=False, **kwargs):
        ...

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        # 编码一条样本，并返回原样本
        # 一般用于在GRPO等RL算法中检查数据合理性
        ...

    def batch_check(self, trajectories: List[Trajectory]) -> List[Optional[Trajectory]]:
        # 批量检查样本
        ...

    def decode(self, token_ids, *, context=None, **kwargs):
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
  - split: 将超长样本切分为多个 max_length 的片段（不支持多模态、LazyDataset、IterablePackingDataset）
  - delete: 单条 encode 抛出 MaxLengthError；batch_encode 丢弃该样本
- default_system: 如果数据集没有 system，则使用默认 system

## 共享接口

`swift.dev.template.Template` 继承该基类，保留 swift 原有工厂和模型家族实现。默认或 False 使用 TRAIN，True 使用 INFER；两边共用显式编码意图。INFER/ENGINE/SCORE 不返回训练 labels；`encode_prompt(..., add_generation_prompt=False)` 使用 SCORE，允许以 user 结尾的前缀且不添加 assistant 开头；非聊天预训练允许显式 False。因果训练的 labels 只左移一次，末位填 -100；存在时 loss_scale/completion_mask 同步移位。非空 `user_data.key_rounds` 按 system 消息归一后、内部工具调用拆分前的 assistant 消息索引选择监督范围。Jinja 从完整渲染中记录所选消息的输出范围；原生 token 标注只在 IDs 与该渲染一致时使用 offsets。逐轮回退遇到不满足前缀一致性的编码会明确拒绝，不猜测边界。

list 批次返回平铺 list，列存批次返回等长列。单条 encode 在 split 下可以返回多条轨迹；encode_prompt 拒绝多条结果。left/right 保留受保护媒体 token，并同步切片序列字段；不能安全切分的多模态和配对样本拒绝 split。追加不能截断已记录 token。decode 不带 context 返回文本，带 context 返回共享的 DecodedResponse。

进程内组件的 set_template 接受模板实例、子类或注册名称。远程 client 只传注册名称和构造配置（swift 复用现有 TemplateSpec），类和实例不跨 HTTP 传递。远程使用前通过现有 swift worker 启动 hook 注册 SwiftTemplate；名称未注册时明确报错，不回退到原生模板。

> Template 不支持使用函数来代替，因为其内部要支持的功能较多。如果需要编写新的 Template，请继承 `Template` 类。
> 一般来说，纯文本模型使用 Template 基类就足够了，在基类中我们使用了 tokenizer.apply_chat_template 来编码模型，对一般的纯文本模型是通用的。

# 模板对应关系

目前模板关系较为简单：

- Template类：纯文本模型通用
- DeepseekV4Template类：DeepSeek V4 使用，重写了 chat template 编码逻辑，`encode_messages` 已内置在 twinkle 中
- Qwen3_5Template类：Qwen3.5多模态模型使用
