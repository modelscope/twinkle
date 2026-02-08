# Sampler

Sampler (采样器) 是 Twinkle 中用于生成模型输出的组件,主要用于 RLHF 训练中的样本生成。采样器支持多种推理引擎,包括 vLLM 和原生 PyTorch。

## 基本接口

```python
class Sampler(ABC):
    
    @abstractmethod
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[SamplingParams] = None,
        adapter_name: str = '',
        *,
        num_samples: int = 1,
    ) -> SampleResponse:
        """对给定输入进行采样"""
        ...
    
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        """添加 LoRA 适配器"""
        ...
    
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """设置模板"""
        ...
```

采样器的核心方法是 `sample`,它接受输入数据并返回生成的样本。

## 可用的采样器

Twinkle 提供了两种采样器实现:

### vLLMSampler

vLLMSampler 使用 vLLM 引擎进行高效推理,支持高吞吐量的批量采样。

- 高性能: 使用 PagedAttention 和连续批处理
- LoRA 支持: 支持动态加载和切换 LoRA 适配器
- 多样本生成: 可以为每个 prompt 生成多个样本
- Tensor Parallel: 支持张量并行加速大模型推理

详见: [vLLMSampler](vLLMSampler.md)

### TorchSampler

TorchSampler 使用原生 PyTorch 和 transformers 进行推理,适合小规模采样或调试。

- 简单易用: 基于 transformers 的标准接口
- 灵活性高: 容易定制和扩展
- 内存占用小: 适合小规模采样

详见: [TorchSampler](TorchSampler.md)

## 如何选择

- **vLLMSampler**: 适合生产环境和大规模训练,需要高吞吐量
- **TorchSampler**: 适合调试、小规模实验或自定义需求

> 在 RLHF 训练中,采样器通常与 Actor 模型分离,使用不同的硬件资源,避免推理和训练相互干扰。
