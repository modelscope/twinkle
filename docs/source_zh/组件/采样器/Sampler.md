# Sampler

Sampler (采样器) 是 Twinkle 中用于生成模型输出的组件,主要用于 RLHF 训练中的样本生成。当前采样器实现使用 vLLM。

## vLLMSampler 采样接口

具体的 `vLLMSampler` 提供以下采样接口：

```python
class vLLMSampler:

    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        return_encoded: bool = False,
        use_base_model: bool = False,
    ) -> List[SampleResponse]:
        """对给定输入进行采样"""
        ...

    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """设置模板"""
        ...
```

核心方法 `sample` 返回 `SampleResponse` 列表。通过 `SamplingParams(num_samples=N)` 设置每个 prompt 的生成数量，每个响应的 `sequences` 包含生成结果。

正常候选的 `new_input_feature` 是完整轨迹，保留 messages、tools、user_data、原始媒体和数值编码字段。`SampledSequence` 继续携带 tokens、logprobs、decoded、stop_reason、routed_experts 和 sampling_masks，不再另有 `new_state`。`max_tokens=0` 仅对 prompt 打分，不追加 assistant 消息。

普通 HTTP 客户端显式启用 `project_request_fields=True`，响应标出未变的请求字段，客户端无需 tokenizer 即可恢复；历史消息未变时，响应只发送生成的消息，历史由客户端恢复。未协商投影的调用方保留原有完整历史响应。改变后的历史和新编码的数值字段仍须返回。显式 `token_only=True` 只返回 tokens 与统计信息；`HTTPSampler` 使用本地模板追加实际 tokens，不重新 tokenize 回复。服务端 DataPlane 保留完整轨迹，不受 HTTP 投影影响。这减少的是响应重复，每轮媒体上传并未去重。

## 可用的采样器

Twinkle 提供 `vLLMSampler` 实现:

### vLLMSampler

vLLMSampler 使用 vLLM 引擎进行高效推理,支持高吞吐量的批量采样。

- 高性能: 使用 PagedAttention 和连续批处理
- LoRA 支持: 支持动态加载和切换 LoRA 适配器
- 多样本生成: 可以为每个 prompt 生成多个样本
- Tensor Parallel: 支持张量并行加速大模型推理

详见: [vLLMSampler](vLLMSampler.md)

标准采样使用 `sampler_type: vllm`，{ref}`客户端流式 GRPO 示例 <client-async-grpo>` 也使用这一配置。采样和训练由客户端的 `StreamingDriver` 调度。`mock` 用于测试。旧的 [TorchSampler](TorchSampler.md) 示例已不可用。

> 在 RLHF 训练中,采样器通常与 Actor 模型分离,使用不同的硬件资源,避免推理和训练相互干扰。
