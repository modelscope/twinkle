# TorchSampler

当前版本不提供 `TorchSampler`。旧的 `from twinkle.sampler import TorchSampler` 示例已不可用，服务端配置校验也会拒绝 `sampler_type: torch`。

采样请使用 [vLLMSampler](vLLMSampler.md)。服务端异步 RL 使用 `sampler_type: vllm_async`，参见 [客户端异步 GRPO](../../使用指引/服务端和客户端/Twinkle客户端.md#client-async-grpo)。
