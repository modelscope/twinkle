# TorchSampler

The current release does not provide `TorchSampler`. The former `from twinkle.sampler import TorchSampler` example is unavailable, and `sampler_type: torch` is rejected by server configuration validation.

Use [vLLMSampler](vLLMSampler.md) for sampling. For server-side async RL, use `sampler_type: vllm_async`; see the [client-orchestrated async RL example](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/README.md).
