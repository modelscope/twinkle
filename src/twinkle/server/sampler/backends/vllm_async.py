# Copyright (c) ModelScope Contributors. All rights reserved.
"""Server sampling concurrency without changing native async RL defaults."""
from twinkle import remote_class
from twinkle.sampler.vllm_sampler.vllm_sampler import SAMPLER_MAX_CONCURRENCY
from twinkle_agentic.async_rl.vllm_sampler_tq import VLLMSamplerTQ


@remote_class(max_concurrency=SAMPLER_MAX_CONCURRENCY)
class ServerVLLMSampler(VLLMSamplerTQ):
    pass
