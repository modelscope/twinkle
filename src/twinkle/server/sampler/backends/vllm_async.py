# Copyright (c) ModelScope Contributors. All rights reserved.
"""Server sampling concurrency over the stateless sampler.

The server's async backend is the plain VLLMSampler: the multi-turn and async-RL control lives on
the CLIENT (the streaming driver drives submit/pause/stop over this handle), while the server side
stays a stateless execution device. The wrapper exists only so the deployment registry can key on
its own class per sampler_type; the decorator keeps the actor's method-concurrency ceiling in one
place with the sampler-side default.
"""
from twinkle import remote_class
from twinkle.sampler.vllm_sampler.vllm_sampler import SAMPLER_MAX_CONCURRENCY, vLLMSampler


@remote_class(max_concurrency=SAMPLER_MAX_CONCURRENCY)
class ServerVLLMSampler(vLLMSampler):
    pass
