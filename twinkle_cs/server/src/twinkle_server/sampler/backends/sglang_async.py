# Copyright (c) ModelScope Contributors. All rights reserved.
"""Server sampling concurrency over the stateless sglang sampler.

Mirrors ``vllm_async``: the plain SGLangSampler under the server-side method-concurrency ceiling
(the engine class itself declares none, so this wrapper is where the async backend gets its
concurrency). The server stays a stateless execution device; async-RL control lives on the client.
"""
from twinkle import remote_class
from twinkle.sampler import SGLangSampler
from twinkle.sampler.vllm_sampler.vllm_sampler import SAMPLER_MAX_CONCURRENCY


@remote_class(max_concurrency=SAMPLER_MAX_CONCURRENCY)
class ServerSGLangSampler(SGLangSampler):
    pass
