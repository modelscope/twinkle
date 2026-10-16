"""Native TransferQueue building blocks for async RL (full-param or single-LoRA).

The public names resolve LAZILY (PEP 562 ``__getattr__``). The control-plane primitives --
``RLContext`` / ``RLContextManager`` / the ``weight_sync`` strategies -- are dependency-light
dataclasses, whereas the runtime (``pipeline`` / ``workers`` / ``vllm_sampler_tq`` / ``data_plane`` /
``native_tq``) pulls in TransferQueue, vLLM and native FSDP. A consumer that only needs a primitive
(dev's ``grpo_async.StreamingGRPOLoop`` drives ``RLContextManager`` + ``build_weight_sync_strategy``
over its own sampler, never the TransferQueue runtime) must not be forced to import that whole stack,
so each name is imported from its defining submodule on first access instead of eagerly here. The
runtime modules themselves import each other by relative submodule path, so they are unaffected.
"""

from __future__ import annotations

import importlib

#: Maps each public name to the submodule that defines it, so ``__getattr__`` imports only what is
#: accessed, and only when. Kept next to ``__all__`` so the two cannot drift.
_EXPORTS = {
    'AdapterSnapshotSync': 'weight_sync',
    'AdvantageWorker': 'workers',
    'AsyncMultiLoraGRPOConfig': 'pipeline',
    'AsyncMultiLoraGRPOPipeline': 'pipeline',
    'BufferRecord': 'streaming_driver',
    'ContextGRPOGroupNSampler': 'native_tq',
    'ContextSchedulePolicy': 'scheduler',
    'ContextScheduler': 'scheduler',
    'ContextStatus': 'context_manager',
    'DriverStats': 'streaming_driver',
    'InPlaceWeightSync': 'weight_sync',
    'LoraContext': 'types',
    'LoraContextManager': 'context_manager',
    'PartitionAdmission': 'types',
    'PreparedPartition': 'types',
    'PromptGroup': 'types',
    'RLContext': 'types',
    'RLContextManager': 'context_manager',
    'RolloutPolicy': 'types',
    'RolloutWorker': 'workers',
    'ScheduleCandidate': 'scheduler',
    'SchedulerConfig': 'scheduler',
    'StreamingDriver': 'streaming_driver',
    'TQDataPlane': 'data_plane',
    'TrainerWorker': 'workers',
    'VLLMSamplerTQ': 'vllm_sampler_tq',
    'WeightSyncStrategy': 'weight_sync',
    'build_weight_sync_strategy': 'weight_sync',
    'create_cpu_actor': 'pipeline',
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Import and cache ``name`` from its defining submodule on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(f'.{module_name}', __name__), name)
        globals()[name] = value  # cache, so __getattr__ is not hit again for this name
        return value
    # Not a re-exported name: fall back to importing a submodule of the same name (e.g. the runtime's
    # ``async_rl.pipeline``), so ``from twinkle_agentic.async_rl import <submodule>`` keeps working.
    try:
        module = importlib.import_module(f'.{name}', __name__)
    except ImportError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc
    globals()[name] = module
    return module


def __dir__():
    return sorted(__all__)
