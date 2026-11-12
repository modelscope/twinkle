# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from twinkle.utils import get_logger

logger = get_logger()

_GIB = 1024**3
_ALLOWED_OVERRIDE_KEYS = frozenset({
    'trainer_model_parallel_width',
    'sampler_model_parallel_width',
    'sampler_replicas',
    'shared_sampler',
    'flex_gpus',
    'max_num_batched_tokens',
    'max_num_seqs',
})


@dataclass(frozen=True)
class RolePlacement:
    """Immutable physical placement of one complete model role."""

    role: str
    model_parallel_width: int
    replica_count: int
    ranks: Tuple[int, ...]
    gpus_per_worker: int
    shared: bool = False

    def __post_init__(self) -> None:
        if self.model_parallel_width < 1 or self.replica_count < 1 or self.gpus_per_worker < 1:
            raise ValueError(f'{self.role}: widths and replica_count must be positive.')
        expected = self.gpus_per_worker * self.replica_count
        if len(self.ranks) != expected:
            raise ValueError(f'{self.role}: {len(self.ranks)} ranks do not match '
                             f'gpus_per_worker*replica_count={expected}.')
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError(f'{self.role}: physical ranks contain duplicates.')
        if self.gpus_per_worker != self.model_parallel_width:
            raise ValueError(f'{self.role}: gpus_per_worker must equal model_parallel_width.')


@dataclass(frozen=True)
class RLResourcePlan:
    """Complete validated resource plan applied before Ray initialization."""

    trainer: RolePlacement
    sampler: RolePlacement
    auxiliaries: Tuple[RolePlacement, ...] = ()
    shared_sampler: Optional[RolePlacement] = None
    flex_ranks: Tuple[int, ...] = ()
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    explanation: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        non_shared = [self.trainer, self.sampler, *self.auxiliaries]
        physical = [rank for role in non_shared for rank in role.ranks]
        if len(physical) != len(set(physical)):
            raise ValueError('non-shared role ranks must not overlap.')
        if set(self.flex_ranks) & set(physical):
            raise ValueError('flex ranks must be disjoint from initialized non-shared roles.')
        if len(self.flex_ranks) != len(set(self.flex_ranks)):
            raise ValueError('flex ranks contain duplicates.')
        allocated = set(physical) | set(self.flex_ranks)
        if self.shared_sampler is not None:
            if not self.shared_sampler.shared:
                raise ValueError('shared_sampler must have shared=True.')
            if not set(self.shared_sampler.ranks).issubset(set(self.trainer.ranks)):
                raise ValueError('shared sampler ranks must be a subset of the trainer world.')
        if allocated and allocated != set(range(max(allocated) + 1)):
            raise ValueError('non-shared and flex physical ranks must form a contiguous range from zero.')

    @property
    def total_physical_gpus(self) -> int:
        ranks = set(self.flex_ranks)
        for role in (self.trainer, self.sampler, *self.auxiliaries):
            ranks.update(role.ranks)
        return len(ranks)

    def describe(self) -> str:
        roles = (self.trainer, self.sampler, *self.auxiliaries,
                 *((self.shared_sampler, ) if self.shared_sampler is not None else ()))
        rows = [
            f'{role.role}: mp={role.model_parallel_width}, replicas={role.replica_count}, '
            f'ranks={list(role.ranks)}, shared={role.shared}' for role in roles
        ]
        rows.append(f'flex_ranks={list(self.flex_ranks)}, physical_gpus={self.total_physical_gpus}')
        rows.extend(self.explanation)
        return '\n'.join(rows)


@dataclass(frozen=True)
class ModelMemoryEstimate:
    """Per-card memory footprint of one model sharded to a given width."""

    parameter_count: int
    weight_bytes: int
    runtime_reserve_bytes: int
    kv_bytes_per_token: int
    graph_reserve_bytes: int


class RLResourceEstimator:
    """Pure, deterministic estimator; it never creates actors or mutates input configs."""

    def __init__(self, *, gpu_memory_gib: float, utilization: float = 0.9):
        if gpu_memory_gib <= 0:
            raise ValueError('gpu_memory_gib must be positive.')
        if not 0 < utilization <= 1:
            raise ValueError('utilization must be in (0, 1].')
        self.card_bytes = int(gpu_memory_gib * _GIB)
        self.utilization = utilization

    @classmethod
    def from_rollout_config(cls, rollout_config: Any) -> 'RLResourceEstimator':
        """``resource_gpu_memory_gib`` (the per-card memory) + the engine's memory utilization."""
        memory = getattr(rollout_config, 'resource_gpu_memory_gib', None)
        if memory is None:
            raise ValueError('resource_gpu_memory_gib is required for resource estimation.')
        utilization = (rollout_config.vllm_gpu_memory_utilization
                       if rollout_config.rollout_sampler == 'vllm'
                       else (getattr(rollout_config, 'sglang_mem_fraction_static', None) or 0.9))
        return cls(gpu_memory_gib=memory, utilization=utilization)

    @staticmethod
    def prompt_length_summary(lengths: Iterable[int]) -> Dict[str, int]:
        """Percentile summary of the encoded prompt lengths (the KV / max-seqs sizing input)."""
        values = sorted(int(value) for value in lengths)
        if not values or values[0] < 0:
            raise ValueError('prompt lengths must be a non-empty sequence of non-negative integers.')

        def percentile(q: float) -> int:
            return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]
        return {
            'p50': percentile(0.50),
            'p95': percentile(0.95),
            'p99': percentile(0.99),
            'max': values[-1],
        }

    @staticmethod
    def _dtype_bytes(dtype: Any, *, quantization: Optional[str] = None) -> float:
        if quantization:
            value = quantization.lower()
            if 'int4' in value or 'awq' in value or 'gptq' in value:
                return 0.5
            if 'int8' in value or '8bit' in value or 'fp8' in value:
                return 1.0
        value = str(dtype or 'float32').lower()
        if any(name in value for name in ('float16', 'bfloat16', 'half')):
            return 2.0
        if 'float64' in value:
            return 8.0
        if 'int8' in value or 'fp8' in value:
            return 1.0
        return 4.0

    @staticmethod
    def _parameter_count(config: Any) -> int:
        """Params from the HF config's shape fields (or the checkpoint's own ``num_parameters``)."""
        explicit = getattr(config, 'num_parameters', None)
        if isinstance(explicit, int) and explicit > 0:
            return explicit
        hidden = int(getattr(config, 'hidden_size', 0) or getattr(config, 'd_model', 0) or 0)
        layers = int(getattr(config, 'num_hidden_layers', 0) or getattr(config, 'num_layers', 0) or 0)
        vocab = int(getattr(config, 'vocab_size', 0) or 0)
        intermediate = int(getattr(config, 'intermediate_size', 0) or 4 * hidden)
        heads = int(getattr(config, 'num_attention_heads', 0) or 1)
        kv_heads = int(getattr(config, 'num_key_value_heads', 0) or heads)
        experts = int(getattr(config, 'num_local_experts', 0) or getattr(config, 'num_experts', 0) or 1)
        shared_experts = int(getattr(config, 'num_shared_experts', 0) or 0)
        if min(hidden, layers, vocab) <= 0:
            raise ValueError('HF config must expose hidden size, layer count, and vocab size for estimation.')
        head_dim = hidden // heads
        attention = 2 * hidden * hidden + 2 * hidden * kv_heads * head_dim
        mlp = 3 * hidden * intermediate * experts + 3 * hidden * intermediate * shared_experts
        return int(vocab * hidden + layers * (attention + mlp + 4 * hidden))

    def model_memory(self, config: Any, *, model_parallel_width: int, quantization: Optional[str] = None,
                     enforce_eager: bool = False, runtime_reserve_gib: float = 1.0) -> ModelMemoryEstimate:
        """Weights + KV-per-token + reserves of one model sharded across ``model_parallel_width``."""
        if model_parallel_width < 1:
            raise ValueError('model_parallel_width must be positive.')
        parameter_count = self._parameter_count(config)
        dtype_bytes = self._dtype_bytes(getattr(config, 'torch_dtype', None), quantization=quantization)
        weight_bytes = math.ceil(parameter_count * dtype_bytes / model_parallel_width)
        layers = int(getattr(config, 'num_hidden_layers', 0) or getattr(config, 'num_layers', 0) or 0)
        hidden = int(getattr(config, 'hidden_size', 0) or getattr(config, 'd_model', 0) or 0)
        heads = int(getattr(config, 'num_attention_heads', 0) or 1)
        kv_heads = int(getattr(config, 'num_key_value_heads', 0) or heads)
        raw_head_dim = getattr(config, 'head_dim', None)
        try:
            head_dim = int(raw_head_dim)
        except (TypeError, ValueError):
            head_dim = 0
        if head_dim <= 0:
            head_dim = hidden // max(1, heads)
        kv_dtype_bytes = self._dtype_bytes(getattr(config, 'torch_dtype', None))
        kv_bytes = math.ceil(2 * layers * kv_heads * head_dim * kv_dtype_bytes / model_parallel_width)
        graph_reserve = 0 if enforce_eager else int(min(1.0, max(0.15, parameter_count / 1e11)) * _GIB)
        return ModelMemoryEstimate(
            parameter_count=parameter_count,
            weight_bytes=weight_bytes,
            runtime_reserve_bytes=int(runtime_reserve_gib * _GIB),
            kv_bytes_per_token=kv_bytes,
            graph_reserve_bytes=graph_reserve)

    def _minimum_width(self, config: Any, candidates: Sequence[int], *, quantization: Optional[str],
                       enforce_eager: bool) -> Tuple[int, ModelMemoryEstimate]:
        """The narrowest candidate width whose weights + reserves fit one card's budget."""
        budget = int(self.card_bytes * self.utilization)
        for width in sorted(set(candidates)):
            estimate = self.model_memory(
                config, model_parallel_width=width, quantization=quantization, enforce_eager=enforce_eager)
            if estimate.weight_bytes + estimate.runtime_reserve_bytes + estimate.graph_reserve_bytes < budget:
                return width, estimate
        largest = max(candidates)
        estimate = self.model_memory(
            config, model_parallel_width=largest, quantization=quantization, enforce_eager=enforce_eager)
        deficit = estimate.weight_bytes + estimate.runtime_reserve_bytes + estimate.graph_reserve_bytes - budget
        raise ValueError(f'model does not fit at width={largest}; per-card memory deficit is '
                         f'{max(0, deficit) / _GIB:.2f} GiB.')

    def estimate(self, *, hf_config: Any, trainer_gpus: int, auxiliary_widths: Mapping[str, int],
                 rollout_config: Any, prompt_lengths: Sequence[int], max_completion_length: int,
                 trainer_parallel_spec: Optional[str] = None) -> RLResourcePlan:
        """One complete plan: trainer / sampler widths, replicas, auxiliaries, flex and shared pools.

        ``rollout_config`` is read for the backend's knobs (quantization, eager, max_model_len,
        resource_overrides, resource_mode, resource_flex_gpus, resource_total_gpus); the whole
        ``resource_overrides`` key set is validated against ``_ALLOWED_OVERRIDE_KEYS`` up front.
        """
        if trainer_gpus < 1:
            raise ValueError('trainer_gpus must be positive and remains fixed for the whole run.')
        overrides = dict(getattr(rollout_config, 'resource_overrides', None) or {})
        unknown = sorted(set(overrides) - _ALLOWED_OVERRIDE_KEYS)
        if unknown:
            raise ValueError(f'unknown resource_overrides keys: {unknown}; allowed keys: '
                             f'{sorted(_ALLOWED_OVERRIDE_KEYS)}.')
        factors = [value for value in range(1, trainer_gpus + 1) if trainer_gpus % value == 0]
        if trainer_parallel_spec:
            from twinkle import DeviceMesh
            trainer_mesh = DeviceMesh.from_spec(trainer_parallel_spec)
            if trainer_mesh.world_size != trainer_gpus:
                raise ValueError(f'explicit parallel_spec {trainer_parallel_spec!r} has world_size='
                                 f'{trainer_mesh.world_size}, expected the fixed trainer GPU count '
                                 f'{trainer_gpus}.')
            trainer_width = trainer_gpus // trainer_mesh.data_world_size
            overridden_width = overrides.get('trainer_model_parallel_width')
            if overridden_width is not None and int(overridden_width) != trainer_width:
                raise ValueError('trainer_model_parallel_width cannot contradict an explicit parallel_spec; '
                                 f'the spec implies width={trainer_width}, override={overridden_width}.')
        else:
            trainer_width, _ = self._minimum_width(hf_config, factors, quantization=None, enforce_eager=True)
        trainer_width = int(overrides.get('trainer_model_parallel_width', trainer_width))
        if trainer_width < 1 or trainer_gpus % trainer_width:
            raise ValueError('trainer_model_parallel_width must be positive and divide the fixed trainer '
                             'GPU count.')
        trainer_memory = self.model_memory(hf_config, model_parallel_width=trainer_width, enforce_eager=True)
        trainer_required = trainer_memory.weight_bytes + trainer_memory.runtime_reserve_bytes
        if trainer_required >= int(self.card_bytes * self.utilization):
            deficit = trainer_required - int(self.card_bytes * self.utilization)
            raise ValueError(f'trainer_model_parallel_width={trainer_width} does not fit; per-card memory '
                             f'deficit is {deficit / _GIB:.2f} GiB before optimizer and activation reserves.')

        backend = rollout_config.rollout_sampler
        configured_width = (rollout_config.vllm_tensor_parallel_size * rollout_config.vllm_pipeline_parallel_size
                            if backend == 'vllm' else rollout_config.sglang_tp_size * rollout_config.sglang_pp_size)
        candidates = sorted(set(factors + [configured_width]))
        quantization = (getattr(rollout_config, 'vllm_quantization', None) if backend == 'vllm'
                        else getattr(rollout_config, 'sglang_quantization', None))
        eager = (getattr(rollout_config, 'vllm_enforce_eager', False) if backend == 'vllm'
                 else getattr(rollout_config, 'sglang_disable_cuda_graph', False))
        sampler_width, memory = self._minimum_width(
            hf_config, candidates, quantization=quantization, enforce_eager=eager)
        sampler_width = int(overrides.get('sampler_model_parallel_width', sampler_width))
        if sampler_width < 1:
            raise ValueError('sampler_model_parallel_width must be positive.')
        # An override changes per-card weight/KV memory, so all downstream limits use the overridden width.
        memory = self.model_memory(
            hf_config, model_parallel_width=sampler_width, quantization=quantization, enforce_eager=eager)
        required = memory.weight_bytes + memory.runtime_reserve_bytes + memory.graph_reserve_bytes
        if required >= int(self.card_bytes * self.utilization):
            deficit = required - int(self.card_bytes * self.utilization)
            raise ValueError(f'sampler_model_parallel_width={sampler_width} does not fit; per-card memory '
                             f'deficit is {deficit / _GIB:.2f} GiB.')
        lengths = self.prompt_length_summary(prompt_lengths)
        total_p95 = lengths['p95'] + int(max_completion_length)
        max_context = lengths['max'] + int(max_completion_length)
        engine_context = (getattr(rollout_config, 'vllm_max_model_len', None) if backend == 'vllm'
                          else getattr(rollout_config, 'sglang_context_length', None))
        if engine_context is not None and max_context > engine_context:
            raise ValueError(f'dataset max total length {max_context} exceeds engine context {engine_context}.')
        budget = int(self.card_bytes * self.utilization)
        kv_budget = budget - memory.weight_bytes - memory.runtime_reserve_bytes - memory.graph_reserve_bytes
        if kv_budget <= 0:
            raise ValueError('sampler KV budget is non-positive after weights and runtime reserves.')
        # Tokens the KV cache can hold. The scheduler chunk is separate: one forward
        # should cover a full max_model_len prefill, not the entire cache. Writing the
        # cache size into max_num_batched_tokens also becomes vLLM's vision-encoder
        # budget and profiles dozens of max-size images.
        kv_tokens = max(1, math.floor(0.95 * kv_budget / max(1, memory.kv_bytes_per_token)))
        context_limit = int(engine_context or 0)
        if context_limit <= 0:
            context_limit = int(getattr(hf_config, 'max_position_embeddings', 0) or 0)
        configured_seqs = getattr(rollout_config, 'vllm_max_num_seqs', None) if backend == 'vllm' else None
        if 'max_num_batched_tokens' in overrides:
            seq_budget = int(overrides['max_num_batched_tokens'])
        else:
            seq_budget = kv_tokens
        max_seqs = max(1, seq_budget // max(1, total_p95))
        if configured_seqs is not None:
            max_seqs = min(max_seqs, configured_seqs)
        max_seqs = int(overrides.get('max_num_seqs', max_seqs))
        if max_seqs < 1:
            raise ValueError('max_num_seqs must be positive.')
        if configured_seqs is not None and max_seqs > configured_seqs:
            raise ValueError(f'max_num_seqs override {max_seqs} exceeds the configured engine limit '
                             f'{configured_seqs}.')
        if 'max_num_batched_tokens' in overrides:
            max_batched = int(overrides['max_num_batched_tokens'])
        else:
            # One prefill step may be as long as the context window. Also keep room for
            # one decode token from every live sequence, which vLLM requires.
            chunk = kv_tokens if context_limit <= 0 else min(kv_tokens, context_limit)
            max_batched = min(kv_tokens, max(chunk, max_seqs))
        if max_batched < 1:
            raise ValueError('max_num_batched_tokens must be positive.')
        if max_batched < max_seqs:
            raise ValueError(f'max_num_batched_tokens={max_batched} is smaller than max_num_seqs={max_seqs}; '
                             'one scheduler step must hold a decode token from every live sequence.')

        sampler_replicas = int(overrides.get('sampler_replicas', 1))
        if sampler_replicas < 1:
            raise ValueError('sampler_replicas must be positive.')
        flex_gpus = int(overrides.get('flex_gpus', getattr(rollout_config, 'resource_flex_gpus', 0)))
        if flex_gpus < 0:
            raise ValueError('flex_gpus must be non-negative.')
        next_rank = trainer_gpus
        trainer = RolePlacement('trainer', trainer_width, trainer_gpus // trainer_width,
                                tuple(range(trainer_gpus)), trainer_width)
        sampler_ranks = tuple(range(next_rank, next_rank + sampler_width * sampler_replicas))
        sampler = RolePlacement('sampler', sampler_width, sampler_replicas, sampler_ranks, sampler_width)
        next_rank += len(sampler_ranks)
        auxiliaries = []
        for role, width in auxiliary_widths.items():
            if width < 1:
                raise ValueError(f'{role}: auxiliary width must be positive.')
            ranks = tuple(range(next_rank, next_rank + width))
            auxiliaries.append(RolePlacement(role, width, 1, ranks, width))
            next_rank += width
        flex_ranks = tuple(range(next_rank, next_rank + flex_gpus))
        next_rank += flex_gpus
        if getattr(rollout_config, 'resource_mode', 'manual') == 'adaptive' and flex_gpus \
                and flex_gpus < sampler_width:
            raise ValueError(f'resource_flex_gpus={flex_gpus} cannot hold one complete sampler replica of '
                             f'width {sampler_width}. Increase the flex reservation or set it to zero.')
        total_cap = getattr(rollout_config, 'resource_total_gpus', None)
        if total_cap is not None and next_rank > total_cap:
            raise ValueError(f'resource plan needs {next_rank} GPUs but resource_total_gpus={total_cap}.')
        enable_shared = bool(overrides.get('shared_sampler',
                                           getattr(rollout_config, 'resource_mode', 'manual') == 'adaptive'))
        shared = None
        if enable_shared:
            if trainer_gpus % sampler_width:
                raise ValueError('shared sampler model_parallel_width must divide the complete trainer world.')
            shared = RolePlacement('shared_sampler', sampler_width, trainer_gpus // sampler_width,
                                   tuple(range(trainer_gpus)), sampler_width, shared=True)
        return RLResourcePlan(
            trainer=trainer,
            sampler=sampler,
            auxiliaries=tuple(auxiliaries),
            shared_sampler=shared,
            flex_ranks=flex_ranks,
            max_num_batched_tokens=max_batched,
            max_num_seqs=max_seqs,
            explanation=(
                f'prompt p50/p95/p99/max={lengths["p50"]}/{lengths["p95"]}/{lengths["p99"]}/{lengths["max"]}',
                f'sampler KV budget={kv_budget / _GIB:.2f} GiB per card',
            ))


def _hf_config(model_config: Any) -> Any:
    """The HF config behind a model-config object, carrying revision / trust_remote_code when set."""
    from transformers import AutoConfig

    kwargs = {}
    if getattr(model_config, 'model_revision', None) is not None:
        kwargs['revision'] = model_config.model_revision
    if getattr(model_config, 'trust_remote_code', None) is not None:
        kwargs['trust_remote_code'] = model_config.trust_remote_code
    config = AutoConfig.from_pretrained(model_config.model, **kwargs)
    # Composite checkpoints (Qwen3.5) keep shape and dtype on text_config. The parent
    # has neither hidden size nor vocab size, so estimation would refuse the model.
    text = getattr(config, 'text_config', None)
    if text is not None and int(getattr(text, 'hidden_size', 0) or 0) > 0:
        if getattr(text, 'torch_dtype', None) is None and getattr(config, 'torch_dtype', None) is not None:
            text.torch_dtype = config.torch_dtype
        return text
    return config


def _prompt_lengths(template: Any, prompt_rows: Sequence[Mapping[str, Any]], limit: int = 2048) -> List[int]:
    """Encoded prompt token counts, sampled down to ``limit`` rows for a large prompt set."""
    if not prompt_rows:
        raise ValueError('resource estimation requires at least one prompt row.')
    if len(prompt_rows) > limit:
        stride = len(prompt_rows) / limit
        rows = [prompt_rows[min(len(prompt_rows) - 1, int(index * stride))] for index in range(limit)]
    else:
        rows = prompt_rows
    lengths = []
    for row in rows:
        encoded = template.encode(dict(row))
        input_ids = encoded.get('input_ids') if isinstance(encoded, dict) else getattr(encoded, 'input_ids', None)
        if input_ids is None:
            raise ValueError('template.encode did not return input_ids required by resource estimation.')
        lengths.append(len(input_ids))
    return lengths


def estimate_plan(*,
                   model_config: Any,
                   distributed_config: Any,
                   rollout_config: Any,
                   template: Any,
                   prompt_rows: Sequence[Mapping[str, Any]],
                   auxiliary_groups: Sequence[Tuple[str, int]],
                   max_completion_length: int) -> Optional[RLResourcePlan]:
    """Estimate and APPLY one initial resource plan; ``None`` keeps the manual plan.

    ``resource_mode='manual'`` returns None before touching anything; 'estimate'/'adaptive' estimate
    from the HF config + prompt lengths, write the derived parallel sizes back into the configs (see
    :func:`apply_plan`), and return the plan the recipe then derives its device groups from
    (:func:`resource_device_groups`) and builds its sampler to (:attr:`RLResourcePlan.sampler`).
    """
    if getattr(rollout_config, 'resource_mode', 'manual') == 'manual':
        return None
    estimator = RLResourceEstimator.from_rollout_config(rollout_config)
    plan = estimator.estimate(
        hf_config=_hf_config(model_config),
        trainer_gpus=distributed_config.nproc_per_node,
        auxiliary_widths=dict(auxiliary_groups),
        rollout_config=rollout_config,
        prompt_lengths=_prompt_lengths(template, prompt_rows),
        max_completion_length=max_completion_length,
        trainer_parallel_spec=getattr(distributed_config, 'parallel_spec', None))
    apply_plan(plan, distributed_config=distributed_config, rollout_config=rollout_config)
    return plan


def apply_plan(plan: RLResourcePlan, *, distributed_config: Any, rollout_config: Any) -> None:
    """Write the planned parallel sizes back into the configs the later builds read."""
    trainer = plan.trainer
    if getattr(distributed_config, 'parallel_spec', None) is None:
        if trainer.model_parallel_width == 1:
            distributed_config.parallel_spec = f'dp{trainer.replica_count}'
        else:
            distributed_config.parallel_spec = f'dp{trainer.replica_count}tp{trainer.model_parallel_width}'
    sampler = plan.sampler
    if rollout_config.rollout_sampler == 'vllm':
        rollout_config.vllm_tensor_parallel_size = sampler.model_parallel_width
        rollout_config.vllm_pipeline_parallel_size = 1
        rollout_config.vllm_data_parallel_size = sampler.replica_count
        rollout_config.vllm_max_num_seqs = plan.max_num_seqs
        engine_kwargs = dict(rollout_config.vllm_engine_kwargs or {})
        engine_kwargs['max_num_batched_tokens'] = plan.max_num_batched_tokens
        rollout_config.vllm_engine_kwargs = engine_kwargs
    else:
        rollout_config.sglang_tp_size = sampler.model_parallel_width
        rollout_config.sglang_pp_size = 1
        rollout_config.sglang_dp_size = sampler.replica_count


def resource_device_groups(plan: RLResourcePlan) -> List[Tuple[str, List[int]]]:
    """The twinkle DeviceGroups a planned run initializes with (trainer, sampler, auxiliaries, flex).

    ``resource_flex`` only reserves the flex ranks: no actor is built on it, :class:`ClusterAdapter`
    places replicas there by rank (``spare_ranks``).
    """
    groups = [('model', list(plan.trainer.ranks)), ('sampler', list(plan.sampler.ranks))]
    groups.extend((role.role, list(role.ranks)) for role in plan.auxiliaries)
    if plan.flex_ranks:
        groups.append(('resource_flex', list(plan.flex_ranks)))
    return groups


class ClusterAdapter:
    """Moves replicas toward the busiest role, and lends the trainer's devices while it waits.

    ``roles`` maps a name to a pool handle (the sampler, each reward / PRM worker): anything with
    ``pressure()`` / ``add_workers`` / ``remove_workers``. The driver makes three calls:

    * :meth:`adjust` after each version change reads every role's ``pressure()`` and averages the
      last ``window`` readings. Once the busiest role's average exceeds the idlest one's by
      ``ratio_up``, one replica's devices move to it: from ``spare_ranks`` when they hold one, else
      by retiring the idlest role's least loaded replica. Moves go on, one per call and ``cooldown``
      calls apart, until the ratio falls under ``ratio_down``.
    * :meth:`lend` while the trainer waits for a batch: when it spent more than ``lend_idle_ratio``
      of the last step waiting, ``model`` is offloaded and the busiest role grows onto
      ``lend_ranks``.
    * :meth:`reclaim` before the batch is handed out retires the replicas on ``lend_ranks`` (their
      requests are resubmitted to the other replicas) and reloads ``model``.

    A new replica is passed to ``warmups[role]`` before it takes requests (the sampler's
    ``checkpoint_manager.warm`` loads the current weights into it). ``widths[role]`` is the device
    count of one replica of that role (default 1).
    """

    def __init__(self,
                 roles: Mapping[str, Any],
                 *,
                 model: Any = None,
                 lend_ranks: Optional[Sequence[int]] = None,
                 spare_ranks: Sequence[int] = (),
                 warmups: Optional[Mapping[str, Callable[[Any], None]]] = None,
                 widths: Optional[Mapping[str, int]] = None,
                 window: int = 3,
                 ratio_up: float = 1.5,
                 ratio_down: float = 1.2,
                 cooldown: int = 5,
                 lend_idle_ratio: float = 0.3):
        if not roles:
            raise ValueError('ClusterAdapter needs at least one role.')
        if window < 1 or cooldown < 0:
            raise ValueError(f'window must be >= 1 and cooldown >= 0, got {window} / {cooldown}.')
        if not 1.0 <= ratio_down <= ratio_up:
            raise ValueError(f'need 1 <= ratio_down <= ratio_up, got {ratio_down} / {ratio_up}.')
        if lend_ranks and model is None:
            raise ValueError('lend_ranks are the trainer\'s devices: pass the trainer model to offload.')
        unknown = sorted((set(warmups or {}) | set(widths or {})) - set(roles))
        if unknown:
            raise ValueError(f'warmups / widths name unknown roles {unknown}; roles are {sorted(roles)}.')
        if set(spare_ranks) & set(lend_ranks or ()):
            raise ValueError('spare_ranks and lend_ranks must not overlap.')
        self.roles = dict(roles)
        self.model = model
        self.lend_ranks = tuple(lend_ranks or ())
        self.warmups = dict(warmups or {})
        self.widths = {name: int((widths or {}).get(name, 1)) for name in self.roles}
        self.window = window
        self.ratio_up = ratio_up
        self.ratio_down = ratio_down
        self.cooldown = cooldown
        self.lend_idle_ratio = lend_idle_ratio
        self._spare = sorted(spare_ranks)
        self._history = {name: deque(maxlen=window) for name in self.roles}
        self._engaged = False
        self._cooldown = 0
        #: the role holding replicas on ``lend_ranks``, until ``reclaim``
        self._lent: Optional[str] = None
        self._lock = threading.Lock()

    def adjust(self) -> None:
        """Read the roles' pressure and move at most one replica's devices to the busiest role."""
        with self._lock:
            for name, handle in self.roles.items():
                self._history[name].append(float(handle.pressure()))
            if self._cooldown > 0:
                self._cooldown -= 1
                return
            if any(len(history) < self.window for history in self._history.values()):
                return
            averages = {name: statistics.fmean(history) for name, history in self._history.items()}
            hot = max(averages, key=averages.get)
            cold = min((name for name in averages if name != hot), key=averages.get, default=None)
            ratio = averages[hot] / max(averages[cold] if cold is not None else 0.0, 1e-6)
            self._engaged = ratio > (self.ratio_down if self._engaged else self.ratio_up)
            if self._engaged and self._move(hot, cold):
                self._cooldown = self.cooldown

    def lend(self, train_idle_ratio: float) -> None:
        """Offload the trainer and grow the busiest role onto ``lend_ranks``; only while it waits."""
        with self._lock:
            if not self.lend_ranks or self._lent is not None or train_idle_ratio <= self.lend_idle_ratio:
                return
            pressures = {name: float(handle.pressure()) for name, handle in self.roles.items()}
            name = max(pressures, key=pressures.get)
            width = self.widths[name]
            count = len(self.lend_ranks) // width
            if count == 0:
                return
            self.model.offload_to_cpu()
            try:
                self.roles[name].add_workers(
                    count, warmup=self.warmups.get(name), ranks=list(self.lend_ranks[:count * width]))
            except BaseException:
                self.model.reload_to_gpu()
                raise
            self._lent = name
            logger.info(f'ClusterAdapter: lent trainer ranks {list(self.lend_ranks[:count * width])} to {name}.')

    def reclaim(self) -> None:
        """Retire the replicas on ``lend_ranks`` and reload the trainer; a no-op when nothing is lent."""
        with self._lock:
            if self._lent is None:
                return
            name, self._lent = self._lent, None
            try:
                self.roles[name].remove_workers(ranks=list(self.lend_ranks))
            finally:
                self.model.reload_to_gpu()

    def _move(self, hot: str, cold: Optional[str]) -> bool:
        """One replica's devices to ``hot``: spare ranks first, else a replica retired from ``cold``."""
        width = self.widths[hot]
        if len(self._spare) < width and cold is not None and len(self._spare) + self.widths[cold] >= width:
            try:
                released = self.roles[cold].remove_workers(1)
            except ValueError:  # its last replica: it has nothing to give
                released = []
            self._spare = sorted(set(self._spare) | (set(released) - set(self.lend_ranks)))
        if len(self._spare) < width:
            return False
        ranks, self._spare = self._spare[:width], self._spare[width:]
        try:
            self.roles[hot].add_workers(1, warmup=self.warmups.get(hot), ranks=ranks)
        except BaseException:
            self._spare = sorted(self._spare + ranks)
            raise
        logger.info(f'ClusterAdapter: ranks {ranks} -> {hot}.')
        return True
