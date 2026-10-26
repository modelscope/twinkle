# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import math
import statistics
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from twinkle.perf import CounterRegistry

if TYPE_CHECKING:
    from .streaming_driver import DriverStats

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


@dataclass
class ResourceWindow:
    """One observe-window sample of the production/consumption time balance (per optimizer step).

    ``trainer_idle`` / ``score_time`` / ``generation_time`` are the window's wall-clock deltas: the
    trainer bubble, the scoring channel's occupancy (reap waits), and the generation channel's
    occupancy (in-flight admit->collect time). ``blocked_passes`` counts admission passes gated by
    backpressure (buffer capacity or the staleness window) -- the production-surplus signal.
    """

    step: int
    trainer_idle: float
    score_time: float
    generation_time: float
    blocked_passes: float
    handover_time: float


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
        """``resource_gpu_memory_gib`` (None -> the cluster's smallest reported GPU) + utilization."""
        memory = getattr(rollout_config, 'resource_gpu_memory_gib', None)
        if memory is None:
            memory = cls._cluster_min_gpu_memory_gib()
        utilization = (rollout_config.vllm_gpu_memory_utilization
                       if rollout_config.rollout_sampler == 'vllm'
                       else (getattr(rollout_config, 'sglang_mem_fraction_static', None) or 0.9))
        return cls(gpu_memory_gib=memory, utilization=utilization)

    @staticmethod
    def _cluster_min_gpu_memory_gib() -> float:
        try:
            from twinkle.infra import cluster_resources
            resources = cluster_resources()
            for key, value in resources.items():
                if key.startswith('accelerator_memory_gib_'):
                    return float(value)
        except Exception:
            pass
        raise ValueError('resource_gpu_memory_gib is required when the cluster does not expose per-GPU memory.')

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
        kv_dtype_bytes = self._dtype_bytes(getattr(config, 'torch_dtype', None))
        kv_bytes = math.ceil(2 * layers * kv_heads * (hidden // heads) * kv_dtype_bytes / model_parallel_width)
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
        max_batched = max(1, math.floor(0.95 * kv_budget / max(1, memory.kv_bytes_per_token)))
        max_batched = int(overrides.get('max_num_batched_tokens', max_batched))
        if max_batched < 1:
            raise ValueError('max_num_batched_tokens must be positive.')
        max_seqs = max(1, max_batched // max(1, total_p95))
        configured_seqs = getattr(rollout_config, 'vllm_max_num_seqs', None) if backend == 'vllm' else None
        if configured_seqs is not None:
            max_seqs = min(max_seqs, configured_seqs)
        max_seqs = int(overrides.get('max_num_seqs', max_seqs))
        if max_seqs < 1:
            raise ValueError('max_num_seqs must be positive.')
        if configured_seqs is not None and max_seqs > configured_seqs:
            raise ValueError(f'max_num_seqs override {max_seqs} exceeds the configured engine limit '
                             f'{configured_seqs}.')

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
    return AutoConfig.from_pretrained(model_config.model, **kwargs)


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


def flex_role_groups(plan: RLResourcePlan, role: str = 'sampler') -> Dict[str, Tuple[int, ...]]:
    """The ``flex_<role>_<i>`` slot map the flex rank span splits into (whole replicas only).

    The flex pool is role-generic at plan level: one span, any elastic role. Which roles may draw
    from it is an adapter runtime concern (``elastic_models``); the per-role split below sizes each
    slot by that role's model-parallel width, so a span reserved for width-2 RMs yields fewer,
    wider slots than one for width-1 samplers.
    """
    width = plan.sampler.model_parallel_width
    if role != 'sampler':
        for aux in plan.auxiliaries:
            if aux.role == role:
                width = aux.model_parallel_width
                break
    count = len(plan.flex_ranks) // width
    return {f'flex_{role}_{index}': tuple(plan.flex_ranks[index * width:(index + 1) * width])
            for index in range(count)}


def elastic_roles(plan: RLResourcePlan) -> Tuple[str, ...]:
    """The roles allowed to draw flex slots: the sampler plus every planned auxiliary role.

    Auxiliaries (reward_<i>/prm_0/reference/teacher) are plan-level roles already; a role joins the
    ELASTIC runtime only when the adapter is handed a matching handle via ``elastic_models`` --
    without one, its overlay groups exist but no decision can ever address it.
    """
    return ('sampler', *(aux.role for aux in plan.auxiliaries))


def resource_device_groups(plan: RLResourcePlan, *, adaptive: bool = False) -> List[Tuple[str, List[int]]]:
    """The twinkle DeviceGroups a planned run initializes with (trainer, sampler, auxiliaries, flex)."""

    def sampler_groups(shared: bool) -> List[Tuple[str, List[int]]]:
        role = plan.shared_sampler if shared else plan.sampler
        if role is None:
            return []
        return [('shared_sampler' if shared else 'sampler', list(role.ranks))]

    groups = [('model', list(plan.trainer.ranks))]
    groups.extend(sampler_groups(shared=False))
    groups.extend((role.role, list(role.ranks)) for role in plan.auxiliaries)
    if plan.flex_ranks:
        # The base group reserves physical capacity. Candidate role groups below are overlays and
        # create no additional placement-group bundles; actors are constructed only when a slot
        # receives that role.
        groups.append(('resource_flex', list(plan.flex_ranks)))
        if adaptive:
            for role in elastic_roles(plan):
                groups.extend((name, list(ranks)) for name, ranks in flex_role_groups(plan, role).items())
    if plan.shared_sampler is not None:
        groups.extend(sampler_groups(shared=True))
    return groups


class ClusterAdapter:
    """The adaptive cluster runtime: replica scaling, shared-GPU borrowing, one-call publication.

    ONE object where dev kept two that pointed at each other (an ``AdaptiveResourceController`` and
    an ``AdaptiveColocateHandover``): it owns the dedicated (disjoint-GPU) sampler actors, the
    optional shared (trainer-GPU) engines, the pre-reserved flex slots, and both weight-sync
    channels. A single change is driven only at batch and publish boundaries:

    * :meth:`observe` folds one batch boundary's time deltas into the decision window;
    * :meth:`decide` names at most one change per publish boundary (activate/deactivate the shared
      pool, scale a flex replica in or out), debounced by ``min_hold_steps``;
    * :meth:`publish` is the WHOLE publication barrier: quiesce admission, run the controller's
      prepare/finish phases around the dedicated-channel weight sync, borrow the shared engines on
      the same fresh weights, and restore admission -- with the full failure path
      (``fail_publish`` + ``reclaim``) that returns borrowed GPUs, drops half-built actors and
      reloads the trainer.
    """

    def __init__(self, *,
                 sampler: Any,
                 plan: RLResourcePlan,
                 model: Any,
                 additional_models: Sequence[Any] = (),
                 backend: str = 'vllm',
                 sleep_level: int = 1,
                 window_steps: int = 8,
                 min_hold_steps: int = 16,
                 elastic_models: Mapping[str, Any] = None):
        if window_steps < 1 or min_hold_steps < 1:
            raise ValueError('window_steps and min_hold_steps must be positive.')
        if sleep_level not in (0, 1):
            raise ValueError('the shared sampler supports only sleep_level 0 or 1.')
        from twinkle.checkpoint_engine import CheckpointEngineManager

        self.sampler = sampler
        self.plan = plan
        self.models = (model, *tuple(additional_models))
        self.backend = backend
        self.sleep_level = sleep_level or 1
        self.window_steps = window_steps
        self.min_hold_steps = min_hold_steps
        # The role -> frozen-model-handle registry for RM-side elasticity (protocol: the handle has
        # ``add_workers(group, ranks)`` / ``remove_workers(worker_ids)`` / ``worker_ids()`` -- the
        # same actor-native shape the sampler itself grew). Roles without a handle stay static.
        self.elastic_models: Dict[str, Any] = dict(elastic_models or {})
        self._flex_groups = flex_role_groups(plan)
        self._free_flex_groups = deque(self._flex_groups)
        self._flex_workers: Dict[str, Any] = {}
        self._rm_flex_workers: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rl-flex-sampler')
        # In-flight change state, at most ONE change at a time: ``_pending`` holds the (group,
        # future) of a background actor construction; ``_draining`` -> ``_removal`` the two phases
        # of a scale-down; ``_addition`` the finished-but-unactivated scale-up worker ids.
        self._pending: Optional[Tuple[str, Future]] = None
        self._addition: Optional[Tuple[str, Any]] = None
        self._draining: Optional[Tuple[str, Any]] = None
        self._removal: Optional[Tuple[str, Any]] = None
        self._decision: Optional[str] = None
        self._decision_step = 0
        self._shared_change = False
        self._shared_previous_enabled = None
        self._shared_enabled = False
        self.in_generation = False
        self._closed = False
        self.last_error = None
        # Hand-over accounting rides the perf package's always-live counters: per-adapter
        # lifecycle, monotonic, observe reads window deltas off it.
        self._counters = CounterRegistry()
        self._windows: deque = deque(maxlen=window_steps)
        self._last_change_step = -min_hold_steps
        self._shared_pressure_windows = 0
        self._last_stats: Dict[str, float] = {}
        # Weight-sync channels: one NCCL channel for the dedicated (disjoint-GPU) actors, one CUDA-IPC
        # channel for the shared (trainer-GPU) engines. The shared channel only exists when the plan
        # reserved a shared pool.
        self.dedicated_manager: Optional[CheckpointEngineManager] = None
        self.shared_manager: Optional[CheckpointEngineManager] = None
        if plan.shared_sampler is not None:
            shared_ids = self._initialize_shared_workers('shared_sampler', plan.shared_sampler.ranks)
            self.shared_manager = CheckpointEngineManager(
                model=model, sampler=sampler, platform='GPU', mode='colocate', channel='sampler:ipc',
                receiver_worker_ids=shared_ids,
                colocate_trainer_ranks=[ranks[0] for ranks in sampler.worker_ranks(shared_ids)])

    @classmethod
    def build(cls, *, model: Any, sampler: Any, plan: RLResourcePlan, rollout_config: Any,
              additional_models: Sequence[Any] = ()) -> 'ClusterAdapter':
        """Build the adapter around an already-placed sampler, or clean everything up on failure.

        The caller builds the sampler to the plan's shape (``DeviceMesh.from_sizes(dp_size=
        plan.sampler.replica_count)``); this activates its dedicated ranks, wires both weight-sync
        channels and the shared engines. Any failure closes the channels, the workers and restores
        the trainer's GPU residency before re-raising.
        """
        adapter = None
        try:
            sampler.configure_workers(plan.sampler.ranks, state='ACTIVE')
            adapter = cls(
                sampler=sampler,
                plan=plan,
                model=model,
                additional_models=additional_models,
                backend=rollout_config.rollout_sampler,
                sleep_level=getattr(rollout_config, 'sleep_level', 1),
                window_steps=int(getattr(rollout_config, 'resource_window_steps', 8)),
                min_hold_steps=int(getattr(rollout_config, 'resource_min_hold_steps', 16)))
            from twinkle.checkpoint_engine import CheckpointEngineManager

            adapter.dedicated_manager = CheckpointEngineManager(
                model=model, sampler=sampler, platform='GPU', mode='standalone', channel='sampler:nccl',
                receiver_worker_ids=sampler.worker_ids(shared=False))
            return adapter
        except BaseException:
            # Either failure point restores the trainer's GPU residency before re-raising: an __init__
            # failure is cleaned inside _initialize_shared_workers; a later failure here goes through
            # the adapter's own channels/workers/reclaim chain, in that order (dev's original order).
            if adapter is not None:
                adapter.close_channels()
                try:
                    sampler.close_workers()
                except Exception:
                    pass
                try:
                    adapter.reclaim()
                except Exception:
                    pass
            else:
                try:
                    sampler.close_workers()
                except Exception:
                    pass
            raise

    # --- the publication barrier -------------------------------------------------------------------

    def publish(self) -> None:
        """The whole weight-publication barrier: dedicated sync + shared borrow + replica changes.

        Runs the controller's ``prepare_publish`` -> dedicated-channel ``sync_weights`` ->
        ``finish_publish`` phases (or the full ``fail_publish`` + ``reclaim`` failure path) around
        the caller's trained weights, borrowing the shared engines onto the same fresh weights so
        every ACTIVE sampler replica generates from one policy version. The recipe's visible loop
        body is ``adapter.publish() if adapter else engine.sync_weights()``.
        """
        suspended = self.sampler.worker_ids()
        self.sampler.suspend_admission(suspended)
        dedicated_ids = self.sampler.worker_ids(shared=False, states=('ACTIVE', 'DRAINING'))
        try:
            self.sampler.pause_generation(_worker_ids=dedicated_ids)
            dedicated_ids = self.prepare_publish()
            self.dedicated_manager.set_receiver_workers(dedicated_ids)
            self.dedicated_manager.sync_weights(merge_and_sync=True)
            self._borrow_shared(publish=True)
            self.finish_publish()
            self.sampler.resume_generation(_worker_ids=dedicated_ids)
            self.sampler.resume_admission(suspended + dedicated_ids)
        except BaseException as error:
            try:
                self.fail_publish(error)
            except Exception as cleanup:
                self.last_error = f'{type(cleanup).__name__}: {cleanup}'
            try:
                self.reclaim()
            except Exception as cleanup:
                self.last_error = self.last_error or f'{type(cleanup).__name__}: {cleanup}'
            raise

    def _initialize_shared_workers(self, group: str, ranks: Sequence[int]) -> Tuple[Any, ...]:
        """Build the shared engines, releasing trainer memory first and restoring it after."""
        worker_ids: Tuple[Any, ...] = ()
        initialization_error = None
        can_reload = True
        self.in_generation = True
        try:
            for one in self.models:
                _wait(one.offload_to_cpu())
            worker_ids = self.sampler.add_workers(group, list(ranks), shared=True, engine_args={
                ('enable_sleep_mode' if self.backend == 'vllm' else 'enable_memory_saver'): True})
            self.sampler.pause_workers(worker_ids)
            self.sampler.pause_generation(_worker_ids=worker_ids)
            _wait(self.sampler.sleep(level=self.sleep_level, _worker_ids=worker_ids))
        except BaseException as exc:
            initialization_error = exc
            try:
                if worker_ids and not self.sampler.remove_workers(worker_ids):
                    raise RuntimeError('Shared sampler initialization still has unfinished worker calls.')
            except Exception:
                pass
            can_reload = not set(worker_ids) & set(self.sampler.worker_ids(shared=True))
            raise
        finally:
            if can_reload:
                try:
                    self._reload_models()
                    self.in_generation = False
                except BaseException:
                    if initialization_error is None:
                        raise
        return worker_ids

    def _borrow_shared(self, *, publish: bool) -> None:
        """Wake the shared engines for a generation window, on the freshly published weights."""
        shared_ids = self.sampler.worker_ids(shared=True)
        outstanding = any(self.sampler.worker_inflight(shared_ids))
        if not shared_ids or (not self._shared_enabled and not outstanding) or self.in_generation:
            return
        started = time.perf_counter()
        # Even with new shared admission stopped, already-submitted requests must keep generating and
        # receive the new policy like everyone else.
        self.in_generation = True
        try:
            _wait(self.sampler.wake_up(tags=['weights'], _worker_ids=shared_ids))
            if publish:
                trainer_ranks = [ranks[0] for ranks in self.sampler.worker_ranks(shared_ids)]
                self.shared_manager.set_receiver_workers(shared_ids, colocate_trainer_ranks=trainer_ranks)
                self.shared_manager.sync_weights(merge_and_sync=True)
            for one in self.models:
                _wait(one.offload_to_cpu())
            _wait(self.sampler.wake_up(tags=['kv_cache'], _worker_ids=shared_ids))
            self.sampler.resume_generation(_worker_ids=shared_ids)
            if self._shared_enabled:
                self.sampler.activate_workers(shared_ids)
        except BaseException:
            try:
                self.reclaim()
            except Exception as cleanup:
                self.last_error = self.last_error or f'{type(cleanup).__name__}: {cleanup}'
            raise
        # Bare add (not time_span): the interval must exclude the except-path cleanup, which calls
        # reclaim() and is timed there -- a time_span around the try would double-count it.
        self._counters.add('handover', time.perf_counter() - started)

    def reclaim(self) -> None:
        """Stop shared admission, sleep the shared engines, and restore the trainer's GPU residency."""
        if not self.in_generation:
            return
        shared_ids = self.sampler.worker_ids(shared=True)
        # time_span's finally carries the old try/finally semantics: failure paths still count.
        with self._counters.time_span('handover'):
            self.sampler.pause_workers(shared_ids)
            self.sampler.pause_generation(_worker_ids=shared_ids)
            _wait(self.sampler.sleep(level=self.sleep_level, _worker_ids=shared_ids))
            # On a pause/sleep failure keep the hand-over state; the trainer's memory is restored
            # only after the engines are down.
            self._reload_models()
            self.in_generation = False

    def _reload_models(self) -> None:
        error = None
        for one in reversed(self.models):
            try:
                _wait(one.reload_to_gpu())
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error

    def close_channels(self) -> None:
        if self.dedicated_manager is not None:
            try:
                self.dedicated_manager.close()
            finally:
                if self.shared_manager is not None:
                    self.shared_manager.close()

    # --- the single-change controller ----------------------------------------------------------------

    @property
    def change_pending(self) -> bool:
        return (self._decision is not None or self._pending is not None or self._draining is not None
                or self._addition is not None or self._removal is not None)

    def register_decision(self, decision: Optional[str], *, step: int) -> None:
        self._decision_step = step
        if decision is not None and not self._closed and not self.change_pending:
            self._decision = decision

    def prepare_publish(self) -> Tuple[Any, ...]:
        """Apply the pending decision after the pause barrier; background construction does no sync."""
        decision, self._decision = self._decision, None
        self._shared_change = decision in ('activate_shared', 'deactivate_shared')
        self._shared_previous_enabled = self._shared_enabled if self._shared_change else None
        if decision == 'activate_shared':
            self._shared_enabled = bool(self.shared_manager is not None)
        elif decision == 'deactivate_shared':
            self._shared_enabled = False
        elif decision == 'scale_up_sampler' and self._free_flex_groups:
            group = self._free_flex_groups.popleft()
            try:
                self._pending = (group, self._executor.submit(self.sampler.add_workers, group,
                                                              list(self._flex_groups[group])))
            except BaseException:
                self._free_flex_groups.appendleft(group)
                raise
        elif decision == 'scale_down_sampler' and self._flex_workers:
            group, worker_ids = next(reversed(self._flex_workers.items()))
            self.sampler.drain_workers(worker_ids)
            self._draining = (group, worker_ids)
        elif decision is not None and decision.startswith('scale_up_rm:'):
            role = decision[len('scale_up_rm:'):]
            handle = self.elastic_models.get(role)
            if handle is not None and self._free_flex_groups:
                group = self._free_flex_groups.popleft()
                try:
                    worker_ids = handle.add_replicas(group)
                    self._rm_flex_workers[role] = (group, worker_ids)
                except BaseException:
                    self._free_flex_groups.appendleft(group)
                    raise
        elif decision is not None and decision.startswith('scale_down_rm:'):
            role = decision[len('scale_down_rm:'):]
            held = self._rm_flex_workers.get(role)
            if held is not None:
                group, worker_ids = held
                # A frozen RM is a stateless forward service: no drain phase, the removal is the
                # whole retirement (the scoring worker's in-flight call finishes on the actor Ray
                # keeps alive until the call returns).
                handle = self.elastic_models[role]
                handle.remove_replicas(worker_ids)
                self._rm_flex_workers.pop(role, None)
                self._free_flex_groups.append(group)
        if decision is not None:
            self._shared_pressure_windows = 0
        if self._pending is not None and self._pending[1].done():
            group, future = self._pending
            self._pending = None
            try:
                self._addition = (group, future.result())
            except Exception as error:
                self.last_error = f'{type(error).__name__}: {error}'
                self._free_flex_groups.appendleft(group)
        if self._draining is not None and self.sampler.workers_idle(self._draining[1]):
            self._removal = self._draining
        receivers = self.sampler.worker_ids(shared=False, states=('ACTIVE', 'DRAINING'))
        if self._addition is not None:
            receivers += self._addition[1]
        removed = () if self._removal is None else self._removal[1]
        return tuple(worker_id for worker_id in receivers if worker_id not in removed)

    def finish_publish(self) -> None:
        """Commit the applied change: activate added workers, retire drained ones, mark the change."""
        changed = self._shared_change or self._removal is not None or self._addition is not None
        if self._removal is not None:
            group, worker_ids = self._removal
            try:
                if not self.sampler.remove_workers(worker_ids):
                    raise RuntimeError('A drained sampler actor received new work during its publication '
                                       'barrier.')
            finally:
                # Even when removal errored, return the slot exactly once, and only if the actors are gone.
                if not set(worker_ids) & set(self.sampler.worker_ids()):
                    self._flex_workers.pop(group)
                    self._free_flex_groups.append(group)
                    self._removal = self._draining = None
        if self._addition is not None:
            group, worker_ids = self._addition
            self.sampler.activate_workers(worker_ids)
            self._flex_workers[group] = worker_ids
            self._addition = None
            self.last_error = None
        if changed:
            self._last_change_step = self._decision_step
        self._shared_change = False
        self._shared_previous_enabled = None

    def fail_publish(self, publication_error: Optional[BaseException] = None) -> None:
        """Roll a half-applied change back: shared enablement, half-built actors, receiver sets."""
        if publication_error is not None:
            self.last_error = f'{type(publication_error).__name__}: {publication_error}'
        if self._shared_previous_enabled is not None:
            self._shared_enabled = bool(self._shared_previous_enabled) and self.shared_manager is not None
        self._shared_change = False
        self._shared_previous_enabled = None
        self._removal = None
        error = None
        if self._addition is not None:
            group, worker_ids = self._addition
            try:
                self.dedicated_manager.close()
            except BaseException as exc:
                error = exc
            try:
                if not self.sampler.remove_workers(worker_ids):
                    raise RuntimeError('Failed publication left a sampler actor with unfinished '
                                       'initialization work.')
            except BaseException as exc:
                error = error or exc
            finally:
                if not set(worker_ids) & set(self.sampler.worker_ids()):
                    self._free_flex_groups.appendleft(group)
                    self._addition = None
        try:
            self.dedicated_manager.set_receiver_workers(
                self.sampler.worker_ids(shared=False, states=('ACTIVE', 'DRAINING')))
        except BaseException as exc:
            error = error or exc
        if error is not None:
            raise error

    def shutdown(self) -> None:
        """Tear the adapter down without leaking a pin, an actor, or the trainer's GPU residency."""
        if self._closed:
            return
        self._closed = True
        self.sampler.close_admission()
        # Retire RM flex replicas first: their actors live on the flex span, and the flex-span
        # teardown below (close_workers / reclaim) must not race a still-alive RM actor on it.
        for role, (group, worker_ids) in list(self._rm_flex_workers.items()):
            try:
                self.elastic_models[role].remove_workers(list(worker_ids))
            except Exception:
                pass
        self._rm_flex_workers.clear()
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
            if self._pending is not None and not self._pending[1].cancelled():
                try:
                    self._pending[1].result()
                except Exception:
                    pass
            try:
                self.sampler.abort_all_inflight()
            finally:
                self.reclaim()
        finally:
            try:
                self.close_channels()
            finally:
                try:
                    self.sampler.close_workers()
                finally:
                    self.reclaim()

    # --- observation / decision ------------------------------------------------------------------------

    def observe(self, *, step: int, driver_stats: DriverStats) -> ResourceWindow:
        """Fold one batch boundary's stat deltas into the decision window (the commit boundary).

        Every signal is a driver-local counter read off ``driver_stats`` (zero RPC): train / idle /
        score / generation are mutually-exclusive wall-clock buckets on the driver thread, blocked
        counts the backpressure-gated admission passes.
        """
        current = {
            'train': float(driver_stats.train_active_time),
            'idle': float(driver_stats.idle_time),
            'score': float(driver_stats.score_time),
            'generation': float(driver_stats.generation_time),
            'blocked': float(driver_stats.capacity_blocked_passes + driver_stats.staleness_blocked_passes),
            'handover': self._counters.get('handover'),
        }
        delta = {key: value - self._last_stats.get(key, value) for key, value in current.items()}
        self._last_stats = current
        total = max(1e-9, delta['train'] + delta['idle'])
        window = ResourceWindow(step, delta['idle'] / total, delta['score'], delta['generation'],
                                delta['blocked'], delta['handover'])
        self._windows.append(window)
        return window

    def decide(self, *, step: int, publish_boundary: bool) -> Optional[str]:
        """The single change to make at a publish boundary, or ``None`` (held by window / cooldown)."""
        if (not publish_boundary or self.change_pending or step - self._last_change_step < self.min_hold_steps
                or len(self._windows) < self.window_steps):
            return None
        idle = statistics.fmean(window.trainer_idle for window in self._windows)
        handover = statistics.fmean(window.handover_time for window in self._windows)
        loaded = self.sampler.inflight_count > 0
        profitable = loaded and idle > 0 and idle > 1.25 * handover
        self._shared_pressure_windows = self._shared_pressure_windows + 1 if profitable else 0
        if self._shared_pressure_windows >= 2:
            if not self._shared_enabled and self.shared_manager is not None:
                return 'activate_shared'
            if self._free_flex_groups:
                return 'scale_up_sampler'
        if idle < 0.75 * handover:
            if self._flex_workers:
                return 'scale_down_sampler'
            if self._shared_enabled:
                return 'deactivate_shared'
        # RM-side elasticity, channel-split: the production side's wall time divides into the scoring
        # channel (score_time -- the reap waits) and the generation channel (generation_time -- in-flight
        # occupancy); the scoring share picks WHICH channel grows, replacing the old merged share that
        # could only ever pick the first role without a replica. The near-dead-band (score_share within
        # 0.35..0.65) refuses the channel call: with the two channels at comparable speed, a replica on
        # either side pays back too little to justify its engine-load cost, and acting would only
        # oscillate -- paired with the per-change hold above (one change per min_hold_steps) this is the
        # anti-oscillation pair. Backpressure (blocked passes) plus a drained idle window is the
        # production-surplus signal for shrinking. The thresholds are constants, not knobs: no consumer
        # has tuning evidence yet.
        if self.elastic_models:
            score = statistics.fmean(window.score_time for window in self._windows)
            generation = statistics.fmean(window.generation_time for window in self._windows)
            blocked = statistics.fmean(window.blocked_passes for window in self._windows)
            channel_total = score + generation
            score_share = score / channel_total if channel_total > 0 else 0.0
            has_rm_replica = any(self._rm_flex_workers.get(role) for role in self.elastic_models)
            if score_share > 0.65 and idle > 0 and self._free_flex_groups:
                return next((f'scale_up_rm:{role}' for role in self.elastic_models
                             if not self._rm_flex_workers.get(role)), None)
            if score_share < 0.35 and has_rm_replica and (blocked > 0 or idle < 0.75 * handover):
                return next((f'scale_down_rm:{role}' for role in self.elastic_models
                             if self._rm_flex_workers.get(role)), None)
        return None


def _wait(result: Any) -> Any:
    """Resolve a lazily-collected remote result (twinkle's LazyCollect), or pass the value through."""
    if callable(result) and getattr(result, '_is_lazy_collect', False):
        return result()
    return result
