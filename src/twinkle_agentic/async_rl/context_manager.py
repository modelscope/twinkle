# Copyright (c) ModelScope Contributors. All rights reserved.
"""Single-owner control plane for async multi-LoRA RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .types import LoraContext, PartitionAdmission, RolloutPolicy


class ContextStatus(StrEnum):
    ADDING = 'ADDING'
    ACTIVE = 'ACTIVE'
    DRAINING = 'DRAINING'
    EXHAUSTED = 'EXHAUSTED'
    FINISHED = 'FINISHED'
    REMOVED = 'REMOVED'
    FAILED = 'FAILED'


@dataclass
class _ContextState:
    context: LoraContext
    policy: RolloutPolicy
    policy_history: list[RolloutPolicy] = field(default_factory=list)
    next_step: int = 0
    completed_partitions: int = 0
    status: ContextStatus = ContextStatus.ACTIVE
    dataset_exhausted: bool = False
    training_partition_id: str | None = None
    live_partitions: dict[str, PartitionAdmission] = field(default_factory=dict)
    rollout_policy_references: dict[str, int] = field(default_factory=dict)
    max_steps: int | None = None


class LoraContextManager:
    """Ray-safe control plane; it deliberately knows no TQ sample readiness."""

    def __init__(self, *, max_staleness: int = 0, max_steps: int | None = None):
        if max_staleness < 0:
            raise ValueError('max_staleness must be non-negative')
        if max_steps is not None and max_steps < 0:
            raise ValueError('max_steps must be non-negative')
        self.max_staleness = int(max_staleness)
        self.max_steps = max_steps
        self._contexts: dict[str, _ContextState] = {}
        self._creation_order = 0
        self._stop_requested = max_steps == 0

    def register_context(self,
                         context: LoraContext,
                         *,
                         adapter_path: str | None = None,
                         policy_version: int = 0,
                         max_steps: int | None = None,
                         status: ContextStatus = ContextStatus.ACTIVE) -> None:
        if context.key in self._contexts:
            return
        if max_steps is not None and max_steps < 0:
            raise ValueError('max_steps must be non-negative')
        self._contexts[context.key] = _ContextState(
            context=context,
            policy=RolloutPolicy(context.key, context.adapter_name, int(policy_version), adapter_path),
            policy_history=[RolloutPolicy(context.key, context.adapter_name, int(policy_version), adapter_path)],
            status=status,
            max_steps=max_steps,
        )

    def reserve_context(self, context: LoraContext, *, max_steps: int | None = None) -> None:
        if context.key in self._contexts:
            raise KeyError(f'context already exists: {context.key}')
        if max_steps is not None and max_steps < 0:
            raise ValueError('max_steps must be non-negative')
        self.register_context(context, status=ContextStatus.ADDING, max_steps=max_steps)

    def activate_context(self,
                         context: LoraContext | str,
                         *,
                         adapter_path: str | None = None,
                         policy_version: int = 0) -> None:
        state = self._state(context)
        if state.status is not ContextStatus.ADDING:
            raise RuntimeError(f'{state.context.key} cannot activate from {state.status}')
        policy = RolloutPolicy(state.context.key, state.context.adapter_name, int(policy_version), adapter_path)
        state.policy = policy
        state.policy_history = [policy]
        state.status = ContextStatus.ACTIVE

    def request_context_drain(self, context: LoraContext | str) -> None:
        state = self._state(context)
        if state.status in (ContextStatus.REMOVED, ContextStatus.FAILED):
            return
        if state.status is ContextStatus.ADDING:
            raise RuntimeError(f'{state.context.key} is still being added')
        state.status = ContextStatus.DRAINING

    def context_is_drained(self, context: LoraContext | str) -> bool:
        return not self._state(context).live_partitions

    def fail_context(self, context: LoraContext | str) -> None:
        state = self._state(context)
        if state.live_partitions:
            raise RuntimeError(f'{state.context.key} still has live partitions')
        state.status = ContextStatus.FAILED

    def unregister_context(self, context: LoraContext | str) -> None:
        state = self._state(context)
        if state.live_partitions:
            raise RuntimeError(f'{state.context.key} still has live partitions')
        state.status = ContextStatus.REMOVED
        self._contexts.pop(state.context.key)

    def context_snapshot(self, context: LoraContext | str) -> dict[str, object]:
        state = self._state(context)
        return {
            'context': state.context,
            'status': state.status,
            'policy_version': state.policy.version,
            'adapter_path': state.policy.adapter_path,
            'next_step': state.next_step,
            'completed_partitions': state.completed_partitions,
            'live_partitions': len(state.live_partitions),
            'dataset_exhausted': state.dataset_exhausted,
            'max_steps': state.max_steps,
        }

    def list_context_snapshots(self) -> list[dict[str, object]]:
        return [self.context_snapshot(key) for key in self._contexts]

    def context_adapter_paths(self, context: LoraContext | str) -> list[str]:
        return [
            policy.adapter_path for policy in self._state(context).policy_history if policy.adapter_path is not None
        ]

    def get_rollout_policy(self, context: LoraContext | str) -> RolloutPolicy:
        return self._state(context).policy

    def acquire_rollout_policy(self, context: LoraContext | str) -> RolloutPolicy:
        """Pin the current policy while one sampler request is using it."""
        state = self._state(context)
        policy = state.policy
        if policy.adapter_path is not None:
            references = state.rollout_policy_references
            references[policy.adapter_path] = references.get(policy.adapter_path, 0) + 1
        return policy

    def release_rollout_policy(self, policy: RolloutPolicy) -> None:
        """Release a policy previously returned by :meth:`acquire_rollout_policy`."""
        if policy.adapter_path is None:
            return
        state = self._state(policy.context_key)
        references = state.rollout_policy_references
        count = references.get(policy.adapter_path, 0)
        if count <= 0:
            raise RuntimeError(f'rollout policy was not acquired: {policy.adapter_path}')
        if count == 1:
            references.pop(policy.adapter_path)
        else:
            references[policy.adapter_path] = count - 1

    def request_rollout_partition(self, context: LoraContext | str, *, target_groups: int,
                                  num_generations: int) -> PartitionAdmission | None:
        state = self._state(context)
        if target_groups <= 0 or num_generations <= 0:
            raise ValueError('target_groups and num_generations must be positive')
        if state.status is not ContextStatus.ACTIVE or self._stop_requested:
            return None
        if state.max_steps is not None and state.completed_partitions + len(state.live_partitions) >= state.max_steps:
            state.dataset_exhausted = True
            state.status = ContextStatus.EXHAUSTED
            self._finish_if_drained(state)
            return None
        if self.max_steps is not None and (self.completed_partitions + len(self.list_live_partitions())
                                           >= self.max_steps):
            return None
        oldest_unreleased_step = (
            min(admission.step
                for admission in state.live_partitions.values()) if state.live_partitions else state.next_step)
        if state.next_step - oldest_unreleased_step > self.max_staleness:
            return None
        admission = PartitionAdmission(
            context=state.context,
            partition_id=state.context.partition_id(state.next_step),
            step=state.next_step,
            target_groups=target_groups,
            num_generations=num_generations,
            created_order=self._creation_order,
        )
        self._creation_order += 1
        state.next_step += 1
        state.live_partitions[admission.partition_id] = admission
        return admission

    def list_live_partitions(self) -> list[PartitionAdmission]:
        return sorted(
            (admission for state in self._contexts.values() for admission in state.live_partitions.values()),
            key=lambda admission: admission.created_order,
        )

    def list_trainable_partitions(self) -> list[PartitionAdmission]:
        """Return at most one partition per context, in partition step order."""
        partitions = []
        for state in self._contexts.values():
            if not state.live_partitions:
                continue
            if state.training_partition_id is not None:
                partitions.append(state.live_partitions[state.training_partition_id])
                continue
            partitions.append(min(state.live_partitions.values(), key=lambda admission: admission.step))
        return sorted(partitions, key=lambda admission: admission.created_order)

    def on_dataset_exhausted(self, context: LoraContext | str) -> None:
        state = self._state(context)
        state.dataset_exhausted = True
        if state.status is ContextStatus.ACTIVE:
            state.status = ContextStatus.EXHAUSTED
        self._finish_if_drained(state)

    def on_partition_training_started(self, admission: PartitionAdmission) -> None:
        state = self._state(admission.context)
        if state.training_partition_id not in (None, admission.partition_id):
            raise RuntimeError(f'{state.context.key} already trains {state.training_partition_id}')
        self._require_live(state, admission)
        oldest_partition = min(state.live_partitions.values(), key=lambda candidate: candidate.step)
        if oldest_partition.partition_id != admission.partition_id:
            raise RuntimeError(f'{admission.partition_id} cannot train before {oldest_partition.partition_id}')
        state.training_partition_id = admission.partition_id

    def on_partition_trained(self, admission: PartitionAdmission, *, adapter_path: str) -> RolloutPolicy:
        state = self._state(admission.context)
        self._require_live(state, admission)
        next_policy = RolloutPolicy(state.context.key, state.context.adapter_name, state.policy.version + 1,
                                    adapter_path)
        state.policy = next_policy
        state.policy_history.append(next_policy)
        return next_policy

    def on_partition_cleared(self, admission: PartitionAdmission) -> None:
        state = self._state(admission.context)
        self._require_live(state, admission)
        state.live_partitions.pop(admission.partition_id)
        if state.training_partition_id == admission.partition_id:
            state.training_partition_id = None
        state.completed_partitions += 1
        if state.max_steps is not None and state.completed_partitions >= state.max_steps:
            state.dataset_exhausted = True
            if state.status is ContextStatus.ACTIVE:
                state.status = ContextStatus.EXHAUSTED
        if self.max_steps is not None and self.completed_partitions >= self.max_steps:
            self._stop_requested = True
        self._finish_if_drained(state)

    @property
    def completed_partitions(self) -> int:
        return sum(state.completed_partitions for state in self._contexts.values())

    def get_completed_partitions(self) -> int:
        return self.completed_partitions

    def is_run_finished(self) -> bool:
        if self._stop_requested:
            return not self.list_live_partitions()
        return bool(self._contexts) and all(state.status is ContextStatus.FINISHED and not state.live_partitions
                                            for state in self._contexts.values())

    def is_rollout_admission_closed(self) -> bool:
        """Whether the producer must stop reading new prompts.

        A global train limit closes admission only. Existing live partitions
        remain available to AdvantageWorker and TrainerWorker until drained.
        """
        return self._stop_requested or all(state.status is not ContextStatus.ACTIVE
                                           for state in self._contexts.values())

    def adapter_paths_to_keep(self) -> set[str]:
        paths: set[str] = set()
        for state in self._contexts.values():
            if state.policy.adapter_path is not None:
                paths.add(state.policy.adapter_path)
            paths.update(state.rollout_policy_references)
        return paths

    def context_status(self, context: LoraContext | str) -> ContextStatus:
        return self._state(context).status

    def _finish_if_drained(self, state: _ContextState) -> None:
        if state.dataset_exhausted and not state.live_partitions and state.status is ContextStatus.EXHAUSTED:
            state.status = ContextStatus.FINISHED

    def _state(self, context: LoraContext | str) -> _ContextState:
        key = context if isinstance(context, str) else context.key
        return self._contexts[key]

    @staticmethod
    def _require_live(state: _ContextState, admission: PartitionAdmission) -> None:
        if state.live_partitions.get(admission.partition_id) != admission:
            raise KeyError(f'unknown live partition {admission.partition_id}')
