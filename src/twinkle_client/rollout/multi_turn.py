# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-side trainable multi-turn rollout over HTTP."""
from __future__ import annotations
import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle.template.base import Template
from twinkle_agentic.rollout.base import _RolloutExecution
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.sampler import vLLMSampler
from ._http_sampler import _HTTPSamplerAdapter

_CALL_OPTIONS = {
    'sampling_params', 'tool_manager', 'adapter_name', 'adapter_uri', 'followup_fn', 'harness', 'global_step'
}


@dataclass
class _Call:
    execution: _RolloutExecution
    env_ids: tuple[int, ...]
    harness_ids: tuple[int, ...]
    executor: ThreadPoolExecutor
    future: Future | None = None


class ClientMultiTurnRollout:
    """Run the shared multi-turn core with per-call HTTP sampling adapters."""

    def __init__(
        self,
        sampler: vLLMSampler,
        template: Template,
        tool_manager: Optional[ToolManager] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_turns: int = 6,
        max_trajectory_tokens: Optional[int] = None,
        *,
        concurrency: Optional[int] = None,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        stop_after_stuck_turns: int = 0,
        max_malformed_retries: int = 2,
        followup_fn=None,
        harness=None,
        tracer=None,
    ):
        if template is None:
            raise ValueError('ClientMultiTurnRollout requires a local Template instance')
        self.sampler = sampler
        self.template = template
        self.tool_manager = tool_manager
        self.sampling_params = self._normalize_sampling_params(sampling_params or SamplingParams())
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.concurrency = concurrency
        self.adapter_name = adapter_name
        self.adapter_uri = adapter_uri
        self.followup_fn = followup_fn
        self.harness = harness
        self._core = MultiTurnRollout(
            sampler=None,
            template=template,
            tool_manager=tool_manager,
            sampling_params=self.sampling_params,
            max_turns=max_turns,
            max_trajectory_tokens=max_trajectory_tokens,
            concurrency=concurrency,
            response_callback=_HTTPSamplerAdapter(
                sampler, adapter_name=adapter_name, adapter_uri=adapter_uri),
            stop_after_stuck_turns=stop_after_stuck_turns,
            max_malformed_retries=max_malformed_retries,
            followup_fn=followup_fn,
            harness=harness,
            tracer=tracer,
        )
        if hasattr(self._core, '_actors'):
            raise RuntimeError('ClientMultiTurnRollout requires the in-process core, not a Ray actor proxy')
        self._lock = threading.Lock()
        self._active: dict[int, _Call] = {}
        self._env_borrows: dict[int, Any] = {}
        self._harness_borrows: dict[int, Any] = {}

    @staticmethod
    def _normalize_sampling_params(value) -> SamplingParams:
        if isinstance(value, dict):
            return SamplingParams.from_dict(dict(value))
        if isinstance(value, SamplingParams):
            return replace(value)
        raise TypeError('sampling_params must be SamplingParams or a dict')

    @staticmethod
    def _as_batch(value, count: int, name: str) -> list[Any]:
        if isinstance(value, list):
            if len(value) != count:
                raise ValueError(f'{name} list length ({len(value)}) does not match trajectories ({count})')
            return list(value)
        return [value] * count

    @staticmethod
    def _duplicate_ids(values) -> tuple[int, ...]:
        seen: set[int] = set()
        duplicates: list[int] = []
        for value in values:
            if value is None:
                continue
            identity = id(value)
            if identity in seen and identity not in duplicates:
                duplicates.append(identity)
            seen.add(identity)
        return tuple(duplicates)

    def _borrowed_identities(self, managers, harnesses) -> tuple[tuple[int, ...], tuple[int, ...]]:
        envs = []
        for manager in managers:
            bound_envs = getattr(manager, 'bound_envs', None)
            if callable(bound_envs):
                envs.extend(bound_envs())
        bare_harnesses = [harness for harness in harnesses if harness is not None and not hasattr(harness, 'lease')]
        duplicate_envs = self._duplicate_ids(envs)
        duplicate_harnesses = self._duplicate_ids(bare_harnesses)
        if duplicate_envs:
            raise ValueError('the same Env cannot be borrowed by multiple episodes; provide an independent Env per '
                             'trajectory and concurrent call')
        if duplicate_harnesses:
            raise ValueError('the same bare harness cannot be borrowed by multiple episodes; use independent harnesses '
                             'or HarnessLeases')
        env_ids = tuple(dict.fromkeys(id(env) for env in envs))
        harness_ids = tuple(dict.fromkeys(id(harness) for harness in bare_harnesses))
        return env_ids, harness_ids

    def _start(self, trajectories: List[Trajectory], kwargs: Dict[str, Any]) -> _Call:
        unknown = set(kwargs) - _CALL_OPTIONS
        if unknown:
            raise ValueError(f'unknown ClientMultiTurnRollout call options: {sorted(unknown)}')
        count = len(trajectories)
        sampling_params = self._normalize_sampling_params(kwargs.get('sampling_params', self.sampling_params))
        manager_arg = kwargs.get('tool_manager', self.tool_manager)
        harness_arg = kwargs.get('harness', self.harness)
        managers = self._as_batch(manager_arg, count, 'tool_manager')
        harnesses = self._as_batch(harness_arg, count, 'harness')
        env_ids, harness_ids = self._borrowed_identities(managers, harnesses)
        adapter_name = kwargs.get('adapter_name', self.adapter_name)
        adapter_uri = kwargs.get('adapter_uri', self.adapter_uri)
        execution = _RolloutExecution()
        adapter = _HTTPSamplerAdapter(self.sampler, adapter_name=adapter_name, adapter_uri=adapter_uri)
        core_kwargs = {
            'sampling_params': sampling_params,
            'response_callback': adapter,
            'tool_manager': manager_arg,
            'adapter_path': adapter_uri,
            'followup_fn': kwargs.get('followup_fn', self.followup_fn),
            'harness': harness_arg,
            'global_step': kwargs.get('global_step'),
            '_execution': execution,
        }
        executor = ThreadPoolExecutor(max_workers=1)
        call = _Call(execution, env_ids, harness_ids, executor)
        with self._lock:
            if any(identity in self._env_borrows for identity in env_ids):
                executor.shutdown(wait=False)
                raise ValueError('an Env is still borrowed by another active call on this rollout')
            if any(identity in self._harness_borrows for identity in harness_ids):
                executor.shutdown(wait=False)
                raise ValueError('a bare harness is still borrowed by another active call on this rollout')
            self._active[id(execution)] = call
            self._env_borrows.update({identity: env for identity, env in zip(env_ids, env_ids)})
            self._harness_borrows.update({identity: harness for identity, harness in zip(harness_ids, harness_ids)})
            try:
                call.future = executor.submit(self._core, trajectories, **core_kwargs)
            except BaseException:
                self._active.pop(id(execution), None)
                for identity in env_ids:
                    self._env_borrows.pop(identity, None)
                for identity in harness_ids:
                    self._harness_borrows.pop(identity, None)
                executor.shutdown(wait=False)
                raise
        call.future.add_done_callback(lambda _future: self._finish(call))
        return call

    def _finish(self, call: _Call) -> None:
        call.executor.shutdown(wait=False)
        with self._lock:
            self._active.pop(id(call.execution), None)
            for identity in call.env_ids:
                self._env_borrows.pop(identity, None)
            for identity in call.harness_ids:
                self._harness_borrows.pop(identity, None)

    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError('ClientMultiTurnRollout expects List[Trajectory]')
        trajectories = list(trajectories)
        if not trajectories:
            return []
        call = self._start(trajectories, kwargs)
        try:
            return call.future.result()
        except (KeyboardInterrupt, SystemExit):
            call.execution.cancel()
            try:
                call.future.result()
            except BaseException:
                pass
            raise

    async def acall(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError('ClientMultiTurnRollout expects List[Trajectory]')
        trajectories = list(trajectories)
        if not trajectories:
            return []
        call = self._start(trajectories, kwargs)
        wrapped = asyncio.wrap_future(call.future)
        task = asyncio.current_task()
        cancel_error: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(wrapped)
                if cancel_error is not None:
                    raise cancel_error
                return result
            except asyncio.CancelledError as error:
                externally_cancelled = task is not None and task.cancelling() > 0
                if cancel_error is None and externally_cancelled:
                    cancel_error = error
                if wrapped.done():
                    if cancel_error is not None:
                        try:
                            wrapped.result()
                        except BaseException as drain_error:
                            raise cancel_error from drain_error
                        raise cancel_error from None
                    raise
                if cancel_error is None:
                    cancel_error = error
                call.execution.cancel()
            except BaseException as error:
                if cancel_error is not None:
                    raise cancel_error from error
                raise

    def cancel(self) -> None:
        """Stop subsequent operations in calls active at this instant."""
        with self._lock:
            executions = [call.execution for call in self._active.values()]
        for execution in executions:
            execution.cancel()
