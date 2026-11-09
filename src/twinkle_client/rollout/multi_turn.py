# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-side trainable multi-turn rollout over HTTP."""
from __future__ import annotations
import asyncio
import copy
import contextlib
import threading
import traceback
import uuid
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from twinkle.template.base import Template
from twinkle_agentic.rollout.base import _RolloutExecution, SampleHandle
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout, RolloutSample, trajectory_to_rollout_sample
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.sampler import vLLMSampler
from twinkle_client.operations import SequentialOperations
from ._http_sampler import _HTTPSamplerAdapter

_CALL_OPTIONS = {
    'sampling_params', 'tool_manager', 'adapter_name', 'adapter_uri', 'followup_fn', 'harness', 'global_step'
}
_UNSET = object()
_RESERVED_EXTRAS = {
    'messages', 'sampling_params', 'tool_manager', 'harness', 'adapter_path', 'adapter_uri', 'policy_version',
    'prompt_idx', 'trajectory_idx', 'submission_id'
}


@dataclass(frozen=True)
class _Admission:
    submission_id: str
    prompt_idx: Any
    trajectory_idx: int
    prompt_extras: dict[str, Any] | None
    policy_version: int
    adapter_path: str | None
    prompt: list[dict]

    def matches(self, handle: SampleHandle) -> bool:
        return (
            self.submission_id == handle.submission_id and self.prompt_idx == handle.prompt_idx
            and self.trajectory_idx == handle.trajectory_idx and self.prompt_extras == handle.prompt_extras
            and self.policy_version == handle.policy_version and self.adapter_path == handle.adapter_path
            and self.prompt == handle.prompt and handle.payload is None and handle.require_logprobs)


@dataclass
class _Call:
    execution: _RolloutExecution
    env_ids: tuple[int, ...]
    harness_ids: tuple[int, ...]
    executor: ThreadPoolExecutor
    adapter: _HTTPSamplerAdapter
    envs: tuple[Any, ...]
    harnesses: tuple[Any, ...]
    future: Future | None = None
    business_error: BaseException | None = None
    uncertain_errors: list[BaseException] = field(default_factory=list)

    @property
    def settled(self) -> bool:
        return not self.uncertain_errors


@dataclass
class _StreamingEntry:
    handle: SampleHandle
    call: _Call
    admission: _Admission
    state: str = 'running'
    consumed: bool = False
    cleanup_error: BaseException | None = None


@dataclass
class _SettlementCredential:
    admission: _Admission
    handle_ref: weakref.ReferenceType[SampleHandle]


class _CleanupUnconfirmedError(RuntimeError):
    """The local wait ended without proof that every operation/resource settled."""


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
        self._condition = threading.Condition(self._lock)
        self._streaming: dict[str, _StreamingEntry] = {}
        self._settlement_credentials: dict[str, _SettlementCredential] = {}
        self._admitting = 0
        self._accepting = True

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

    def _borrowed_identities(self, managers, harnesses):
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
        return env_ids, harness_ids, tuple(envs), tuple(bare_harnesses)

    def _run_call(self, call: _Call, trajectories: List[Trajectory], core_kwargs: Dict[str, Any]):
        result = None
        try:
            result = self._core(trajectories, **core_kwargs)
        except BaseException as error:
            call.business_error = error
        if call.adapter.uncertain_error is not None:
            call.uncertain_errors.append(call.adapter.uncertain_error)
        # The worker, not add_done_callback/admit, performs provider checks. Normal
        # synchronous returns and default no-op ensure_idle implementations are valid
        # idle evidence; provider-reported uncertainty remains quarantined.
        for resource in (*call.envs, *call.harnesses):
            ensure_idle = getattr(resource, 'ensure_idle', None)
            if not callable(ensure_idle):
                continue
            try:
                ensure_idle()
            except BaseException as error:
                call.uncertain_errors.append(error)
        if call.business_error is not None:
            for error in call.uncertain_errors:
                call.business_error.add_note(f'resource completion is unconfirmed: {error}')
            raise call.business_error
        if call.uncertain_errors:
            raise self._uncertainty_error(call)
        return result

    @staticmethod
    def _uncertainty_error(call: _Call) -> _CleanupUnconfirmedError | None:
        if call.settled:
            return None
        error = _CleanupUnconfirmedError(
            f'execution/resource completion is unconfirmed: {call.uncertain_errors[0]}')
        error.__cause__ = call.uncertain_errors[0]
        if call.business_error is not None and call.business_error is not call.uncertain_errors[0]:
            error.add_note(f'episode business error was: {call.business_error}')
        return error

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
        env_ids, harness_ids, envs, bare_harnesses = self._borrowed_identities(managers, harnesses)
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
        call = _Call(execution, env_ids, harness_ids, executor, adapter, envs, bare_harnesses)
        with self._lock:
            if any(identity in self._env_borrows for identity in env_ids):
                executor.shutdown(wait=False)
                raise ValueError('an Env is still borrowed by another active call on this rollout')
            if any(identity in self._harness_borrows for identity in harness_ids):
                executor.shutdown(wait=False)
                raise ValueError('a bare harness is still borrowed by another active call on this rollout')
            self._active[id(execution)] = call
            self._env_borrows.update(dict(zip(env_ids, envs, strict=False)))
            self._harness_borrows.update(dict(zip(harness_ids, bare_harnesses, strict=False)))
            try:
                call.future = executor.submit(self._run_call, call, trajectories, core_kwargs)
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
        if not call.settled:
            # Keep the call and resource objects alive. Future.done alone cannot
            # make an uncertain Env/harness eligible for another episode.
            return
        with self._lock:
            self._active.pop(id(call.execution), None)
            for identity in call.env_ids:
                self._env_borrows.pop(identity, None)
            for identity in call.harness_ids:
                self._harness_borrows.pop(identity, None)

    @staticmethod
    def _wait_future(future: Future) -> BaseException | None:
        control_error = None
        while True:
            try:
                future.result()
            except (KeyboardInterrupt, SystemExit) as error:
                if control_error is None:
                    control_error = error
                if future.done():
                    break
            except BaseException:
                break
            else:
                break
        return control_error

    def _remember_credential(self, entry: _StreamingEntry) -> None:
        submission_id = entry.admission.submission_id

        def remove(handle_ref):
            with self._condition:
                current = self._settlement_credentials.get(submission_id)
                if current is not None and current.handle_ref is handle_ref:
                    self._settlement_credentials.pop(submission_id, None)
                    self._condition.notify_all()

        self._settlement_credentials[submission_id] = _SettlementCredential(
            entry.admission, weakref.ref(entry.handle, remove))

    def _entry_for_handle(self, handle: SampleHandle) -> _StreamingEntry:
        entry = self._streaming.get(handle.submission_id)
        if entry is None or entry.handle is not handle or not entry.admission.matches(handle):
            raise RuntimeError('handle is unknown, forged, already consumed, or its admission metadata was modified')
        return entry

    def _finish_collection(self, entry: _StreamingEntry) -> None:
        if entry.call.settled:
            self._streaming.pop(entry.admission.submission_id, None)
            self._remember_credential(entry)
        else:
            entry.state = 'uncertain'
            entry.cleanup_error = self._uncertainty_error(entry.call)
        self._condition.notify_all()

    def _drain_entry(self, entry: _StreamingEntry) -> bool:
        entry.call.execution.cancel()
        control_error = self._wait_future(entry.call.future)
        cleanup_error = self._uncertainty_error(entry.call)
        with self._condition:
            if cleanup_error is None:
                entry.state = 'settled'
                self._streaming.pop(entry.admission.submission_id, None)
                self._remember_credential(entry)
            else:
                entry.state = 'uncertain'
                entry.cleanup_error = cleanup_error
            self._condition.notify_all()
        if control_error is not None:
            raise control_error from cleanup_error
        if cleanup_error is not None:
            raise cleanup_error
        return True

    @staticmethod
    def _raise_wait_control(control_error, cleanup_error=None):
        if control_error is not None:
            raise control_error from cleanup_error
        if cleanup_error is not None:
            raise cleanup_error

    def admit(
        self,
        prompt: list[dict[str, Any]],
        trajectory_idx: int = 0,
        *,
        prompt_idx: Any = 0,
        sampling_params: SamplingParams | dict[str, Any] | None = None,
        prompt_extras: dict[str, Any] | None = None,
        adapter_path: str | None = None,
        policy_version: int = 0,
        tool_manager=_UNSET,
        harness=_UNSET,
    ) -> SampleHandle:
        """Admit one independent episode and return its local handle immediately."""
        if prompt_extras is not None and not isinstance(prompt_extras, dict):
            raise TypeError('prompt_extras must be a dict or None')
        reserved = set(prompt_extras or {}) & _RESERVED_EXTRAS
        if reserved:
            raise ValueError(f'prompt_extras contains reserved fields: {sorted(reserved)}')
        params = self._normalize_sampling_params(
            self.sampling_params if sampling_params is None else sampling_params)
        if params.num_samples != 1:
            raise ValueError(f'admit requires num_samples=1, got {params.num_samples}')
        if not params.logprobs:
            params = replace(params, logprobs=1)
        extras = copy.deepcopy(prompt_extras) if prompt_extras else None
        original_prompt = copy.deepcopy(prompt)
        submission_id = uuid.uuid4().hex
        handle = SampleHandle(
            submission_id=submission_id,
            prompt_idx=prompt_idx,
            trajectory_idx=trajectory_idx,
            prompt_extras=copy.deepcopy(extras),
            require_logprobs=True,
            policy_version=policy_version,
            prompt=copy.deepcopy(original_prompt),
            payload=None,
            adapter_path=adapter_path,
        )
        admission = _Admission(
            submission_id, prompt_idx, trajectory_idx, copy.deepcopy(extras), policy_version, adapter_path,
            copy.deepcopy(original_prompt))
        kwargs: dict[str, Any] = {
            'sampling_params': params,
            'adapter_uri': adapter_path,
        }
        if tool_manager is not _UNSET:
            kwargs['tool_manager'] = tool_manager
        if harness is not _UNSET:
            kwargs['harness'] = harness
        trajectory = {'messages': original_prompt}
        if prompt_extras and 'tools' in prompt_extras:
            trajectory['tools'] = copy.deepcopy(prompt_extras['tools'])
        with self._condition:
            if not self._accepting:
                raise RuntimeError('streaming admission is closed')
            self._admitting += 1
        call = None
        try:
            call = self._start([trajectory], kwargs)
            entry = _StreamingEntry(handle, call, admission)
            with self._condition:
                if submission_id in self._streaming:
                    raise RuntimeError(f'duplicate submission id: {submission_id}')
                self._streaming[submission_id] = entry
        except BaseException as registration_error:
            if call is not None:
                call.execution.cancel()
                control_error = self._wait_future(call.future)
                cleanup_error = self._uncertainty_error(call)
                if cleanup_error is not None:
                    registration_error.add_note(str(cleanup_error))
                if control_error is not None:
                    registration_error.add_note(f'interrupted while draining: {control_error}')
            raise
        finally:
            with self._condition:
                self._admitting -= 1
                self._condition.notify_all()
        return handle

    @property
    def is_idle(self) -> bool:
        """Whether every owned call has a confirmed outcome; no remote probing."""
        with self._condition:
            return self._admitting == 0 and all(
                call.future is not None and call.future.done() and call.settled for call in self._active.values())

    def drain(self) -> None:
        """Close streaming admission, signal every episode, then settle all of them.

        This instance cannot admit more streaming work afterward. User environments
        remain borrowed resources: this method never closes or resets them.
        """
        issues = []
        with self._condition:
            self._accepting = False
            for entry in self._streaming.values():
                entry.call.execution.cancel()
            while self._admitting:
                try:
                    self._condition.wait()
                except (KeyboardInterrupt, SystemExit) as error:
                    issues.append(error)
            entries = list(self._streaming.values())
            for entry in entries:
                entry.call.execution.cancel()
        for entry in entries:
            try:
                if self.stop(entry.admission.submission_id) is not True:
                    raise RuntimeError(f'missing settlement credential: {entry.admission.submission_id}')
            except BaseException as error:
                error.add_note(f'streaming submission={entry.admission.submission_id}')
                issues.append(error)
        if issues:
            raise BaseExceptionGroup('streaming episodes did not drain cleanly', issues)

    @contextlib.contextmanager
    def streaming(self, driver):
        """Own one driver's streaming lifetime, draining episodes before its abort.

        Use a dedicated rollout instance. Cleanup attempts every episode even when
        the shared driver's own teardown failed; primary errors remain primary.
        """
        primary = None
        try:
            yield driver
        except BaseException as error:
            primary = error
            raise
        finally:
            issues = []

            def finish():
                for operation in (self.drain, driver.abort):
                    try:
                        operation()
                    except BaseException as error:
                        issues.append(error)

            cleanup = SequentialOperations()
            try:
                cleanup.run(finish)
            except BaseException as error:
                issues.append(error)
            finally:
                try:
                    cleanup.close()
                except BaseException as error:
                    issues.append(error)
            if primary is not None:
                for error in issues:
                    primary.add_note('streaming cleanup:\n' + ''.join(traceback.format_exception(error)))
            elif issues:
                raise BaseExceptionGroup('streaming cleanup failed', issues)

    def poll(self, handles: list[SampleHandle]) -> list[SampleHandle]:
        """Return locally completed handles without resolving their outcomes."""
        completed = []
        with self._condition:
            for handle in handles:
                entry = self._entry_for_handle(handle)
                if entry.state == 'running' and entry.call.future.done():
                    completed.append(handle)
        return completed

    def collect(self, handle: SampleHandle) -> RolloutSample:
        """Consume one completed episode, including its original failure, at most once."""
        with self._condition:
            entry = self._entry_for_handle(handle)
            if entry.consumed:
                raise RuntimeError(f'submission {handle.submission_id} was already collected')
            if entry.state == 'stopping':
                raise RuntimeError(f'submission {handle.submission_id} is being stopped')
            if not entry.call.future.done():
                raise RuntimeError(f'submission {handle.submission_id} is not ready')
            entry.consumed = True
            entry.state = 'collecting'
        try:
            outputs = entry.call.future.result()
            if len(outputs) != 1:
                raise RuntimeError(f'episode collect expected one trajectory, got {len(outputs)}')
            admission = entry.admission
            extra = copy.deepcopy(admission.prompt_extras) if admission.prompt_extras else {}
            extra.update({
                'trajectory_idx': admission.trajectory_idx,
                'adapter_uri': admission.adapter_path,
                'submission_id': admission.submission_id,
            })
            sample = trajectory_to_rollout_sample(outputs[0], str(admission.prompt_idx), extra)
            sample.policy_version = admission.policy_version
            sample.version_span = 0
            sample.prompt = copy.deepcopy(admission.prompt)
        except BaseException:
            with self._condition:
                self._finish_collection(entry)
            raise
        with self._condition:
            self._finish_collection(entry)
        return sample

    def stop(self, submission_id: str) -> bool:
        """Confirm cleanup only; a settled business failure also returns True."""
        control_error = None
        with self._condition:
            while True:
                entry = self._streaming.get(submission_id)
                if entry is None:
                    credential = self._settlement_credentials.get(submission_id)
                    if credential is None:
                        self._raise_wait_control(control_error)
                        return False
                    handle = credential.handle_ref()
                    if handle is None:
                        self._settlement_credentials.pop(submission_id, None)
                        self._raise_wait_control(control_error)
                        return False
                    if not credential.admission.matches(handle):
                        raise RuntimeError('settled handle admission metadata was modified')
                    self._raise_wait_control(control_error)
                    return True
                if entry.cleanup_error is not None:
                    self._raise_wait_control(control_error, entry.cleanup_error)
                if entry.state in {'stopping', 'collecting'}:
                    try:
                        self._condition.wait()
                    except (KeyboardInterrupt, SystemExit) as error:
                        if control_error is None:
                            control_error = error
                    continue
                entry.state = 'stopping'
                break
        return self._drain_entry(entry)

    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        if isinstance(trajectories, dict):
            raise TypeError('ClientMultiTurnRollout expects List[Trajectory]')
        trajectories = list(trajectories)
        if not trajectories:
            return []
        call = self._start(trajectories, kwargs)
        try:
            return call.future.result()
        except (KeyboardInterrupt, SystemExit) as control_error:
            call.execution.cancel()
            while True:
                try:
                    call.future.result()
                except (KeyboardInterrupt, SystemExit) as drain_error:
                    if call.future.done():
                        # A completed Future may contain a control exception raised by the worker itself.
                        cause = drain_error if drain_error is not control_error else None
                        raise control_error from cause
                    # Repeated interrupts must not abandon a Future that is still running.
                    continue
                except BaseException as drain_error:
                    raise control_error from drain_error
                raise control_error from None

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
