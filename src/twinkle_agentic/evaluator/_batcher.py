"""A single-worker micro-batcher for structurally compatible samplers."""

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Condition, Thread
from time import monotonic
from typing import Any, Hashable, Mapping

from twinkle.data_format import SamplingParams, Trajectory

from ._contracts import BackendContractError, SamplerBatchError


def _freeze(value: Any) -> Hashable:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f'Cannot safely batch unhashable value of type {type(value).__name__}') from exc
    return value


def _params_key(params: SamplingParams) -> Hashable:
    return _freeze(vars(params))


@dataclass
class _BatchRequest:
    trajectory: Trajectory
    sampling_params: SamplingParams
    sampler_kwargs: Mapping[str, Any]
    compatibility_key: Hashable
    future: Future
    request_id: int


class SamplerBatcher:
    """Serialize sampler calls while coalescing equal requests into batches."""

    def __init__(self, sampler: Any, *, batch_size: int, batch_wait_ms: float, sampler_kwargs: Mapping[str, Any]):
        self._sampler = sampler
        self._batch_size = batch_size
        self._batch_wait_seconds = batch_wait_ms / 1000
        self._sampler_kwargs = dict(sampler_kwargs)
        self._queue: deque[_BatchRequest] = deque()
        self._condition = Condition()
        self._closed = False
        self._request_id = 0
        self._worker = Thread(target=self._run, name='twinkle-evaluator-sampler-batcher', daemon=False)
        self._worker.start()

    def submit(self, trajectory: Trajectory, sampling_params: SamplingParams) -> Any:
        future: Future = Future()
        key = (_params_key(sampling_params), _freeze(self._sampler_kwargs))
        with self._condition:
            if self._closed:
                raise RuntimeError('Sampler batcher is closed')
            request = _BatchRequest(
                trajectory=trajectory,
                sampling_params=sampling_params,
                sampler_kwargs=self._sampler_kwargs,
                compatibility_key=key,
                future=future,
                request_id=self._request_id,
            )
            self._request_id += 1
            self._queue.append(request)
            self._condition.notify()
        return future.result()

    def _pop_first(self) -> _BatchRequest | None:
        return self._queue.popleft() if self._queue else None

    def _take_compatible(self, key: Hashable, capacity: int) -> list[_BatchRequest]:
        selected: list[_BatchRequest] = []
        retained: deque[_BatchRequest] = deque()
        while self._queue:
            request = self._queue.popleft()
            if request.compatibility_key == key and len(selected) < capacity:
                selected.append(request)
            else:
                retained.append(request)
        self._queue = retained
        return selected

    def _minimum_physical_batch_size(self) -> int:
        mesh = getattr(self._sampler, 'device_mesh', None)
        for name in ('data_world_size', 'dp_world_size'):
            value = getattr(mesh, name, None)
            if isinstance(value, int) and value > 0:
                return value
        return 1

    def _complete_batch(self, requests: list[_BatchRequest]) -> None:
        inputs = [request.trajectory for request in requests]
        physical_size = max(len(inputs), self._minimum_physical_batch_size())
        physical_inputs = inputs + [inputs[-1]] * (physical_size - len(inputs))
        try:
            responses = list(self._sampler.sample(
                physical_inputs,
                sampling_params=requests[0].sampling_params,
                **requests[0].sampler_kwargs,
            ))
            if len(responses) != physical_size:
                raise BackendContractError(
                    f'Sampler returned {len(responses)} responses for physical batch size {physical_size}')
        except Exception as exc:
            error = SamplerBatchError(f'Sampler batch for {len(requests)} request(s) failed: {exc}')
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(error)
            return
        for request, response in zip(requests, responses):
            if not request.future.done():
                request.future.set_result(response)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed and not self._queue:
                    return
                first = self._pop_first()
                assert first is not None
                deadline = monotonic() + self._batch_wait_seconds
                selected = [first]
                selected.extend(self._take_compatible(first.compatibility_key, self._batch_size - len(selected)))
                while len(selected) < self._batch_size and not self._closed:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                    selected.extend(self._take_compatible(first.compatibility_key, self._batch_size - len(selected)))
                    if len(selected) > self._batch_size:
                        overflow = selected[self._batch_size:]
                        selected = selected[:self._batch_size]
                        self._queue.extendleft(reversed(overflow))
            self._complete_batch(selected)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            while self._queue:
                request = self._queue.popleft()
                if not request.future.done():
                    request.future.set_exception(RuntimeError('Sampler batcher was closed'))
            self._condition.notify_all()
        self._worker.join()
