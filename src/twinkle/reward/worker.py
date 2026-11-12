# Copyright (c) ModelScope Contributors. All rights reserved.
"""Scorer workers: one component per ORM / PRM, ``worker(trajectories) -> one score each``.

Each worker is its own pool -- the driver hands it trajectory references, the list is spread over
its replicas, and the cluster adapter reads its ``pressure()`` like any other role. ``None`` (or a
NaN) is a score the scorer could not produce, so the caller's nan-aware sums skip it.

- :class:`FunctionWorker` runs a plain callable (a rule, a twinkle ``Reward``, a model handle).
- :class:`ClassifyRMWorker` scores with a pooling sampler's classify head (``sampler.encode``).
- :class:`GenerativeWorker` asks a generative judge and parses the number out of its answer.
"""
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Union

from twinkle import remote_class, remote_function
from twinkle.data_format import PoolingParams, SamplingParams, Trajectory, conversation

__all__ = ['FunctionWorker', 'ClassifyRMWorker', 'GenerativeWorker', 'SCORER_MAX_CONCURRENCY']

#: The default ceiling of concurrent ``__call__`` on one scorer replica.
SCORER_MAX_CONCURRENCY = 64


def _normalized(values: List[Any]) -> List[Optional[float]]:
    return [None if value is None or value != value else float(value) for value in values]


class _Scorer:
    """The call protocol and load report every scorer worker shares; subclasses implement ``_score``."""

    def _score(self, trajectories: List[Trajectory]) -> List[Any]:
        raise NotImplementedError

    @remote_function(enable_continous_work=True, lazy_collect=True)
    def __call__(self, trajectories: List[Trajectory]) -> List[Optional[float]]:
        """One score per trajectory, in order."""
        state = self.__dict__.setdefault('_scorer_inflight', [threading.Lock(), 0])
        with state[0]:
            state[1] += 1
        try:
            values = list(self._score(list(trajectories)))
        finally:
            with state[0]:
                state[1] -= 1
        if len(values) != len(trajectories):
            raise ValueError(f'{type(self).__name__} returned {len(values)} scores for {len(trajectories)} '
                             'trajectories; a scorer must return exactly one score per trajectory, in order.')
        return _normalized(values)

    def _pressure(self) -> float:
        state = self.__dict__.get('_scorer_inflight')
        capacity = (getattr(self, '_max_concurrency', None) or int(os.environ.get('TWINKLE_ACTOR_MAX_CONCURRENCY')
                                                                     or 0) or SCORER_MAX_CONCURRENCY)
        return (state[1] if state else 0) / capacity


@remote_class(max_concurrency=SCORER_MAX_CONCURRENCY)
class FunctionWorker(_Scorer):
    """Run ``fn(trajectories) -> list[float]`` (a rule, a twinkle ``Reward``, any callable scorer)."""

    def __init__(self, fn: Callable[[List[Trajectory]], List[Any]]):
        self.fn = fn

    def _score(self, trajectories: List[Trajectory]) -> List[Any]:
        return self.fn(trajectories)


@remote_class(max_concurrency=SCORER_MAX_CONCURRENCY)
class ClassifyRMWorker(_Scorer):
    """Score with a classifier reward model served by a pooling sampler (``runner='pooling'``).

    The score is entry ``label`` of the classify head's output; the sampler re-encodes each
    conversation with its own template.
    """

    def __init__(self,
                 sampler: Any,
                 pooling_params: Optional[Union[PoolingParams, Dict[str, Any]]] = None,
                 adapter_path: Optional[str] = None,
                 label: int = 0):
        self.sampler = sampler
        self.pooling_params = pooling_params if pooling_params is not None else PoolingParams(task='classify')
        self.adapter_path = adapter_path
        self.label = label

    def _score(self, trajectories: List[Trajectory]) -> List[Any]:
        responses = self.sampler.encode([conversation(trajectory) for trajectory in trajectories],
                                        pooling_params=self.pooling_params,
                                        adapter_path=self.adapter_path)
        return [response.data[self.label] if response.data else None for response in responses]


@remote_class(max_concurrency=SCORER_MAX_CONCURRENCY)
class GenerativeWorker(_Scorer):
    """LLM-as-judge: ``prompt_fn(trajectory)`` builds the judge's trajectory, ``parse_fn(text)`` reads the score."""

    def __init__(self,
                 sampler: Any,
                 prompt_fn: Callable[[Trajectory], Trajectory],
                 parse_fn: Callable[[str], Optional[float]],
                 sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
                 adapter_path: Optional[str] = None):
        self.sampler = sampler
        self.prompt_fn = prompt_fn
        self.parse_fn = parse_fn
        self.sampling_params = sampling_params
        self.adapter_path = adapter_path

    def _score(self, trajectories: List[Trajectory]) -> List[Any]:
        kwargs = {'adapter_path': self.adapter_path} if self.adapter_path is not None else {}
        responses = self.sampler.sample([self.prompt_fn(trajectory) for trajectory in trajectories],
                                        self.sampling_params, **kwargs)
        return [self.parse_fn(response.sequences[0].decoded if response.sequences else '') for response in responses]
