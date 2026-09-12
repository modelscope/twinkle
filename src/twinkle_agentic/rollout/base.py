# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from twinkle.data_format import Trajectory
from twinkle.data_format.sampling import SamplingParams
from .trace import TraceWriter

# Termination reasons surfaced via ``trajectory['stop_reason']``. The sampler
# path takes the first three from the sampler itself; the API path has to name
# them, and one vocabulary for both is what lets a consumer read either.
STOP_NO_TOOL = 'stop'
STOP_LENGTH = 'length'
STOP_MAX_TURNS = 'max_turns'
STOP_GENERATION_ERROR = 'generation_error'

# Runaway guard: a ``followup_fn`` is expected to return None eventually. This
# only bounds a callback that never does, so one bad hook cannot spin forever.
MAX_FOLLOWUPS = 20


class Rollout(ABC):
    """A batch of trajectories in, the same batch with the model's turns appended.

    The concrete subclass may source each assistant turn from a local sampler or
    an HTTP endpoint, or not drive the turns at all and let an agent program drive
    them against an endpoint of ours. Everything independent of that choice lives
    here: option validation, spreading a per-call argument over the batch, and
    the thread pool that runs episodes.

    One episode per thread, and a subclass only writes the episode. Both
    backends are latency-bound on something that is not the caller's CPU -- an
    HTTP round trip, a sandbox, a sampler that routes each request to whichever
    worker is free -- so the threads overlap the waiting. Nothing crosses
    between episodes, which is what makes the pool safe and also what the old
    lockstep loop had to give up: there, one slow sandbox round trip held up the
    next generation for every trajectory in the batch.
    """

    # Set by _init_common. Declared at class level so a subclass that does its
    # own setup still answers these attributes instead of raising from a base
    # method it inherited.
    max_turns: int = 1
    sampling_params: Optional[SamplingParams] = None
    tracer: Optional[TraceWriter] = None
    concurrency: Optional[int] = None

    # ------------------------------------------------------------------ setup

    def _init_common(
        self,
        *,
        max_turns: int = 1,
        sampling_params: Optional[SamplingParams] = None,
        concurrency: Optional[int] = None,
        tracer: Optional[TraceWriter] = None,
    ) -> None:
        """Validate and store the options every rollout takes.

        ``max_turns`` bounds a loop this class drives. A subclass that does not
        drive one -- an episode run by an agent program, which stops when it
        decides it is done -- leaves it alone.
        """
        if max_turns < 1:
            raise ValueError(f'max_turns must be >= 1, got {max_turns}')
        sp = sampling_params or SamplingParams()
        if sp.num_samples != 1:
            # n>1 would fork the conversation at turn 1 and there is no defined
            # way to continue a forked trajectory: ask for several rollouts by
            # passing the trajectory several times instead.
            raise ValueError(f'{type(self).__name__} supports num_samples=1 only, '
                             f'got {sp.num_samples}')
        if concurrency is not None and concurrency < 1:
            raise ValueError(f'concurrency must be >= 1 or None, got {concurrency}')
        self.max_turns = max_turns
        self.sampling_params = sp
        # None means one thread per trajectory. A cap below the batch size costs
        # throughput rather than buying safety, so it has to be asked for.
        self.concurrency = concurrency
        self.tracer = tracer

    # ------------------------------------------------------------------- drive

    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        """Run one episode per trajectory and return them in the input order.

        Order is restored from the future map rather than from completion order,
        because callers pair the result with their own list positionally -- a
        GRPO group is a slice of this list.
        """
        if isinstance(trajectories, dict):
            raise TypeError(f'{type(self).__name__}.__call__ expects a List[Trajectory]; '
                            'wrap a single trajectory as [trajectory].')
        trajectories = list(trajectories)
        n = len(trajectories)
        if n == 0:
            return []

        ctx = self._resolve_call(kwargs, n)
        outs: List[Optional[Trajectory]] = [None] * n
        workers = min(n, self.concurrency or n)
        if workers == 1:
            # No pool for a single episode: a thread would only make the
            # traceback of a failing one harder to read.
            outs = [self._run_one(trajectories[i], i, ctx) for i in range(n)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._run_one, trajectories[i], i, ctx): i for i in range(n)}
                for fut in as_completed(futures):
                    outs[futures[fut]] = fut.result()

        result: List[Trajectory] = [o if o is not None else dict(trajectories[i]) for i, o in enumerate(outs)]
        if self.tracer is not None:
            self.tracer.write(result, global_step=kwargs.get('global_step'))
        return result

    @abstractmethod
    def _run_one(self, trajectory: Trajectory, index: int, ctx: Dict[str, Any]) -> Trajectory:
        """One trajectory, start to finish, in its own thread.

        ``ctx`` is whatever ``_resolve_call`` produced; ``index`` is the
        trajectory's position in the batch, which is how per-trajectory entries
        in ``ctx`` are addressed.
        """
        raise NotImplementedError()

    def _resolve_call(self, kwargs: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Fold per-call ``**kwargs`` over the constructor defaults, once.

        Done before the pool starts so a bad argument raises from the caller's
        frame instead of inside n threads, and so ``_broadcast`` runs once
        rather than per episode.
        """
        return {}

    @staticmethod
    def _unpack_followup(followup: Any) -> Tuple[str, Optional[SamplingParams]]:
        """``followup_fn`` may answer with text, or text plus its own budget."""
        if isinstance(followup, tuple):
            text, params = followup
            return text, params
        return followup, None

    @staticmethod
    def _broadcast(arg, n: int, *, name: str, required: bool = False, per_trajectory: bool = False) -> List[Any]:
        """One value shared by the batch, or a list already aligned 1:1 with it.

        A list of the wrong length is refused rather than zipped short: the
        mismatch would silently pair trajectories with the wrong tool manager,
        which reads downstream as a model that used the wrong sandbox.

        ``per_trajectory`` refuses to share one instance across a batch at all.
        It is for arguments that carry episode state: episodes now run in
        parallel threads, so a shared one would have several conversations
        writing to the same object instead of merely interleaving in it.
        """
        if arg is None:
            if required:
                raise ValueError(f'{name} is required but was not provided. '
                                 'Pass it at construction time or as a per-call kwarg.')
            return [None] * n
        if isinstance(arg, list):
            if len(arg) != n:
                raise ValueError(f'per-call {name} list length ({len(arg)}) does '
                                 f'not match number of trajectories ({n})')
            return list(arg)
        if per_trajectory and n > 1:
            raise ValueError(f'{name} holds per-episode state and cannot be shared by '
                             f'{n} trajectories running in parallel threads: pass a list '
                             f'of {n}, one per trajectory.')
        return [arg] * n
