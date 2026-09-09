# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from twinkle.data_format import Trajectory, user_data_get
from twinkle.data_format.sampling import SamplingParams
from .bridge import _to_plain

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

    The concrete multi-turn loop may source each assistant turn from a local
    sampler or an HTTP endpoint. Everything independent of that choice lives
    here: option validation, spreading a per-call argument over the batch, the
    thread pool that runs episodes, and trace dumping.

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
    trace_dir: Optional[str] = None
    trace_callback: Optional[Callable[[Dict[str, Any]], bool]] = None
    success_callback: Optional[Callable[[Dict[str, Any]], bool]] = None
    concurrency: Optional[int] = None

    # ------------------------------------------------------------------ setup

    def _init_common(
        self,
        *,
        max_turns: int,
        sampling_params: Optional[SamplingParams] = None,
        concurrency: Optional[int] = None,
        trace_dir: Optional[str] = None,
        trace_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        success_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        """Validate and store the options every multi-turn rollout takes."""
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
        self.trace_dir = trace_dir
        self.trace_callback = trace_callback
        self.success_callback = success_callback
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

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
        if self.trace_dir:
            self._write_rollout_traces(result, global_step=kwargs.get('global_step'))
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

    # ------------------------------------------------------------------ trace

    _TRACE_SKIP_KEYS = (
        'input_ids',
        'labels',
        'completion_mask',
        'attention_mask',
        'position_ids',
        'logprobs',
        'pixel_values',
        'image_grid_thw',
        'mm_token_type_ids',
    )

    @classmethod
    def _serialize_for_trace(cls, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Drop tensor-like / oversized fields; keep messages + metadata.

        Trace files are for human forensics; raw token ids, labels and
        image buffers would bloat the file by orders of magnitude without
        adding diagnostic value (the chat-template rendering of
        ``messages`` already captures the textual content).
        """
        slim = {k: v for k, v in traj.items() if k not in cls._TRACE_SKIP_KEYS}
        return _to_plain(slim)

    @staticmethod
    def _extract_ground_truth(traj: Dict[str, Any]) -> str:
        """Pull ``ground_truth`` out of packed ``user_data``."""
        return user_data_get(traj.get('user_data'), 'ground_truth', '') or ''

    @staticmethod
    def _resolve_traj_id(traj: Dict[str, Any], fallback_idx: int) -> str:
        """Stable-ish trajectory id for filenames.

        Prefers an explicit ``id`` / ``prompt_id`` key in ``user_data``
        (sanitised for filesystem safety); else falls back to
        ``{timestamp_ms}-{fallback_idx}`` so concurrent rollouts do not
        overwrite each other's files.
        """
        for key in ('id', 'prompt_id'):
            val = user_data_get(traj.get('user_data'), key)
            if val not in (None, ''):
                safe = re.sub(r'[^A-Za-z0-9_\-.]+', '_', str(val))[:64]
                if safe:
                    return safe
        return f'{int(time.time() * 1000)}-{fallback_idx}'

    def _build_trace_record(
        self,
        traj: Dict[str, Any],
        *,
        idx: int,
        success: bool,
    ) -> Dict[str, Any]:
        """Assemble one trace record. Subclasses override to add fields.

        ``idx`` is the trajectory's position in the rollout output list,
        so subclasses can correlate the record with any per-call state
        they stashed on ``self`` during ``__call__``.
        """
        return {
            'trajectory': self._serialize_for_trace(traj),
            'ground_truth': self._extract_ground_truth(traj),
            'stop_reason': traj.get('stop_reason'),
            'truncated': bool(traj.get('truncated')),
            'success': success,
        }

    def _write_rollout_traces(
        self,
        outs: List[Dict[str, Any]],
        *,
        global_step: Optional[int] = None,
    ) -> None:
        """Dump one pretty-printed JSON file per selected trajectory.

        ``trace_callback`` (if set) decides WHETHER to store;
        ``success_callback`` (if set) decides the filename prefix
        (``ok-`` vs ``fail-``). Defaults: store-all / mark-fail.

        Observability must never break training -- any I/O or encoding
        problem on a single trajectory is swallowed so the remaining
        dumps and the optimisation loop continue unaffected.
        """
        if not self.trace_dir:
            return
        for idx, traj in enumerate(outs):
            try:
                should_store = True
                if self.trace_callback is not None:
                    try:
                        should_store = bool(self.trace_callback(traj))
                    except Exception:
                        should_store = False
                if not should_store:
                    continue

                success = False
                if self.success_callback is not None:
                    try:
                        success = bool(self.success_callback(traj))
                    except Exception:
                        success = False

                record = self._build_trace_record(traj, idx=idx, success=success)
                prefix = 'ok' if success else 'fail'
                # global_step prefix lets file listings sort by training step.
                step_tag = f'step{int(global_step):06d}-' if global_step is not None else ''
                fname = f'{step_tag}{prefix}-{self._resolve_traj_id(traj, idx)}.json'
                path = os.path.join(self.trace_dir, fname)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(record, f, ensure_ascii=False, indent=2, default=str)
            except Exception:
                # Per-trajectory failure never aborts the loop.
                pass
