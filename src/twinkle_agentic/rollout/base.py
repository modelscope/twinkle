# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import Trajectory, user_data_get
from twinkle.data_format.sampling import SamplingParams
from .bridge import _to_plain


class Rollout(ABC):
    """A batch of trajectories in, the same batch with the model's turns appended.

    Implementations differ in where the turns come from -- a local sampler,
    whose token ids are spliced into the trajectory, or an HTTP endpoint, which
    only ever returns text -- and the difference is real enough that they stay
    separate classes: only one of them produces something trainable.

    Everything that is *not* generation is here: option validation, spreading a
    per-call argument over the batch, and the trace dump. It moved up because
    the two implementations had drifted into sharing it by reaching across the
    class boundary for each other's underscore methods.
    """

    # Set by _init_common. Declared at class level so a subclass that does its
    # own setup still answers these attributes instead of raising from a base
    # method it inherited.
    max_turns: int = 1
    sampling_params: Optional[SamplingParams] = None
    trace_dir: Optional[str] = None
    trace_callback: Optional[Callable[[Dict[str, Any]], bool]] = None
    success_callback: Optional[Callable[[Dict[str, Any]], bool]] = None

    @abstractmethod
    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        raise NotImplementedError()

    # ------------------------------------------------------------------ setup

    def _init_common(
        self,
        *,
        max_turns: int,
        sampling_params: Optional[SamplingParams] = None,
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
        self.max_turns = max_turns
        self.sampling_params = sp
        self.trace_dir = trace_dir
        self.trace_callback = trace_callback
        self.success_callback = success_callback
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

    @staticmethod
    def _broadcast(arg, n: int, *, name: str, required: bool = False) -> List[Any]:
        """One value shared by the batch, or a list already aligned 1:1 with it.

        A list of the wrong length is refused rather than zipped short: the
        mismatch would silently pair trajectories with the wrong tool manager,
        which reads downstream as a model that used the wrong sandbox.
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
        return [arg] * n

    # ------------------------------------------------------------------ trace

    _TRACE_SKIP_KEYS = (
        'input_ids',
        'labels',
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
