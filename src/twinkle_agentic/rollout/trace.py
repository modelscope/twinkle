# Copyright (c) ModelScope Contributors. All rights reserved.
"""Where a rollout's trajectories go to be read by a human.

Writing traces is not part of what a rollout is: a rollout produces
trajectories, and whether any of them are also written out is a separate
question with its own policy -- which ones, under what name, holding what. So it
is a collaborator a rollout is handed rather than three options it carries, and
a caller who wants different answers subclasses this instead of editing a
rollout.
"""
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import user_data_get
from twinkle.utils import get_logger
from twinkle_agentic.utils.token_utils import _to_plain

logger = get_logger()


class TraceWriter:
    """One pretty-printed JSON file per selected trajectory, for forensics.

    Args:
        directory: where the files go. Created if it does not exist, at
            construction time, so a misconfigured path fails before a run
            rather than after it.
        should_store: decides whether a trajectory is written at all. Default
            writes every one, which is the right default for a small run and
            the wrong one for a long training job -- that is what this
            narrows.
        is_success: decides the filename prefix, ``ok-`` or ``fail-``, so a
            directory listing separates the two. Default marks everything
            failed, since only the caller knows what the task wanted.

    Both predicates are optional and neither is consulted for anything but the
    trace, which is what keeps a training run independent of them: a predicate
    that raises is reported and treated as "no" rather than allowed to end an
    episode that already finished.
    """

    # Dropped from a trace: tensor-like or oversized fields. Raw token ids,
    # labels and image buffers would multiply the file size without adding
    # anything the rendered ``messages`` does not already say.
    SKIP_KEYS = (
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

    def __init__(
        self,
        directory: str,
        *,
        should_store: Optional[Callable[[Dict[str, Any]], bool]] = None,
        is_success: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        if not directory:
            raise ValueError('TraceWriter needs a directory to write to; omit the '
                             'writer entirely to not trace.')
        self.directory = directory
        self.should_store = should_store
        self.is_success = is_success
        os.makedirs(directory, exist_ok=True)

    # ----------------------------------------------------------------- record

    def record(self, trajectory: Dict[str, Any], *, index: int, success: bool) -> Dict[str, Any]:
        """What gets written for one trajectory. Override to say more.

        ``index`` is its position in the batch, which is how an override
        addresses per-episode state its rollout kept alongside the batch.
        """
        return {
            'trajectory': self.serialize(trajectory),
            'ground_truth': user_data_get(trajectory.get('user_data'), 'ground_truth', '') or '',
            'stop_reason': trajectory.get('stop_reason'),
            'truncated': bool(trajectory.get('truncated')),
            'success': success,
        }

    @classmethod
    def serialize(cls, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """The trajectory minus its tensors: messages and metadata."""
        return _to_plain({k: v for k, v in trajectory.items() if k not in cls.SKIP_KEYS})

    def filename(self,
                 trajectory: Dict[str, Any],
                 *,
                 index: int,
                 success: bool,
                 global_step: Optional[int] = None) -> str:
        """``[step-]{ok|fail}-{id}.json``.

        The id prefers an explicit ``id`` / ``prompt_id`` in ``user_data``,
        sanitised for filesystem safety; failing that it falls back to a
        timestamp, so that concurrent rollouts do not overwrite each other. The
        step prefix, when there is one, lets a listing sort by training step.
        """
        traj_id = ''
        for key in ('id', 'prompt_id'):
            val = user_data_get(trajectory.get('user_data'), key)
            if val not in (None, ''):
                traj_id = re.sub(r'[^A-Za-z0-9_\-.]+', '_', str(val))[:64]
                if traj_id:
                    break
        if not traj_id:
            traj_id = f'{int(time.time() * 1000)}-{index}'
        step_tag = f'step{int(global_step):06d}-' if global_step is not None else ''
        return f'{step_tag}{"ok" if success else "fail"}-{traj_id}.json'

    # ------------------------------------------------------------------ write

    def write(self, trajectories: List[Dict[str, Any]], *, global_step: Optional[int] = None) -> None:
        """Write the selected trajectories. Never raises.

        Observability must not break training: a problem with one trajectory --
        a predicate that raised, a value that would not encode, a full disk --
        is logged and skipped so the remaining dumps and the optimisation loop
        carry on.
        """
        for index, trajectory in enumerate(trajectories):
            try:
                if not self._ask(self.should_store, trajectory, default=True):
                    continue
                success = self._ask(self.is_success, trajectory, default=False)
                record = self.record(trajectory, index=index, success=success)
                path = os.path.join(self.directory,
                                    self.filename(trajectory, index=index, success=success, global_step=global_step))
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(record, f, ensure_ascii=False, indent=2, default=str)
            except Exception as exc:
                logger.warning(f'TraceWriter skipped trajectory {index}: {exc}')

    @staticmethod
    def _ask(predicate: Optional[Callable[[Dict[str, Any]], bool]], trajectory: Dict[str, Any], *,
             default: bool) -> bool:
        """A predicate's answer, or ``default`` when there is none.

        A predicate that raises answers False rather than propagating: it was
        asked about a trajectory that has already been produced, and its opinion
        is not worth losing the rest of the batch's traces over.
        """
        if predicate is None:
            return default
        try:
            return bool(predicate(trajectory))
        except Exception as exc:
            logger.warning(f'TraceWriter predicate {getattr(predicate, "__name__", predicate)!r} '
                           f'raised, reading as False: {exc}')
            return False
