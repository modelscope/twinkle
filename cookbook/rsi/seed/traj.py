# Copyright (c) ModelScope Contributors. All rights reserved.
"""Earlier trajectories as inspiration: pick one, summarise it, ask for a sibling."""
import random
from typing import List, Optional, Sequence

from twinkle.data_format import Trajectory
from twinkle_agentic.summarizer import Summarizer
from twinkle_agentic.utils.message_utils import msg_content_text, normalize_tool_calls
from .base import Seeder

__all__ = ['TrajectorySeeder']


class TrajectorySeeder(Seeder):
    """One earlier episode per round, retold as something to vary from.

    The trajectory is summarised rather than replayed: what carries over should be
    what the episode was about, and a transcript quoted in full would have the
    model copy the moves instead. Tool calls stay in the summary by name, because
    a task built with no tools is not the kind being asked for.

    ``trajectories`` is public and read fresh every round, so a run that appends
    what it just produced -- the tasks a round kept, say -- widens the pool as it
    goes without rebuilding anything.

    Args:
        trajectories: episodes to draw from. Empty means this seeder declines
            every round, which is the same as not passing one at all.
        summarizer: shortens the retold transcript. Without one the whole
            transcript goes over, which is only sensible for short episodes.
        seed_template: overrides what a drawn episode is handed over as.
        rng: draw order, for a reproducible run.
    """

    # 'Different' is the whole point: the same task again trains nothing, and a
    # model handed an example without this reliably reproduces it.
    _seed_template = ('Here is an earlier task:\n\n{seed}\n\nBuild something in the same spirit '
                      'but different -- it may be more involved, or more useful.')

    def __init__(self,
                 trajectories: Sequence[Trajectory] = (),
                 *,
                 summarizer: Optional[Summarizer] = None,
                 seed_template: Optional[str] = None,
                 rng: Optional[random.Random] = None):
        self.trajectories: List[Trajectory] = list(trajectories)
        self.summarizer = summarizer
        self._seed_template = seed_template or self._seed_template
        self.rng = rng or random.Random()

    def __call__(self) -> Optional[str]:
        if not self.trajectories:
            return None
        summary = self._summary(self.rng.choice(self.trajectories))
        # A trajectory with nothing readable in it -- no content, no calls -- would
        # otherwise be handed over as an empty example, which reads as an
        # instruction to build nothing.
        return self._seed_template.format(seed=summary) if summary else None

    def _summary(self, trajectory: Trajectory) -> str:
        """The episode as ``role: what it said and called``, shortened if asked.

        The system turn is dropped: it is the challenger's own instruction, so
        quoting it back describes the machinery instead of the task.
        """
        turns: List[str] = []
        for message in trajectory.get('messages') or []:
            if not isinstance(message, dict):
                continue
            role = message.get('role') or ''
            if role == 'system':
                continue
            parts = [msg_content_text(message).strip()]
            for call in normalize_tool_calls(message) or ():
                fn = call.get('function') or {}
                if isinstance(fn, dict) and fn.get('name'):
                    parts.append(f"calls {fn['name']}({fn.get('arguments') or ''})")
            body = '\n'.join(part for part in parts if part)
            if body:
                turns.append(f'{role}: {body}')
        text = '\n'.join(turns)
        if not text:
            return ''
        return self.summarizer(text) if self.summarizer is not None else text
