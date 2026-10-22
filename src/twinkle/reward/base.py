# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, List

from twinkle.data_format import Trajectory


class Reward:
    """Score completed trajectories: ``reward(trajectories, **kwargs) -> one float each."""

    def __call__(self, trajectories: List[Trajectory], **kwargs: Any) -> List[float]:
        ...
