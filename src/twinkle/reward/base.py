# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List

from twinkle.data_format import Trajectory


class Reward:

    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        ...