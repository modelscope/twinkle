from typing import List

from twinkle.trajectory import Trajectory


class Reward:

    def calculate(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        ...