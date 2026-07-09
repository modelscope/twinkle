# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

from twinkle.data_format import Trajectory


class Rollout(ABC):

    @abstractmethod
    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        raise NotImplementedError()
