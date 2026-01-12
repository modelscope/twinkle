# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod

from twinkle.model.base import TwinkleModel
from twinkle.sampler import Sampler


class WeightLoader(ABC):

    @abstractmethod
    def __call__(self, model: TwinkleModel, sampler: Sampler):
        """Sync weights from model to sampler

        Args:
            model: The actor model instance
            sampler: The sampler instance
        """
        ...