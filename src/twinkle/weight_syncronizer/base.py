from twinkle.model.base import TwinkleModel
from twinkle.sampler import Sampler


class WeightSynchronizer:

    def __call__(self, module: TwinkleModel, sampler: Sampler):
        ...