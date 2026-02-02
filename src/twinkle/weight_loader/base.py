# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from enum import Enum
from twinkle.model.base import TwinkleModel
from twinkle.sampler import Sampler

class ROLLOUT_MODE(Enum):
    AUTO = 'auto' # Auto detect
    HYBRID = "hybrid" # Rollout engine and training engine(fsdp/megatron) fused in same process
    COLOCATED = "colocated" # Rollout engine colocated with hybrid engine in same ray placement group but in separate process
    STANDALONE = "standalone" # Standalone rollout server with separate GPU resource, disaggregated architecture.

class WeightLoader(ABC):

    @abstractmethod
    def __call__(self, model: TwinkleModel, sampler: Sampler):
        """Sync weights from model to sampler

        Args:
            model: The actor model instance
            sampler: The sampler instance
        """
        ...