from abc import ABC, abstractmethod

from twinkle.data_format import Trajectory
from twinkle_agentic.data_format import Chunks


class Chunker(ABC):
    """
    TODO: Experimental feature, wait for testing
    """

    @abstractmethod
    def __call__(self, trajectory: Trajectory) -> Chunks:
        raise NotImplementedError
