from abc import ABC, abstractmethod

from twinkle_agentic.data_format import Chunks


class Condenser(ABC):
    """
    TODO: Experimental feature, wait for testing
    """

    @abstractmethod
    def __call__(self, chunks: Chunks, **kwargs) -> Chunks:
        raise NotImplementedError
