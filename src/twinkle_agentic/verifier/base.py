from abc import ABC, abstractmethod


class Verifier(ABC):
    """Reward verifier that scores a sample on a 5-level scale (0-4)."""

    NUM_LEVELS = 5

    @abstractmethod
    def __call__(self, trajectory: dict, **kwargs) -> int:
        pass
