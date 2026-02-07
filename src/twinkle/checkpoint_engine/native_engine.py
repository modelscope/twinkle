from typing import Any, Generator
from .base import CheckpointEngine
import torch


class ColocatedCheckpointEngine(CheckpointEngine):
    """Checkpoint engine for colocated trainer and rollout on same GPU.

    This is a simple pass-through engine that directly shares the weight
    generator between trainer and rollout without network transfer.
    It's used for Hybrid mode where trainer and rollout share the same GPU.

    Usage:
    >>> engine = ColocatedCheckpointEngine(bucket_size=512<<20)
    >>> engine.send_weights(model.get_hf_state_dict())
    >>> for name, tensor in engine.receive_weights():
    ...     weights.append((name, tensor))
    """

    def __init__(self, bucket_size: int, is_master: bool = False, **kwargs) -> None:
        """Initialize the colocated checkpoint engine.

        Args:
            bucket_size: Size of the transfer bucket in bytes (not used but kept for API compatibility).
            is_master: Whether this is the master process (not used).
        """
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.weights = None

    def prepare(self) -> dict[str, Any]:
        """No preparation needed for colocated mode."""
        return {}

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """No topology building needed for colocated mode."""
        return {}, {}

    def init_process_group(self, **kwargs):
        """No process group needed for colocated mode."""
        pass

    def finalize(self):
        """No cleanup needed for colocated mode."""
        self.weights = None

    def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Store the weights generator for later retrieval.

        Note: This is a synchronous method since no network transfer is needed.

        Args:
            weights: A generator yielding (name, tensor) pairs.
        """
        self.weights = weights

    def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Retrieve the stored weights generator.

        Note: This is a synchronous method since no network transfer is needed.

        Yields:
            Tuples of (name, tensor) from the stored generator.
        """
        if self.weights is not None:
            yield from self.weights
            self.weights = None