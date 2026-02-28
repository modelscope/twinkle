# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/base.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator, TypedDict

if TYPE_CHECKING:
    import torch


class TensorMeta(TypedDict):
    """Metadata for a tensor in the weight bucket."""
    name: str
    shape: 'torch.Size'
    dtype: 'torch.dtype'
    offset: int


class CheckpointEngine(ABC):
    """Abstract base class for checkpoint engines.

    A checkpoint engine handles weight synchronization between trainer and rollout
    processes. The typical workflow is:

    In trainer process (rank 0):
    >>> engine = CheckpointEngineRegistry.new('nccl', bucket_size=512<<20)
    >>> engine.is_master = True  # set before prepare()
    >>> engine.prepare()
    >>> engine.init_process_group(rank=0, world_size=5, master_metadata=metadata)
    >>> await engine.send_weights(weight_generator())
    >>> engine.finalize()

    In rollout process:
    >>> engine = CheckpointEngineRegistry.new('nccl', bucket_size=512<<20)
    >>> engine.prepare()
    >>> engine.init_process_group(rank=1, world_size=5, master_metadata=metadata)
    >>> async for name, tensor in engine.receive_weights():
    ...     weights.append((name, tensor))
    >>> engine.finalize()
    """

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Prepare the checkpoint engine before weight synchronization.

        This method should:
        1. Allocate weight transfer buffers.
        2. Setup communication channels (e.g., ZMQ sockets).
        3. Return metadata needed for topology building.

        Returns:
            A dictionary containing metadata (e.g., master IP and port).
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build communication topology between trainer and rollout workers.

        This method determines the rank assignment for each worker in the
        temporary NCCL/HCCL process group used for weight synchronization.

        Args:
            trainer_world_size: Number of trainer workers.
            rollout_world_size: Number of rollout workers.
            metadata: List of metadata from all workers' prepare() calls.

        Returns:
            A tuple of (trainer_kwargs, rollout_kwargs), where each dict
            contains lists of arguments to pass to init_process_group().
            Keys typically include: 'rank', 'world_size', 'master_metadata'.
        """
        raise NotImplementedError

    @abstractmethod
    def init_process_group(self, **kwargs):
        """Initialize the process group for weight synchronization.

        Args:
            **kwargs: Arguments from build_topology(), typically including:
                - rank: The rank of this worker in the sync group.
                - world_size: Total number of workers in the sync group.
                - master_metadata: Metadata from the master (trainer rank 0).
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """Finalize the checkpoint engine after weight synchronization.

        This method should:
        1. Free weight transfer buffers.
        2. Destroy the temporary process group (if rebuild_group=True).
        3. Clean up communication channels.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_weights(self, weights: Generator[tuple[str, 'torch.Tensor'], None, None]):
        """Send model weights to rollout workers.

        This method streams weights in buckets to avoid memory issues with
        large models. Only trainer rank 0 actually sends weights; other
        trainer ranks consume the generator without sending.

        Args:
            weights: A generator yielding (name, tensor) pairs.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self) -> AsyncGenerator[tuple[str, 'torch.Tensor'], None]:
        """Receive model weights from trainer.

        This method receives weights in buckets and yields them as they
        become available, enabling streaming weight loading.

        Yields:
            Tuples of (name, tensor) for each weight.
        """
        raise NotImplementedError
