# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/base.py
"""Base classes for checkpoint engine.

CheckpointEngine is an abstraction layer to synchronize weights between
trainer and rollout. It provides unified APIs:
- send_weights: Get named tensors from generator and send them in streaming manner.
- receive_weights: Return a tensor generator that yields named tensors in streaming manner.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, TypedDict

import torch

logger = logging.getLogger(__name__)


class TensorMeta(TypedDict):
    """Metadata for a tensor in the weight bucket."""
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


class CheckpointEngineRegistry:
    """Registry for checkpoint engine backends."""

    _registry: dict[str, type["CheckpointEngine"]] = {}

    @classmethod
    def register(cls, backend: str):
        """Register a checkpoint engine backend.

        Args:
            backend: The backend name (e.g., 'naive', 'nccl', 'hccl').
        """
        def wrapper(engine_cls: type["CheckpointEngine"]):
            cls._registry[backend] = engine_cls
            return engine_cls
        return wrapper

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        """Get the checkpoint engine class by backend name.

        Args:
            backend: The backend name.

        Returns:
            The checkpoint engine class.
        """
        if backend not in cls._registry:
            raise ValueError(f"Checkpoint engine '{backend}' not registered. "
                           f"Available backends: {list(cls._registry.keys())}")
        return cls._registry[backend]

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        """Create a new checkpoint engine instance.

        Args:
            backend: The backend name.
            *args: Positional arguments for the engine constructor.
            **kwargs: Keyword arguments for the engine constructor.

        Returns:
            A new checkpoint engine instance.
        """
        return cls.get(backend)(*args, **kwargs)


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
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send model weights to rollout workers.

        This method streams weights in buckets to avoid memory issues with
        large models. Only trainer rank 0 actually sends weights; other
        trainer ranks consume the generator without sending.

        Args:
            weights: A generator yielding (name, tensor) pairs.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive model weights from trainer.

        This method receives weights in buckets and yields them as they
        become available, enabling streaming weight loading.

        Yields:
            Tuples of (name, tensor) for each weight.
        """
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
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
