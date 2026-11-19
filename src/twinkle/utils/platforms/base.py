# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from abc import ABC
from typing import Any, Dict, List, Tuple, Type


class Platform(ABC):

    @staticmethod
    def visible_device_env(platform: str = None) -> str:
        return Platform.get_platform(platform).visible_device_env()

    @staticmethod
    def device_prefix(platform: str = None) -> str:
        return Platform.get_platform(platform).device_prefix()

    @staticmethod
    def get_platform_names() -> List[str]:
        return ['GPU', 'NPU', 'XPU', 'MXC', 'MPS']

    @staticmethod
    def get_platform(platform: str = None) -> Type['Platform']:
        if platform is None:
            from .mps import is_mps_available
            if shutil.which('npu-smi'):
                from .npu import NPU, ensure_npu_backend
                ensure_npu_backend()
                return NPU
            elif shutil.which('xpu-smi'):
                from .xpu import XPU, ensure_xpu_compat
                ensure_xpu_compat()
                return XPU
            elif shutil.which('mx-smi'):
                from .metax import Metax
                return Metax
            elif shutil.which('nvidia-smi'):
                from .gpu import GPU
                return GPU
            elif is_mps_available():
                from .mps import MPS
                return MPS
            else:
                from .gpu import GPU
                return GPU
        elif platform.upper() in ('GPU', 'CUDA'):
            from .gpu import GPU
            return GPU
        elif platform.upper() in ('MXC', 'METAX'):
            from .metax import Metax
            return Metax
        elif platform.upper() == 'XPU':
            from .xpu import XPU, ensure_xpu_compat
            ensure_xpu_compat()
            return XPU
        elif platform.upper() == 'NPU':
            from .npu import NPU, ensure_npu_backend
            ensure_npu_backend()
            return NPU
        elif platform.upper() == 'MPS':
            from .mps import MPS
            return MPS
        else:
            raise ValueError(f'Unsupported platform: {platform}.')

    @staticmethod
    def get_rank() -> int:
        """Get the global rank"""
        return int(os.getenv('RANK', -1))

    @staticmethod
    def get_local_rank() -> int:
        """Get the local rank"""
        return int(os.getenv('LOCAL_RANK', -1))

    @staticmethod
    def get_world_size() -> int:
        """Get the world size"""
        return int(os.getenv('WORLD_SIZE') or os.getenv('_PATCH_WORLD_SIZE') or 1)

    @staticmethod
    def get_local_world_size() -> int:
        """Get the local world size"""
        return int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))

    @staticmethod
    def get_nnodes() -> int:
        """Get the node count"""
        return int(os.getenv('NNODES', 1))

    @staticmethod
    def get_node_rank() -> int:
        """Get the current node rank"""
        return int(os.getenv('NODE_RANK', 0))

    @staticmethod
    def is_local_master() -> bool:
        """Get if current is the local master"""
        local_rank = Platform.get_local_rank()
        return local_rank in {-1, 0}

    @staticmethod
    def is_master() -> bool:
        """Get if current is the global master"""
        rank = Platform.get_rank()
        return rank in {-1, 0}

    @staticmethod
    def is_last_rank() -> bool:
        """Get if current is the last rank"""
        rank = Platform.get_rank()
        world_size = Platform.get_world_size()
        return rank in {-1, world_size - 1}

    @staticmethod
    def get_peer_index(target_size, rank=None, world_size=None):
        if rank is None:
            rank = Platform.get_rank()
        if rank < 0:
            rank = 0
        if world_size is None:
            world_size = Platform.get_world_size()
        if world_size <= 0:
            world_size = 1

        k, m = divmod(target_size, world_size)
        start_idx = rank * k + min(rank, m)
        end_idx = (rank + 1) * k + min(rank + 1, m)
        if target_size < world_size:
            start_idx = rank % target_size
            end_idx = start_idx + 1

        return slice(start_idx, end_idx)

    @staticmethod
    def get_local_device(idx: int = None, *, platform: str = None):
        platform = Platform.get_platform(platform)
        if idx is None:
            idx = Platform.get_local_rank()
        if idx < 0:
            idx = 0
        return platform.get_local_device(idx)

    @staticmethod
    def device_backend(platform: str = None):
        platform = Platform.get_platform(platform)
        return platform.device_backend()

    @staticmethod
    def get_vllm_device_uuid(device_id: int = 0, platform=None) -> str:
        platform = Platform.get_platform(platform)
        return platform.get_vllm_device_uuid(device_id)

    # ── Backend capability queries ────────────────────────────────────────

    @staticmethod
    def bootstrap() -> None:
        """One-shot runtime fixups, run when ``twinkle.utils.platforms`` is imported."""

    @staticmethod
    def requires_unbound_default_pg() -> bool:
        """Whether the default process group's ``bound_device_id`` must be cleared.

        Queried once right after ``init_process_group``.  ``True`` for backends
        that cannot carry a device binding into later subgroup creation (see the
        NPU/XPU overrides for the concrete failure modes).
        """
        return False

    @staticmethod
    def requires_unbound_pg_for_subgroups() -> bool:
        """Whether the binding must be cleared again right before ``new_group``.

        Deliberately separate from ``requires_unbound_default_pg``: that one is
        about the state left behind by ``init_process_group``, this one is about
        a default PG created by someone else (eager init outside twinkle) that
        is about to be split into model-parallel subgroups.  ``False`` means
        "no extra guard needed here" and asserts nothing about the backend.
        """
        return False

    @staticmethod
    def binds_device_id_to_process_group() -> bool:
        """Whether ``init_process_group`` should receive ``device_id=``.

        True where the device backend builds its communicator lazily on the
        bound device (CUDA/NCCL and NPU/HCCL both benefit); False where the
        bound id propagates into subgroup creation and breaks it (see
        ``requires_unbound_default_pg`` and its platform overrides).
        """
        return True

    @staticmethod
    def ensures_socket_env() -> bool:
        """Whether the platform's communication library needs deterministic
        socket-port env injected before a process group (or a stateless vLLM PG)
        is created on this host -- multi-job collisions otherwise fail the bind
        (e.g. HCCL's fixed base port).  The injection itself lives next to the
        platform (``ensure_hccl_socket_env``); the flag only decides whether to run it.
        """
        return False

    @staticmethod
    def communicator_backend() -> str:
        """The ``twinkle.comm`` backend name standalone process groups use on this platform."""
        raise NotImplementedError('This platform does not declare a communicator backend')

    @staticmethod
    def ray_resources() -> Dict[str, float]:
        """Accelerator resources that must be registered with ``ray.init`` explicitly.

        Empty when Ray can autodetect the devices itself, which is the case
        for CUDA GPUs.
        """
        return {}

    @staticmethod
    def checkpoint_engine() -> Tuple[Type[Any], Dict[str, Any]]:
        """The weight-sync engine class for this platform plus its default kwargs.

        Returning the kwargs here keeps platform-specific engine options (such
        as NPU's communicator reuse) in a single place instead of duplicating
        them in every construction site.
        """
        raise NotImplementedError('This platform has no checkpoint engine implementation')

    @staticmethod
    def ensure_vllm_patch() -> None:
        """Platform no-op hook; overridden by platforms whose vLLM needs explicit fixes."""

    @staticmethod
    def device_module():
        """The ``torch.<x>`` namespace of this platform's accelerator, or ``None``.

        The device-API family (``current_device`` / ``set_device`` / ``empty_cache`` /
        ``synchronize`` / ``ipc_collect`` / ``manual_seed_all``) dispatches through this
        so generic components never branch on hardware names. ``None`` (the base and
        MPS default) means "no accelerator" and callers fall back to CPU / no-op.
        """
        return None

    @staticmethod
    def apply_determinism() -> None:
        """Platform env/backend switches set when deterministic training is requested."""

    @staticmethod
    def supports_device_ipc() -> bool:
        """Whether this platform's accelerator supports device-IPC weight transfer.

        False makes the IPC engines fall back to shared-memory buckets.
        """
        return True

    @staticmethod
    def ensure_ipc_reducer() -> None:
        """Register the tensor reducer ``reduce_tensor`` needs before IPC use."""

    @staticmethod
    def requires_gloo_subgroups() -> bool:
        """Whether Megatron must create Gloo process groups alongside its device groups."""
        return False

    @staticmethod
    def synthesizes_causal_mask() -> bool:
        """Whether the platform's attention backend builds the causal mask itself.

        True lets the processor drop the dense 4D mask it would otherwise send.
        """
        return False

    @staticmethod
    def packs_padded_cu_seqlens() -> bool:
        """Whether the backend's PackedSeqParams consumer requires the padded cu_seqlens fields."""
        return False

    @staticmethod
    def aligns_param_dtype_to_base() -> bool:
        """Whether every model parameter must share the base dtype.

        True forces the load-time dtype alignment walk (FSDP2 requires matching
        dtypes on NPU); False keeps only LoRA parameters aligned, which is the
        backend-agnostic requirement.
        """
        return False

    @staticmethod
    def configure_model_config(config: Any) -> None:
        """Mutate a Megatron ``ModelConfig`` in place before model construction.

        Platforms whose mcore stack needs specific flags (e.g. MindSpeed's patched
        TE attention) apply them here; the default is no-op.
        """

    @staticmethod
    def resolve_object_collective_group(device_mesh: Any):
        """The process group Python object collectives should use, or ``None``.

        ``None`` (the default) keeps the group the caller passed in; platforms
        whose device backend cannot carry ``all_gather_object`` (HCCL) return a
        Gloo group instead. Only consulted when torch.distributed is initialized.
        """
        return None

    @staticmethod
    def apply_vllm_patch(platform: str = None) -> None:
        """Apply the current platform's explicit vLLM fixes, once per process.

        Call this before the first ``vllm`` import in a process. Platforms whose vLLM
        support ships inside a companion wheel (``vllm_metax``, ...) apply their patch
        here; platforms that register through vLLM's plugin mechanism (``vllm_ascend``)
        need nothing and stay no-op.
        """
        Platform.get_platform(platform).ensure_vllm_patch()

    @staticmethod
    def get_device_rng_state(platform: str = None):
        """Return device-specific RNG state (e.g. CUDA / NPU / MPS).

        Backend-agnostic replacement for hard-coded ``torch.cuda.get_rng_state()``.
        Returns ``None`` when no accelerator is available, so callers can safely
        skip persistence on CPU-only or unsupported devices.
        """
        return Platform.get_platform(platform).get_device_rng_state()

    @staticmethod
    def set_device_rng_state(state, *, platform: str = None) -> None:
        """Restore device-specific RNG state.

        No-op when ``state`` is ``None`` (e.g. checkpoint produced on a different
        backend) or when the current platform has no accelerator available.
        """
        if state is None:
            return
        Platform.get_platform(platform).set_device_rng_state(state)
