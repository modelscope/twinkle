# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, Any, Optional, Literal

from .base import TrainStrategy
from twinkle import DeviceMesh


class AccelerateStrategy(TrainStrategy):
    """A training strategy that uses `accelerate` to wrap models.

    Args:
        device_mesh: The model device mesh.
        mixed_precision: The mixed precision type.
        ddp_config: Any ddp config passed into accelerate.
        fsdp_config: Any fsdp config passed into accelerate.
    """

    def __init__(self,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 ddp_config: Dict[str, Any] = None,
                 fsdp_config: Dict[str, Any] = None,
                 ):
        from accelerate import Accelerator

        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        parallelism_config = self._parallelism_config_from_device_mesh(device_mesh)
        fsdp_plugin = self._fsdp_config_from_device_mesh(device_mesh, fsdp_config)

        kwargs_handlers = []
        if ddp_config is not None:
            from accelerate import DistributedDataParallelKwargs
            ddp_config = DistributedDataParallelKwargs(**ddp_config)
            kwargs_handlers.append(ddp_config)

        self.accelerator = Accelerator(
            parallelism_config=parallelism_config,
            mixed_precision=mixed_precision,
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=kwargs_handlers,
        )

    @staticmethod
    def _parallelism_config_from_device_mesh(device_mesh: DeviceMesh):
        from accelerate import ParallelismConfig
        if device_mesh is None:
            return None

        dp_size = device_mesh.get_dim_size("dp") if device_mesh.has_dim("dp") else 1
        fsdp_size = device_mesh.get_dim_size("fsdp") if device_mesh.has_dim("fsdp") else 1
        tp_size = device_mesh.get_dim_size("tp") if device_mesh.has_dim("tp") else 1
        cp_size = device_mesh.get_dim_size("cp") if device_mesh.has_dim("cp") else 1
        sp_size = device_mesh.get_dim_size("sp") if device_mesh.has_dim("sp") else 1

        if tp_size == 1 and cp_size == 1 and sp_size == 1:
            # Only ddp
            return None

        parallelism_config = ParallelismConfig(
            dp_replicate_size=dp_size,
            dp_shard_size=fsdp_size,
            tp_size=tp_size,
            cp_size=cp_size,
            sp_size=sp_size,
        )

        return parallelism_config

    def _fsdp_config_from_device_mesh(self, device_mesh: DeviceMesh, fsdp_config: Dict[str, Any]):
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp import ShardingStrategy as FSDPShardingStrategy
        from torch.distributed.fsdp import BackwardPrefetch

        if device_mesh is None:
            return None

        fsdp_size = device_mesh.get_dim_size("fsdp") if device_mesh.has_dim("fsdp") else 1
        dp_size = device_mesh.get_dim_size("dp") if device_mesh.has_dim("dp") else 1

        if fsdp_size == 1 and dp_size == 1:
            return None
        
        fsdp_config = fsdp_config or {}

        sharding_strategy = fsdp_config.pop("sharding_strategy", None)
        if dp_size > 1 and fsdp_size > 1:
            # HSDP
            if sharding_strategy not in (FSDPShardingStrategy.HYBRID_SHARD, FSDPShardingStrategy._HYBRID_SHARD_ZERO2):
                sharding_strategy = FSDPShardingStrategy.HYBRID_SHARD
        elif fsdp_size > 1:
            # FSDP
            sharding_strategy = FSDPShardingStrategy.FULL_SHARD
        elif sharding_strategy is None:
            sharding_strategy = FSDPShardingStrategy.NO_SHARD

        fsdp_version = fsdp_config.pop('fsdp_config', 2)
        assert fsdp_version == 2, 'Currently only support fsdp_version = 2'
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=fsdp_version,
            sharding_strategy=sharding_strategy,
            backward_prefetch=fsdp_config.pop("backward_prefetch", BackwardPrefetch.BACKWARD_PRE),
            mixed_precision_policy=self.mixed_precision,
            cpu_offload=fsdp_config.pop("cpu_offload", False),
            activation_checkpointing=fsdp_config.pop('activation_checkpointing', False),
            auto_wrap_policy=fsdp_config.pop('auto_wrap_policy', 'transformer_based_wrap'), # noqa
            reshard_after_forward=fsdp_config.pop('reshard_after_forward', True),
            **fsdp_config,
        )
        return fsdp_plugin

    def wrap_model(self, model, *args):
        return self.accelerator.prepare(model, *args)

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model, keep_torch_compile=False)
