# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, Literal, Optional

from twinkle import DeviceMesh
from .load_context import fsdp_pretrained_load_context


class AccelerateStrategy:
    """A training strategy that uses `accelerate` to wrap models.

    Args:
        device_mesh: The model device mesh.
        mixed_precision: The mixed precision type.
        ddp_config: Any ddp config passed into accelerate.
        fsdp_config: Any fsdp config passed into accelerate.
    """

    def __init__(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
        ddp_config: Dict[str, Any] = None,
        fsdp_config: Dict[str, Any] = None,
        memory_efficient_init: bool = False,
    ):
        from accelerate import Accelerator

        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self._memory_efficient_init = memory_efficient_init
        parallelism_config = self._parallelism_config_from_device_mesh(device_mesh)
        fsdp_plugin = self._fsdp_config_from_device_mesh(device_mesh, fsdp_config, memory_efficient_init)

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

    def pretrained_load_context(self):
        return fsdp_pretrained_load_context(self._memory_efficient_init and self.device_mesh is not None)

    @staticmethod
    def _parallelism_config_from_device_mesh(device_mesh: DeviceMesh):
        # TODO should test with transformers v5.0
        from accelerate import ParallelismConfig
        if device_mesh is None:
            return None

        dp_size = device_mesh.get_dim_size('dp') if device_mesh.has_dim('dp') else 1
        fsdp_size = device_mesh.get_dim_size('fsdp') if device_mesh.has_dim('fsdp') else 1
        tp_size = device_mesh.get_dim_size('tp') if device_mesh.has_dim('tp') else 1
        cp_size = device_mesh.get_dim_size('cp') if device_mesh.has_dim('cp') else 1
        sp_size = device_mesh.get_dim_size('sp') if device_mesh.has_dim('sp') else 1

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

    def _fsdp_config_from_device_mesh(self, device_mesh: DeviceMesh, fsdp_config: Dict[str, Any],
                                      memory_efficient: bool):
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp import BackwardPrefetch
        from torch.distributed.fsdp import ShardingStrategy as FSDPShardingStrategy

        if device_mesh is None:
            return None

        fsdp_size = device_mesh.get_dim_size('fsdp') if device_mesh.has_dim('fsdp') else 1
        dp_size = device_mesh.get_dim_size('dp') if device_mesh.has_dim('dp') else 1

        if fsdp_size == 1:
            return None

        fsdp_config = fsdp_config or {}

        sharding_strategy = fsdp_config.pop('sharding_strategy', None)
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
            backward_prefetch=fsdp_config.pop('backward_prefetch', BackwardPrefetch.BACKWARD_PRE),
            mixed_precision_policy=self.mixed_precision,
            cpu_offload=fsdp_config.pop('cpu_offload', False),
            activation_checkpointing=fsdp_config.pop('activation_checkpointing', False),
            auto_wrap_policy=fsdp_config.pop('auto_wrap_policy', 'transformer_based_wrap'),  # noqa
            reshard_after_forward=fsdp_config.pop('reshard_after_forward', True),
            cpu_ram_efficient_loading=fsdp_config.pop('cpu_ram_efficient_loading', memory_efficient),
            **fsdp_config,
        )
        return fsdp_plugin

    def wrap_model(self, model, *args):
        return self.accelerator.prepare(model, *args)

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model, keep_torch_compile=False)

    def _get_fsdp_plugin(self):
        state = self.accelerator.state
        return state.fsdp_plugin if hasattr(state, 'fsdp_plugin') else None

    def _prepare_fsdp2_sd_options(self):
        fsdp_plugin = self._get_fsdp_plugin()
        if fsdp_plugin is None or fsdp_plugin.fsdp_version != 2:
            return None

        from torch.distributed.checkpoint.state_dict import StateDictOptions
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        return StateDictOptions(
            full_state_dict=fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT,
            cpu_offload=getattr(fsdp_plugin.state_dict_config, 'offload_to_cpu', False),
            broadcast_from_rank0=getattr(fsdp_plugin.state_dict_config, 'rank0_only', False),
        )

    def needs_wrapped_optimizer_state(self) -> bool:
        fsdp_plugin = self._get_fsdp_plugin()
        return fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2

    def save_optimizer_checkpoint(self, model, optimizer, output_path: str):
        import torch
        fsdp_plugin = self._get_fsdp_plugin()
        if fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

            optim_state = get_optimizer_state_dict(model, optimizer, options=self._prepare_fsdp2_sd_options())
            if self.accelerator.process_index == 0:
                torch.save(optim_state, output_path)
            return

        if self.accelerator.process_index == 0:
            torch.save(optimizer.state_dict(), output_path)

    def load_optimizer_checkpoint(self, model, optimizer, input_path: str):
        import torch
        fsdp_plugin = self._get_fsdp_plugin()
        if fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

            optim_state = None
            rank0_only = getattr(fsdp_plugin.optim_state_dict_config, 'rank0_only', False)
            if self.accelerator.process_index == 0 or not rank0_only:
                optim_state = torch.load(input_path, weights_only=True)
            set_optimizer_state_dict(model, optimizer, optim_state, options=self._prepare_fsdp2_sd_options())
            return

        optimizer.load_state_dict(torch.load(input_path, map_location='cpu', weights_only=False))

    def get_full_state_dict(self, model) -> dict:
        """Collect full state dict."""
        from twinkle.utils import torch_util
        unwrapped = self.unwrap_model(model)
        state_dict = {}
        for name, param in unwrapped.named_parameters():
            local = torch_util.to_local_tensor(param)
            state_dict[name] = local.cpu()
            del local
        return state_dict
