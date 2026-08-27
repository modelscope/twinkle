# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from datetime import timedelta
from typing import Any, Dict, Literal, Mapping, Optional

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
        from accelerate.utils import InitProcessGroupKwargs

        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self._memory_efficient_init = memory_efficient_init
        # Tensor parallelism state. The actual sharding is done by transformers at load time
        # (see `init_kwargs`); here we only record the degree and reject mesh shapes
        # the current transformers version cannot honor.
        self._tp_size = device_mesh.get_dim_size('tp') if (device_mesh is not None and device_mesh.has_dim('tp')) else 1
        self._tp_enabled = self._tp_size > 1
        if self._tp_enabled:
            self._validate_tp_device_mesh(device_mesh)
        parallelism_config = self._parallelism_config_from_device_mesh(device_mesh)
        fsdp_plugin = self._fsdp_config_from_device_mesh(device_mesh, fsdp_config, memory_efficient_init)

        kwargs_handlers = [
            InitProcessGroupKwargs(
                timeout=timedelta(seconds=int(os.environ.get('TWINKLE_DIST_TIMEOUT_SECONDS', '7200'))))
        ]
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

    def init_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs the model construction call needs for this strategy.

        For TP this is what makes transformers shard the model at load time. accelerate does not
        apply TP itself: `Accelerator.prepare` only checks that the model was already sharded
        (`model.tp_size == parallelism_config.tp_size`) and replicates whatever is left on its own
        device mesh. We also hand transformers accelerate's own device mesh so both sides operate on
        a single mesh -- otherwise the sharded and the replicated params end up on different meshes
        and forward fails with a cross-mesh DTensor error.
        """
        if not self._tp_enabled:
            return {}
        load_kwargs = self._tp_load_config(self._tp_size)
        device_mesh = self.accelerator.torch_device_mesh
        if device_mesh is not None:
            load_kwargs['device_mesh'] = device_mesh
        return load_kwargs

    @staticmethod
    def _tp_load_config(tp_size: int) -> Dict[str, Any]:
        """Version-adaptive native-TP entry for `from_pretrained`.

        transformers >=5.16 shards via `DistributedConfig(tp_size=...)`; <=5.12 via `tp_plan="auto"`
        plus `tp_size`. Tell them apart by whether `DistributedConfig` carries a `tp_size` field.
        """
        try:
            from transformers.distributed import DistributedConfig
            if 'tp_size' in getattr(DistributedConfig, '__dataclass_fields__', {}):
                return {'distributed_config': DistributedConfig(tp_size=tp_size)}
        except ImportError:
            pass
        return {'tp_plan': 'auto', 'tp_size': tp_size}

    @staticmethod
    def _validate_tp_device_mesh(device_mesh) -> None:
        """Fail fast on TP mesh shapes this integration does not yet support.

        v1 of transformers-backend TP handles pure tensor parallelism only (`tp_size ==
        world_size`); composing TP with FSDP/DP is not validated yet. So a TP mesh must not carry
        any other non-trivial parallel dimension.
        """
        offending = {}
        for name in (device_mesh.mesh_dim_names or ()):
            if name == 'tp':
                continue
            if device_mesh.get_dim_size(name) > 1:
                offending[name] = device_mesh.get_dim_size(name)
        if offending:
            raise ValueError('Tensor parallelism in the transformers backend currently supports pure TP only '
                             '(composing TP with FSDP/DP is not validated yet), but the device mesh also has '
                             f'non-trivial dimensions {offending}. Use a mesh where only `tp` > 1.')
        tp_size = device_mesh.get_dim_size('tp')
        if tp_size != device_mesh.world_size:
            raise ValueError(f'Pure TP requires tp_size ({tp_size}) to equal the mesh world size '
                             f'({device_mesh.world_size}).')

    def capture_pre_ep_state_if_needed(self, model, *, enable_ep: bool) -> None:
        return

    def prepare_adapter_config(self, config_or_dir, *, enable_ep: bool):
        return config_or_dir

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

    def offload_to_cpu(self, model, optimizer=None) -> None:
        """Move parameters/buffers (and any optimizer state) to host memory for colocation.

        Under colocation the trainer shares its GPU with a vLLM sampler and the two do not fit at
        once; between a training step and a rollout the trainer steps aside, then :meth:`reload_to_gpu`
        brings it back. Only placement changes -- nothing is discarded. The device moved off is
        recorded so the reverse lands exactly where the params were, and re-calling either direction is
        harmless because ``.to`` on an already-resident tensor is a no-op.
        """
        import torch
        if getattr(self, '_offload_device', None) is None:
            self._offload_device = self.accelerator.device
        self.unwrap_model(model).to('cpu')
        if optimizer is not None:
            _move_optimizer_state(optimizer, torch.device('cpu'))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload_to_gpu(self, model, optimizer=None) -> None:
        """Bring back what :meth:`offload_to_cpu` handed over, to the device it was moved off."""
        device = getattr(self, '_offload_device', None) or self.accelerator.device
        self.unwrap_model(model).to(device)
        if optimizer is not None:
            _move_optimizer_state(optimizer, device)

    def wrap_model(self, model, *args):
        return self.accelerator.prepare(model, *args)

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model, keep_torch_compile=False)

    def load_peft_weights(self, model, adapter_weights: Mapping[str, Any], adapter_name: str) -> None:
        from peft.utils import set_peft_model_state_dict

        set_peft_model_state_dict(model, adapter_weights, adapter_name=adapter_name)

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

    def get_adapter_state_dict(self, model, adapter_name: str) -> dict:
        """Collect only LoRA adapter parameters."""
        from twinkle.utils import torch_util
        unwrapped = self.unwrap_model(model)
        state_dict = {}
        adapter_suffix = f'.{adapter_name}.'
        for name, param in unwrapped.named_parameters():
            if not _is_lora_state_key(name) or adapter_suffix not in name:
                continue
            local = torch_util.to_local_tensor(param)
            state_dict[name] = local.cpu()
            del local
        return state_dict


def _is_lora_state_key(name: str) -> bool:
    return 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name


def _move_optimizer_state(optimizer, device) -> None:
    """Move an optimizer's per-parameter state tensors (Adam moments, etc.) to ``device``.

    Kept separate from the param move because the optimizer holds its state as plain tensors in
    ``optimizer.state`` rather than as module parameters, so ``model.to`` never touches them --
    leaving them on the GPU would defeat the point of offloading during a colocated rollout.
    """
    import torch
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
