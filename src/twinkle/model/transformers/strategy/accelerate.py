# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from datetime import timedelta
from typing import Any, Dict, Literal, Mapping, Optional

from twinkle import DeviceMesh
from twinkle.model.state_dict_utils import collect_local_state, is_saveable_adapter_key
from twinkle.utils import get_logger
from .base import TransformerStrategyBase
from .base import fsdp_pretrained_load_context

logger = get_logger()


class AccelerateStrategy(TransformerStrategyBase):
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

        requested_strategy = fsdp_config.pop('sharding_strategy', None)
        if dp_size > 1:
            # HSDP
            hybrid_strategies = (FSDPShardingStrategy.HYBRID_SHARD, FSDPShardingStrategy._HYBRID_SHARD_ZERO2)
            sharding_strategy = requested_strategy if requested_strategy in hybrid_strategies else \
                FSDPShardingStrategy.HYBRID_SHARD
        else:
            # FSDP
            sharding_strategy = FSDPShardingStrategy.FULL_SHARD
        if requested_strategy is not None and requested_strategy is not sharding_strategy:
            logger.warning(
                f'fsdp_config sharding_strategy={requested_strategy} is overridden with {sharding_strategy} to '
                f'match the device mesh (dp={dp_size}, fsdp={fsdp_size}).')

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

    def offload_to_cpu(self, model, optimizer=None, *, offload_model: bool = True) -> None:
        """Move parameters/buffers and/or optimizer state to host memory for colocation.

        Under colocation the trainer shares its GPU with a vLLM sampler. ``offload_model=False``
        leaves the module resident and moves only ``optimizer`` when one is passed. Only placement
        changes -- nothing is discarded. The device moved off is recorded so the reverse lands
        exactly where the params were. Re-calling either direction is harmless: ``.to`` on an
        already-resident tensor is a no-op, and :meth:`reload_to_gpu` restores only what moved.
        """
        import torch
        from twinkle.model.offload import move_optimizer_state, record_offload
        if not offload_model and optimizer is None:
            record_offload(self, model=False, optimizer=False)
            return
        if getattr(self, '_offload_device', None) is None:
            self._offload_device = self.accelerator.device
        if offload_model:
            self.unwrap_model(model).to('cpu')
        if optimizer is not None:
            move_optimizer_state(optimizer, torch.device('cpu'))
        record_offload(self, model=offload_model, optimizer=optimizer is not None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload_to_gpu(self, model, optimizer=None) -> None:
        """Bring back what :meth:`offload_to_cpu` handed over, to the device it was moved off."""
        from twinkle.model.offload import clear_offload_record, move_optimizer_state, offload_record
        moved_model, moved_optimizer = offload_record(self)
        if not moved_model and not moved_optimizer:
            return
        device = getattr(self, '_offload_device', None) or self.accelerator.device
        if moved_model:
            self.unwrap_model(model).to(device)
        if moved_optimizer and optimizer is not None:
            move_optimizer_state(optimizer, device)
        clear_offload_record(self)

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

    @staticmethod
    def _prepare_full_optimizer_state_dict_options(*, for_load: bool):
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        return StateDictOptions(
            full_state_dict=True,
            cpu_offload=not for_load,
            broadcast_from_rank0=for_load,
        )

    def needs_wrapped_optimizer_state(self) -> bool:
        fsdp_plugin = self._get_fsdp_plugin()
        return fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2

    def save_optimizer_checkpoint(self, model, optimizer, output_path: str, *, param_name_mapping=None):
        import torch

        from twinkle.model.state_dict_utils import remap_optimizer_state_names
        fsdp_plugin = self._get_fsdp_plugin()
        if fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

            # PyTorch initializes Adam state through the native optimizer step.
            # AcceleratedOptimizer may skip that step when gradients are not synchronized.
            inner_optimizer = getattr(optimizer, 'optimizer', optimizer)
            optim_state = get_optimizer_state_dict(
                model,
                inner_optimizer,
                options=self._prepare_full_optimizer_state_dict_options(for_load=False),
            )
            if self.accelerator.process_index == 0:
                remap_optimizer_state_names(optim_state, param_name_mapping or {})
                torch.save(optim_state, output_path)
            return

        if self.accelerator.process_index == 0:
            optim_state = optimizer.state_dict()
            remap_optimizer_state_names(optim_state, param_name_mapping or {})
            torch.save(optim_state, output_path)

    def load_optimizer_checkpoint(self, model, optimizer, input_path: str, *, param_name_mapping=None):
        import torch

        from twinkle.model.state_dict_utils import remap_optimizer_state_names
        fsdp_plugin = self._get_fsdp_plugin()
        if fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

            inner_optimizer = getattr(optimizer, 'optimizer', optimizer)
            optim_state = {}
            if self.accelerator.process_index == 0:
                optim_state = torch.load(input_path, map_location='cpu', weights_only=True)
                remap_optimizer_state_names(optim_state, param_name_mapping or {})
            set_optimizer_state_dict(
                model,
                inner_optimizer,
                optim_state,
                options=self._prepare_full_optimizer_state_dict_options(for_load=True),
            )
            return

        # weights_only=True: optimizer state is tensors + plain containers (verified), so a
        # hostile checkpoint's pickle payload never reaches an unrestricted unpickler.
        optim_state = torch.load(input_path, map_location='cpu', weights_only=True)
        remap_optimizer_state_names(optim_state, param_name_mapping or {})
        optimizer.load_state_dict(optim_state)

    def get_full_state_dict(self, model) -> dict:
        """Collect full state dict, including persistent buffers."""
        from twinkle.utils import torch_util
        unwrapped = self.unwrap_model(model)
        state = collect_local_state(unwrapped, include_buffers=True)
        return {name: torch_util.to_local_tensor(tensor).cpu() for name, tensor in state.items()}

    def load_full_state_dict(self, model, state_dict) -> None:
        """Load a full (non-sharded) state dict into the model in-place.

        Used by full-parameter training to (re)load base weights, e.g. after a
        tenant releases an exclusive full-parameter deployment.
        """
        fsdp_plugin = self._get_fsdp_plugin()
        if fsdp_plugin is not None and fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import set_model_state_dict

            from twinkle.model.state_dict_utils import with_tied_weight_aliases
            set_model_state_dict(
                model,
                with_tied_weight_aliases(self.unwrap_model(model), state_dict),
                options=self._prepare_full_optimizer_state_dict_options(for_load=True),
            )
            return
        unwrapped = self.unwrap_model(model)
        unwrapped.load_state_dict(state_dict, strict=False)

    def get_adapter_state_dict(self, model, adapter_name: str) -> dict:
        """Collect LoRA parameters, trained modules_to_save copies, and their buffers."""
        from twinkle.utils import torch_util
        unwrapped = self.unwrap_model(model)
        adapter_suffix = f'.{adapter_name}.'
        state = collect_local_state(unwrapped, include_buffers=True)
        return {
            name: torch_util.to_local_tensor(tensor).cpu()
            for name, tensor in state.items()
            if is_saveable_adapter_key(name) and adapter_suffix in name
        }


def _move_optimizer_state(optimizer, device) -> None:
    """Move an optimizer's per-parameter state tensors (Adam moments, etc.) to ``device``."""
    from twinkle.model.offload import move_optimizer_state
    move_optimizer_state(optimizer, device)
