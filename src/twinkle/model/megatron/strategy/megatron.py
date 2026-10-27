# Copyright (c) ModelScope Contributors. All rights reserved.
import contextlib
import gc

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from typing import Any, Dict, List, Literal, Optional

from twinkle import DeviceMesh, Platform, torch_util
from twinkle.utils import get_logger

logger = get_logger()


def finalize_model_grads_for_lora(model, *args, **kwargs):
    """Enter Megatron native finalize only when the wrapped model really implements
    ``finish_grad_sync()`` (gated by runtime capability, not config metadata)."""
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed import finalize_model_grads as _native_finalize_model_grads
    from peft import PeftModel as _PeftModel

    def _get_base_model(m):
        if isinstance(m, _PeftModel):
            return _get_base_model(m.base_model.model)
        return m

    base_model = _get_base_model(model[0])
    if isinstance(base_model, MegatronDDP) or hasattr(base_model, 'finish_grad_sync'):
        return _native_finalize_model_grads(model, *args, **kwargs)
    return None


# transformer_engine exposes no per-call FlashAttention version selector; pinning asserts the
# requested build is installed and switches the other availability flags OFF.
_FLASH_PIN_FLAGS = {2: 'is_installed', 3: 'v3_is_installed', 4: 'v4_is_installed'}
_FLASH_PIN_PREFIXES = ('flash_attention_', 'flash_')


def _flash_version_pin(attn_impl: Optional[str]) -> Optional[int]:
    """The FlashAttention major version ``attn_impl`` pins (accepts both 'flash_3' and
        'flash_attention_3' spellings); unversioned names -> None."""
    if attn_impl is None:
        return None
    key = str(attn_impl).lower()
    for prefix in _FLASH_PIN_PREFIXES:
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return int(key[len(prefix):])
    return None


def _apply_flash_version_pin(attn_impl: Optional[str]) -> Optional[int]:
    """Force transformer_engine to use exactly the pinned FlashAttention version by switching
        its module-level flags. MUST run in the model-building process (a driver-side call
        would not reach a Ray worker)."""
    version = _flash_version_pin(attn_impl)
    if version is None:
        return None
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils as fa_utils

    if version not in _FLASH_PIN_FLAGS:
        raise NotImplementedError(
            f'attn_impl={attn_impl!r} pins FlashAttention v{version}, which transformer_engine has no '
            f'availability flag for. Supported: {sorted(_FLASH_PIN_FLAGS)}.')
    # Only flags this TE build actually defines: v4_is_installed is absent on older versions, and a
    # bare getattr would surface as AttributeError instead of an actionable message.
    present = {v: flag for v, flag in _FLASH_PIN_FLAGS.items() if hasattr(fa_utils, flag)}
    if version not in present:
        raise ValueError(f'attn_impl={attn_impl!r} requests flash-attn v{version}, but the installed '
                         f'transformer_engine has no {_FLASH_PIN_FLAGS[version]!r} flag, so this version cannot be '
                         f'selected. Versions this TE build can pin: {sorted(present)}.')
    installed = {v: getattr(fa_utils, flag) for v, flag in present.items()}
    if not installed[version]:
        raise ValueError(f'attn_impl={attn_impl!r} requests flash-attn v{version}, which is not installed. '
                         f'Detected: ' + ', '.join(f'FA{v}={state}' for v, state in sorted(installed.items())))
    for other, flag in present.items():
        if other != version:
            setattr(fa_utils, flag, False)
    return version


class MegatronStrategy:

    #: ddp_config keys that select Megatron-FSDP; ``use_custom_fsdp`` is megatron's own
    #: deprecated alias for ``use_megatron_fsdp``.
    _FSDP_KEYS = ('use_megatron_fsdp', 'use_custom_fsdp')

    @classmethod
    def uses_fsdp(cls, ddp_config: Optional[Dict[str, Any]]) -> bool:
        """Whether this ddp_config asks for Megatron-FSDP; public because the wrapper-class,
        checkpoint-sharding and optimizer wiring all read it before a strategy exists."""
        ddp_config = ddp_config or {}
        return any(ddp_config.get(key) for key in cls._FSDP_KEYS)

    def __init__(
        self,
        model_dir,
        device_mesh: Optional[DeviceMesh] = None,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        seed: int = 42,
        variable_seq_lengths: bool = True,
        config: PretrainedConfig = None,
        ddp_config: Dict[str, Any] = None,
        bridge_backend='mcore-bridge',
        align_grad_reduce: bool = True,
        nccl_comm_warmup: bool = False,
        attn_impl: Optional[str] = None,
        **kwargs,
    ):
        from megatron.core import mpu
        self.device_mesh = device_mesh
        self.use_distributed_optimizer = use_distributed_optimizer
        self.mixed_precision = mixed_precision
        self.model_dir = model_dir
        self.seed = seed
        self.variable_seq_lengths = variable_seq_lengths
        self.ddp_config = ddp_config or {}
        # The bridge backend owns model construction. Accepted as a NAME string or a live
        # instance: only the string survives the Ray worker boundary.
        if isinstance(bridge_backend, str):
            from ..bridge import resolve_bridge_backend
            self._backend = resolve_bridge_backend(bridge_backend)
        else:
            self._backend = bridge_backend
        self._align_grad_reduce = align_grad_reduce
        # FlashAttention pin flips transformer_engine module globals, so it must run in this
        # (the model-building) process; attn_impl is consumed here, never forwarded into the config.
        pinned = _apply_flash_version_pin(attn_impl)
        if pinned is not None:
            logger.info(f'Forcing Flash Attention v{pinned} as the attention backend.')
        if config is None:
            from transformers import AutoConfig
            self.hf_config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
        else:
            self.hf_config = config
        num_experts = getattr(self.hf_config, 'num_experts', getattr(self.hf_config, 'num_local_experts', None))
        if (num_experts not in (None, 0, 1) and (self.device_mesh.tp_world_size or 1) > 1
                and not getattr(self.device_mesh, 'sequence_parallel', False)):
            # Megatron 0.15.3 requires sequence parallelism for MoE training when tensor
            # parallelism is enabled; keep the policy here so launch scripts need not know.
            self.device_mesh.sequence_parallel = True
            logger.info('Auto-enabled sequence_parallel for MoE model with tensor parallelism.')
        if 'overlap_grad_reduce' not in self.ddp_config:
            self.ddp_config['overlap_grad_reduce'] = False
        if 'overlap_param_gather' not in self.ddp_config:
            self.ddp_config['overlap_param_gather'] = False
        if 'align_param_gather' not in self.ddp_config:
            self.ddp_config['align_param_gather'] = False
        if 'grad_reduce_in_fp32' not in self.ddp_config:
            self.ddp_config['grad_reduce_in_fp32'] = True

        # Determine params_dtype and activation checkpointing kwargs
        params_dtype = torch.bfloat16
        if self.mixed_precision == 'fp16':
            params_dtype = torch.float16
        elif self.mixed_precision == 'no':
            params_dtype = torch.float32
        self._params_dtype = params_dtype

        vpp_size = self.device_mesh.vpp_size
        if vpp_size in (0, 1):
            vpp_size = None

        parallel_kwargs = {
            'tensor_model_parallel_size': self.device_mesh.tp_world_size or 1,
            'pipeline_model_parallel_size': self.device_mesh.pp_world_size or 1,
            'context_parallel_size': self.device_mesh.cp_world_size or 1,
            'expert_model_parallel_size': self.device_mesh.ep_size or 1,
            'expert_tensor_parallel_size': self.device_mesh.etp_world_size or 1,
            'virtual_pipeline_model_parallel_size': vpp_size,
        }
        if not vpp_size:
            # non-interleave does not support overlap_p2p_comm
            kwargs['overlap_p2p_comm'] = False
        kwargs.setdefault('overlap_p2p_comm', True)
        kwargs.setdefault('batch_p2p_comm', not kwargs['overlap_p2p_comm'])

        init_kwargs = {
            'order': self.device_mesh.order,
            **parallel_kwargs,
        }
        if Platform.device_prefix() == 'npu':
            init_kwargs['create_gloo_process_groups'] = True
        mpu.initialize_model_parallel(**init_kwargs)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        model_parallel_cuda_manual_seed(self.seed)
        self.config = self.get_model_config(self.hf_config, parallel_kwargs, **kwargs)
        self._finalize_quantized_param_config()
        self._check_fsdp()
        if nccl_comm_warmup:
            self._warmup_communicators()

    def _check_fsdp(self):
        """Reject Megatron-FSDP combinations that cannot work, rather than silently rewriting
        the user's distributed configuration."""
        if not self.uses_fsdp(self.ddp_config):
            return
        if not self.use_distributed_optimizer:
            # FSDP shards the parameters; only the distributed optimizer maintains matching
            # master-weight shards with a write-back path.
            raise ValueError('Megatron-FSDP requires use_distributed_optimizer=True: it shards the '
                             'parameters, and only the distributed optimizer maintains the matching '
                             'master-weight shards. Drop use_megatron_fsdp, or enable the distributed optimizer.')
        if self.device_mesh is not None and getattr(self.device_mesh, 'cp_world_size', 1) > 1:
            # megatron asserts the same pairing on its own CLI; checked here so it fails at
            # config-build time rather than part-way through wrapping on every rank.
            raise ValueError('Megatron-FSDP does not support context parallelism '
                             f'(cp_world_size={self.device_mesh.cp_world_size}). Use DDP for a CP run.')

    def _finalize_quantized_param_config(self):
        """Derive the DDP ``fp8/fp4_param_gather`` flags from ``fp8/fp4_param``.

        Quantized parameters are described on two config objects megatron never cross-checks:
        building them quantized vs the optimizer writing master weights back INTO them. Setting
        only the first is a silent no-op whose loss curve simply never moves, so an explicitly
        requested value is honoured, an absent one is filled, and a contradictory one raises.
        """
        for config_attr, ddp_key in (('fp8_param', 'fp8_param_gather'), ('fp4_param', 'fp4_param_gather')):
            if not getattr(self.config, config_attr, False):
                continue
            if self.ddp_config.get(ddp_key) is False:
                raise ValueError(
                    f'{config_attr}=True builds the parameters quantized, but {ddp_key}=False stops the '
                    f'distributed optimizer from ever writing the updated master weights back into them, so '
                    f'the model would not train at all. Drop {ddp_key}=False, or turn off {config_attr}.')
            self.ddp_config[ddp_key] = True
            if not self.use_distributed_optimizer:
                # The quantize-back step exists only on DistributedOptimizer.
                raise ValueError(f'{config_attr}=True requires use_distributed_optimizer=True: quantized '
                                 'parameters are updated by re-quantizing the distributed optimizer\'s master '
                                 'shards, which is the only code path that writes them.')

    @property
    def sequence_parallel(self) -> bool:
        """Read from device_mesh so auto-enable in args.py is visible."""
        return getattr(self.device_mesh, 'sequence_parallel', False)

    @property
    def bridge(self):
        return self.config.bridge

    @property
    def backend(self):
        """The BridgeBackend this strategy delegates model construction to."""
        return self._backend

    @property
    def params_type(self) -> torch.dtype:
        return self._params_dtype

    def _check_device_mesh(self):
        from megatron.core import parallel_state as mpu

        assert self.device_mesh.dp_world_size == mpu.get_data_parallel_world_size()
        assert self.device_mesh.dp_rank == mpu.get_data_parallel_rank()

        # Only validate world sizes match
        if self.device_mesh.tp_world_size > 1:
            assert self.device_mesh.tp_world_size == mpu.get_tensor_model_parallel_world_size()
            assert self.device_mesh.tp_rank == mpu.get_tensor_model_parallel_rank()

        if self.device_mesh.pp_world_size > 1:
            assert self.device_mesh.pp_world_size == mpu.get_pipeline_model_parallel_world_size()
            assert self.device_mesh.pp_rank == mpu.get_pipeline_model_parallel_rank()
            assert self.device_mesh.is_pp_last_rank() == mpu.is_pipeline_last_stage()
            assert self.device_mesh.is_pp_first_rank() == mpu.is_pipeline_first_stage()

        if self.device_mesh.cp_world_size > 1:
            assert self.device_mesh.cp_world_size == mpu.get_context_parallel_world_size()
            assert self.device_mesh.cp_rank == mpu.get_context_parallel_rank()

        if self.device_mesh.vpp_size is not None and self.device_mesh.vpp_size > 1:
            assert self.device_mesh.vpp_size == mpu.get_virtual_pipeline_model_parallel_world_size()

    def wrap_model(
        self,
        model: List[nn.Module],
    ) -> List[nn.Module]:
        if self.device_mesh.world_size <= 1:
            from megatron.core.distributed import DistributedDataParallelConfig
            ddp_config = DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                use_distributed_optimizer=False,
            )
            for m in model:
                if not hasattr(m, 'ddp_config'):
                    m.ddp_config = ddp_config
            return model

        self._check_device_mesh()
        return self._wrap_with_megatron_ddp(model, self.use_distributed_optimizer, self.ddp_config)

    def unwrap_model(self, model: List[nn.Module]) -> List[nn.Module]:
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from megatron.core.distributed import FullyShardedDataParallel as MegatronFSDP
        from megatron.core.transformer.module import Float16Module
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        _models = []
        for _model in model:
            # Unwrap DDP first
            while isinstance(_model, (MegatronDDP, MegatronFSDP, TorchDDP, Float16Module)):
                _model = _model.module
            _models.append(_model)
        return _models

    def get_sharded_sd_metadata(self) -> Dict[str, Any]:
        """Optimizer-state sharding metadata; must agree with the wrapper chosen at wrap time,
        so it lives beside that choice. FSDP shards as DTensors, the rest dp_reshardable."""
        metadata = {'singleton_local_shards': False, 'chained_optim_avoid_prefix': True}
        if self.uses_fsdp(self.ddp_config):
            metadata['distrib_optim_sharding_type'] = 'fsdp_dtensor'
        else:
            metadata['distrib_optim_sharding_type'] = 'dp_reshardable'
        return metadata

    def finish_param_config(self, model: List[nn.Module], optimizer: Any):
        self.config.grad_scale_func = getattr(optimizer, 'scale_loss') if optimizer is not None else None
        ddp_config = self.ddp_config
        if ddp_config['overlap_grad_reduce']:
            assert self.config.no_sync_func is None, (
                'When overlap_grad_reduce is True, config.no_sync_func must be None; '
                'a custom no_sync_func is not supported when overlapping grad-reduce')
            self.config.no_sync_func = [model_chunk.no_sync for model_chunk in model]  # noqa
            if len(model) == 1:
                self.config.no_sync_func = self.config.no_sync_func[0]  # noqa
            self.config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]  # noqa
            if len(model) == 1:
                self.config.grad_sync_func = self.config.grad_sync_func[0]  # noqa
            if not self._align_grad_reduce:
                # align_grad_reduce=False opts out of bucket-aligned reduction; grad_sync_func
                # is the aligned-overlap hook, so clear it (legacy does the same).
                self.config.grad_sync_func = None
        if ddp_config['overlap_param_gather'] and ddp_config['align_param_gather']:
            # Only DDP exposes start_param_sync; Megatron-FSDP drives its own parameter
            # all-gathers from module hooks, so there is nothing to hand over.
            missing = [m for m in model if not hasattr(m, 'start_param_sync')]
            if missing:
                logger.info('Skipping config.param_sync_func: the data-parallel wrapper '
                            f'({type(missing[0]).__name__}) does not implement start_param_sync, which is '
                            'a DDP-only hook. Its parameter gathering is driven internally instead.')
                return
            self.config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]  # noqa
            if len(model) == 1:
                self.config.param_sync_func = self.config.param_sync_func[0]  # noqa

    @staticmethod
    def _wrap_with_megatron_ddp(
        model: List[nn.Module],
        use_distributed_optimizer: bool,
        ddp_config: Dict[str, Any],
    ) -> List[nn.Module]:
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.module import Float16Module

        # Megatron-FSDP is a sibling of DDP taking the same constructor args, so the choice is
        # only which class to instantiate -- and it must come from ddp_config, not a flag flowing
        # into the config object (which the run would silently ignore).
        if MegatronStrategy.uses_fsdp(ddp_config):
            from megatron.core.distributed import FullyShardedDataParallel as data_parallel_cls
        else:
            data_parallel_cls = MegatronDDP

        wrapped_models = []
        for _model in model:
            _model = MegatronStrategy._move_model_to_gpu(_model)
            config: TransformerConfig = _model.config  # noqa

            if not isinstance(_model, Float16Module) and (config.fp16 or config.bf16):
                _model = Float16Module(config, _model)

            ddp_config_cls = DistributedDataParallelConfig(
                **ddp_config,
                use_distributed_optimizer=use_distributed_optimizer,
            )
            wrapped_model = data_parallel_cls(
                config=config,
                ddp_config=ddp_config_cls,
                module=_model,
            )

            # Broadcast params from data parallel src rank
            # In torchrun mode, all ranks enter here simultaneously, so this works
            wrapped_model.broadcast_params()
            wrapped_models.append(wrapped_model)

        return wrapped_models

    def reduce_loss(self, local_loss, local_count, logits, logps):
        count = local_count.clamp(min=1).to(torch.int64)
        cp_size = self.device_mesh.cp_world_size or 1
        grad_count = (count // cp_size).clamp(min=1) if cp_size > 1 else count
        # The dict's 'loss' must be a real copy: mcore scales the FIRST tuple element in place for
        # gradient accumulation (output_tensor /= num_tokens; /= num_microbatches), so a bare
        # detach() shares storage and the reported loss gets silently divided too.
        return local_loss, grad_count, {
            'loss': local_loss.detach().clone(),
            'logits': logits.detach() if logits is not None else None,
            'logps': logps.detach() if logps is not None else None,
            'num_tokens': count
        }

    def get_model_config(
        self,
        hf_config: PretrainedConfig,
        parallel_kwargs: Dict[str, Any],
        **kwargs,
    ):
        return self._backend.build_model_config(hf_config, parallel_kwargs, self, **kwargs)

    def create_megatron_model(
        self,
        load_weights: bool = True,
    ) -> List[nn.Module]:
        return self._backend.create_model(
            self.config, self.model_dir, load_weights=load_weights, move_to_gpu=self._move_model_to_gpu)

    @staticmethod
    def _move_model_to_gpu(model: nn.Module) -> nn.Module:
        model = model.to(Platform.get_local_device())
        torch_util.synchronize()
        return model

    @staticmethod
    def _warmup_communicators() -> None:
        """One dummy all-reduce per NCCL communicator (opt-in via ``nccl_comm_warmup``) so the
        first real step does not pay connection setup; requires mpu, initialized by __init__."""
        from megatron.core import mpu

        dummy = torch.zeros(1, device=torch_util.get_current_device())
        warmed = 0
        for getter, kwargs in (
            (mpu.get_data_parallel_group, {
                'with_context_parallel': True
            }),
            (mpu.get_data_parallel_group, {}),
            (mpu.get_context_parallel_group, {}),
            (mpu.get_tensor_model_parallel_group, {}),
            (mpu.get_pipeline_model_parallel_group, {}),
            (mpu.get_model_parallel_group, {}),
            (mpu.get_embedding_group, {}),
            (mpu.get_position_embedding_group, {}),
        ):
            try:
                groups = getter(**kwargs)
            except (AssertionError, ValueError, TypeError):
                continue
            for group in groups if isinstance(groups, list) else [groups]:
                if group is not None:
                    torch.distributed.all_reduce(dummy, group=group)
                    warmed += 1
        torch_util.synchronize()
        logger.info(f'NCCL communicator warm-up done ({warmed} groups).')

    @contextlib.contextmanager
    def transactional_grad_schedule(self, model_chunks: List[nn.Module], *, forward_only: bool,
                                    transactional: bool, loss_weight: float, finalize_gradients: bool,
                                    completing_pending: bool, global_normalizer, **kwargs):
        """Temporarily rewrite this strategy's gradient hooks for one schedule, then restore
        them -- the mcore ``config`` is shared by every later step, so per-call surgery must be
        undone even when the schedule raises. The model decides WHAT this pass is (the keyword
        flags); this strategy decides HOW the hooks bend.

        Args:
            transactional: Scale the loss by ``loss_weight / normalizer / ga_steps``.
            finalize_gradients: False defers: gradients accumulate without syncing.
            completing_pending: True runs the stored finalize with ``num_tokens=None``.
            global_normalizer: Token normalizer the transactional scale divides by.
        """
        config = self.config
        original_grad_scale_func = config.grad_scale_func
        original_no_sync_func = config.no_sync_func
        original_grad_sync_func = config.grad_sync_func
        original_finalize_model_grads_func = config.finalize_model_grads_func

        scale = float(loss_weight)
        if transactional:
            gradient_accumulation_steps = int(kwargs.get('gradient_accumulation_steps') or 1)
            if gradient_accumulation_steps < 1:
                raise ValueError('gradient_accumulation_steps must be positive')
            scale = scale / global_normalizer.clamp(min=1) / gradient_accumulation_steps

        def transaction_grad_scale(value):
            if original_grad_scale_func is not None:
                value = original_grad_scale_func(value)
            return value * scale

        def disabled_no_sync():
            return contextlib.nullcontext()

        def disabled_grad_sync(*args, **kwargs):
            return None

        def disabled_finalize(*args, **kwargs):
            return None

        def finalize_accumulated_gradients(model, num_tokens=None, **finalize_kwargs):
            return original_finalize_model_grads_func(model, None, **finalize_kwargs)

        outer_no_sync = contextlib.ExitStack()
        try:
            with outer_no_sync:
                if not forward_only and (transactional or loss_weight != 1.0):
                    config.grad_scale_func = transaction_grad_scale
                if not forward_only and not finalize_gradients:
                    for model_chunk in model_chunks:
                        if hasattr(model_chunk, 'no_sync'):
                            outer_no_sync.enter_context(model_chunk.no_sync())
                    if isinstance(original_no_sync_func, list):
                        config.no_sync_func = [disabled_no_sync for _ in original_no_sync_func]
                    else:
                        config.no_sync_func = disabled_no_sync
                    if isinstance(original_grad_sync_func, list):
                        config.grad_sync_func = [disabled_grad_sync for _ in original_grad_sync_func]
                    else:
                        config.grad_sync_func = disabled_grad_sync
                    config.finalize_model_grads_func = disabled_finalize
                elif completing_pending and original_finalize_model_grads_func is not None:
                    config.finalize_model_grads_func = finalize_accumulated_gradients
                yield
        finally:
            config.grad_scale_func = original_grad_scale_func
            config.no_sync_func = original_no_sync_func
            config.grad_sync_func = original_grad_sync_func
            config.finalize_model_grads_func = original_finalize_model_grads_func

    @torch.no_grad()
    def offload_to_cpu(self, model: List[nn.Module], optimizer: Any = None, *, offload_model: bool = True) -> None:
        """Release this rank's training memory for colocation (buffer-level work is Megatron's
        own offload_to_cpu). ``offload_model=False`` leaves the chunks resident and moves only
        ``optimizer`` when one is passed. Reversed by :meth:`reload_to_gpu`; both read what is
        currently resident, so a repeated call is a no-op."""
        from twinkle.model.offload import record_offload
        if not offload_model and optimizer is None:
            record_offload(self, model=False, optimizer=False)
            return
        # Touch every chunk first so a Megatron-FSDP model still refuses, even when only the
        # optimizer was requested: its parameter storage is not the DDP bucket layout this path moves.
        for chunk in model:
            self._flat_buffers(chunk)
        # Buffers are about to have their storage freed; make sure nothing is still writing to it.
        torch_util.synchronize()
        if offload_model:
            for chunk in model:
                buffers = self._flat_buffers(chunk)
                for buffer in buffers:
                    buffer.offload_to_cpu()
                self._move_loose_params(chunk, buffers, 'cpu')
        if optimizer is not None:
            self._move_optimizer_state(optimizer, 'cpu')
        record_offload(self, model=offload_model, optimizer=optimizer is not None)
        gc.collect()
        torch_util.empty_cache()

    @torch.no_grad()
    def reload_to_gpu(self, model: List[nn.Module], optimizer: Any = None) -> None:
        """Bring back what :meth:`offload_to_cpu` released. A no-op when nothing was moved."""
        from twinkle.model.offload import clear_offload_record, offload_record
        moved_model, moved_optimizer = offload_record(self)
        if not moved_model and not moved_optimizer:
            return
        device = Platform.get_local_device()
        if moved_model:
            for chunk in model:
                buffers = self._flat_buffers(chunk)
                for buffer in buffers:
                    buffer.reload_from_cpu()
                self._move_loose_params(chunk, buffers, device)
        if moved_optimizer and optimizer is not None:
            self._move_optimizer_state(optimizer, device)
        clear_offload_record(self)
        gc.collect()
        torch_util.empty_cache()

    @staticmethod
    def _flat_buffers(chunk: nn.Module) -> List[Any]:
        """The flat parameter/gradient buckets of a DDP-wrapped chunk. Cannot duck-type on the
        attribute name: nn.Module already defines ``buffers()`` as a method."""
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from megatron.core.distributed import FullyShardedDataParallel as MegatronFSDP
        if isinstance(chunk, MegatronDDP):
            return [*chunk.buffers, *chunk.expert_parallel_buffers]
        if isinstance(chunk, MegatronFSDP):
            raise NotImplementedError(
                'Offloading a Megatron-FSDP model is not supported: it holds its parameters in '
                '`param_and_grad_buffer`, which is not the per-bucket structure DDP uses. Freeing '
                'only the parts this method understands would leave most of the training memory '
                'resident while reporting success, which defeats the purpose.')
        return []

    @staticmethod
    def _move_loose_params(chunk: nn.Module, buffers: List[Any], device: Any) -> None:
        """Move the parameters no flat buffer holds: frozen ones (a LoRA base weight), chunks
        never wrapped (single-rank / reference models), or runs without a distributed optimizer
        (grad_data exists, param_data stays None). Asks the buffers what they actually hold so
        gradient-only buckets do not masquerade as full coverage."""
        params_are_pooled = any(buffer.param_data is not None for buffer in buffers)
        for param in chunk.parameters():
            if params_are_pooled and param.requires_grad:
                continue
            param.data = param.data.to(device, non_blocking=True)
            if param.grad is not None:
                param.grad = param.grad.to(device, non_blocking=True)

    @staticmethod
    def _move_optimizer_state(optimizer: Any, device: Any) -> None:
        """Move an optimizer's tensor state between host and device.

        Megatron has offload_to_cpu/restore_from_cpu for this, but they hardcode ``.cuda()``
        and ``torch.cuda.empty_cache()``, unusable on NPU -- this is the same walk with the
        device left as a parameter.
        """
        from megatron.core.optimizer.optimizer import ChainedOptimizer
        optimizers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
        for _optimizer in optimizers:
            # Megatron wraps a plain torch optimizer; fall back to the object itself so a bare one
            # also works. `optimizer` being None means this rank owns no parameters, which is
            # legitimate under a custom pipeline layout.
            inner = getattr(_optimizer, 'optimizer', _optimizer)
            if inner is None or getattr(_optimizer, 'is_stub_optimizer', False):
                continue
            # The inner param_groups hold the optimizer's own handle on each parameter: separate
            # fp32 master shards under bf16/fp16 (must be moved too), or -- for FP32 training --
            # VIEWS of the DDP flat buffer already offloaded above, whose storage was freed. Their
            # bytes are safe in the buffer's host copy; reassigning a freed view's .data would
            # detach the optimizer from the buffer the model reads, so skip exactly those.
            for group in inner.param_groups:
                for param in group['params']:
                    if isinstance(param, torch.Tensor) and param.untyped_storage().size() > 0:
                        param.data = param.data.to(device, non_blocking=True)
            # Momentum/variance are always fresh allocations, never buffer views; the freed-storage
            # guard just keeps the walk uniform.
            for state in inner.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.untyped_storage().size() > 0:
                        state[key] = value.to(device, non_blocking=True)
