# Copyright (c) ModelScope Contributors. All rights reserved.
import os

import torch
import torch.nn as nn
from transformers import PreTrainedConfig
from typing import Any, Dict, List, Literal, Optional

from twinkle import DeviceMesh, Platform, torch_util
from twinkle.utils import get_logger
from .._mindspeed_runtime import configure_mindspeed_runtime_args

logger = get_logger()


def finalize_model_grads_for_lora(model, *args, **kwargs):
    """Only enter Megatron native finalize when the wrapped model has sync capability.

    In single-rank/no-op wrap cases Twinkle attaches ``ddp_config`` to the bare
    module for optimizer compatibility, but that does not mean the model really
    implements ``finish_grad_sync()``. Native Megatron finalize ultimately calls
    that method, so we gate by runtime capability instead of config metadata.
    """
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


class MegatronStrategy:

    #: ddp_config keys that select a sharded (Megatron-FSDP) data-parallel wrapper instead of plain
    #: DDP. ``use_custom_fsdp`` is megatron's own deprecated alias -- its arguments.py still derives
    #: it from ``use_megatron_fsdp`` -- so a config carrying either key means the same thing here.
    _FSDP_KEYS = ('use_megatron_fsdp', 'use_custom_fsdp')

    #: What to raise CUDA_DEVICE_MAX_CONNECTIONS to for FSDP. megatron only requires "> 1"; 32 is
    #: the value its own error message suggests, and matches legacy Megatron-SWIFT.
    _FSDP_DEVICE_MAX_CONNECTIONS = '32'

    @classmethod
    def uses_fsdp(cls, ddp_config: Optional[Dict[str, Any]]) -> bool:
        """Whether this ddp_config asks for Megatron-FSDP rather than DDP.

        Public because the decision is needed before a strategy instance exists (see
        ``apply_process_env``) and it must not be spelled out twice: the wrapper class, the
        checkpoint sharding type and the device-connection count all have to agree, and they are
        chosen at three different points in the run.
        """
        ddp_config = ddp_config or {}
        return any(ddp_config.get(key) for key in cls._FSDP_KEYS)

    @classmethod
    def apply_process_env(cls, ddp_config: Optional[Dict[str, Any]] = None) -> None:
        """Set the environment megatron needs latched BEFORE the CUDA context is created.

        Must run before ``torch_util.set_device()``, which creates the context: the CUDA driver reads
        CUDA_DEVICE_MAX_CONNECTIONS when it builds the context, so writing it afterwards is silently
        a no-op. That is why this is a classmethod called by the model rather than something the
        strategy's ``__init__`` does -- the strategy is only constructed after the device is set.

        CUDA_DEVICE_MAX_CONNECTIONS caps the hardware work queues per context, and the two
        data-parallel modes want opposite things from it:

        - DDP with tensor parallelism wants ``1``, which forces kernels to be issued in call order so
          that a collective launched just before a GEMM really does overlap with it. megatron is
          explicit that this ordering is "necessary for a speedup but not for correctness"
          (tensor_parallel/layers.py) and only *warns* when the value is something else.
        - FSDP wants more than 1, because it issues several parameter all-gathers concurrently to
          overlap them with compute, and a single queue serializes those instead.

        So this is a performance knob in both directions: choosing the wrong one is slow, not wrong.
        """
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        if not cls.uses_fsdp(ddp_config):
            os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            return
        # Only override the value the framework itself would have chosen. Someone who picked their
        # own connection count knows their topology better than we do -- except for '1', which is
        # the single value FSDP cannot work with, and is also exactly what twinkle's own Ray worker
        # bootstrap leaves behind.
        current = os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')
        if current in (None, '1'):
            os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = cls._FSDP_DEVICE_MAX_CONNECTIONS
            logger.info(f'Megatron-FSDP requires CUDA_DEVICE_MAX_CONNECTIONS > 1; set to '
                        f'{cls._FSDP_DEVICE_MAX_CONNECTIONS} (was {current!r}). Export your own value to override.')

    def __init__(
        self,
        model_dir,
        device_mesh: Optional[DeviceMesh] = None,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        seed: int = 42,
        variable_seq_lengths: bool = True,
        config: PreTrainedConfig = None,
        ddp_config: Dict[str, Any] = None,
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
        if config is None:
            from transformers import AutoConfig
            self.hf_config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
        else:
            self.hf_config = config
        num_experts = getattr(self.hf_config, 'num_experts', getattr(self.hf_config, 'num_local_experts', None))
        if (num_experts not in (None, 0, 1) and (self.device_mesh.tp_world_size or 1) > 1
                and not getattr(self.device_mesh, 'sequence_parallel', False)):
            # Megatron 0.15.3 requires sequence parallelism for MoE training when
            # tensor parallelism is enabled. Keep this policy in the framework so
            # cookbook scripts do not need to know a model-family-specific
            # runtime constraint just to launch a valid MoE run.
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
        if 'overlap_p2p_comm' not in kwargs:
            kwargs['overlap_p2p_comm'] = True
            kwargs['batch_p2p_comm'] = not kwargs['overlap_p2p_comm']

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

    def _check_fsdp(self):
        """Reject Megatron-FSDP combinations that cannot work.

        Raised rather than fixed up, matching ``_finalize_quantized_param_config``: silently
        rewriting a user's distributed configuration is how a run ends up doing something other than
        what its config says, which is the failure mode this whole area keeps producing.
        """
        if not self.uses_fsdp(self.ddp_config):
            return
        if not self.use_distributed_optimizer:
            # Megatron-FSDP shards the parameters, so the optimizer has to own matching shards of the
            # master weights; the non-distributed optimizer keeps whole tensors and has no path that
            # writes back into a sharded parameter.
            raise ValueError('Megatron-FSDP requires use_distributed_optimizer=True: it shards the '
                             'parameters, and only the distributed optimizer maintains the matching '
                             'master-weight shards. Drop use_megatron_fsdp, or enable the distributed optimizer.')
        if self.device_mesh is not None and getattr(self.device_mesh, 'cp_world_size', 1) > 1:
            # megatron asserts the same pairing on its own CLI ('Hybrid context parallelism not
            # supported with Megatron FSDP'). Checked here so it fails while the config is being
            # built, rather than part-way through wrapping the model on every rank.
            raise ValueError('Megatron-FSDP does not support context parallelism '
                             f'(cp_world_size={self.device_mesh.cp_world_size}). Use DDP for a CP run.')

    def _finalize_quantized_param_config(self):
        """Keep the DDP param-gather flags in step with the model's quantized-parameter config.

        Quantized (FP8/FP4) parameters are described twice, on two config objects megatron never
        cross-checks: ``TransformerConfig.fp8_param`` / ``fp4_param`` decide that the parameters are
        BUILT quantized, while ``DistributedDataParallelConfig.fp8_param_gather`` /
        ``fp4_param_gather`` decide that the distributed optimizer writes its FP32 master shards back
        INTO them, in ``DistributedOptimizer._copy_main_params_to_model_params``.

        Setting only the first is a silent no-op rather than a slow path. That method dispatches on
        the DDP flags alone, and its generic fallback then skips exactly the parameters the skipped
        branch was supposed to handle ("FP8 params are quantized in the above quantize_param_shard
        function" / the matching NVFP4 comment). So the master weights advance every step while the
        model parameters stay at their loaded values: no error, no warning, and a loss curve that
        simply never moves.

        Derived rather than merely validated because there is no configuration in which one wants the
        pair split; an explicitly requested value is still honoured, and only an absent one is filled.
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
                # The quantize-back step exists only on DistributedOptimizer; the plain optimizer has
                # no equivalent, so the parameters would again never be updated.
                raise ValueError(f'{config_attr}=True requires use_distributed_optimizer=True: quantized '
                                 'parameters are updated by re-quantizing the distributed optimizer\'s master '
                                 'shards, which is the only code path that writes them.')

    @property
    def sequence_parallel(self) -> bool:
        """Read from device_mesh so auto-enable in args.py is visible."""
        return getattr(self.device_mesh, 'sequence_parallel', False)

    def init_kwargs(self) -> Dict[str, Any]:
        # Parallelism is configured through the megatron model config, not at construction time.
        return {}

    @property
    def bridge(self):
        return self.config.bridge

    @property
    def params_type(self) -> torch.dtype:
        if self._params_dtype is not None:
            dtype_map = {
                'fp32': torch.float32,
                'fp16': torch.float16,
                'bf16': torch.bfloat16,
            }
            return dtype_map.get(self._params_dtype, torch.bfloat16)

        if self.mixed_precision == 'bf16':
            return torch.bfloat16
        elif self.mixed_precision == 'fp16':
            return torch.float16
        return torch.float32

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
        """Metadata describing how the distributed optimizer's state is sharded on disk.

        Lives on the strategy because it has to agree with the wrapper class chosen in
        ``_wrap_with_megatron_ddp``, and those are two decisions made at very different times -- the
        model is wrapped at startup, the metadata is read on every save and every resume. Keeping
        them in one class is what stops them drifting apart into a checkpoint whose declared layout
        does not match the one actually written.

        Megatron-FSDP shards the optimizer state as DTensors, which is a different on-disk layout
        from the distributed optimizer's own; megatron only accepts ``fsdp_dtensor`` for FSDP runs and
        the reshardable type for the rest.
        """
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
        if ddp_config['overlap_param_gather'] and ddp_config['align_param_gather']:
            # Only DDP exposes start_param_sync. Megatron-FSDP has no equivalent because it drives
            # its own parameter all-gathers from the module hooks rather than being prompted by the
            # forward-backward schedule, so there is nothing to hand over -- and reaching for the
            # attribute anyway would be an AttributeError at wrap time.
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

        # Megatron-FSDP is a sibling of DDP, not a subclass of it, but it takes the same three
        # constructor arguments and the same DistributedDataParallelConfig, so the choice is only
        # about which class to instantiate. It has to be made from ddp_config rather than left to
        # ``use_megatron_fsdp`` flowing into the config object: the config would accept the flag and
        # the run would still be wrapped in DDP -- FSDP silently never engaged, no error anywhere.
        if MegatronStrategy.uses_fsdp(ddp_config):
            from megatron.core.distributed import FullyShardedDataParallel as data_parallel_cls
        else:
            data_parallel_cls = MegatronDDP

        wrapped_models = []
        for _model in model:
            _model = MegatronStrategy._move_model_to_gpu(_model)
            config: TransformerConfig = _model.config  # noqa

            if not isinstance(model, Float16Module) and (config.fp16 or config.bf16):
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
        return local_loss, grad_count, {
            'loss': local_loss.detach(),
            'logits': logits.detach() if logits is not None else None,
            'logps': logps.detach() if logps is not None else None,
            'num_tokens': count
        }

    def get_model_config(
        self,
        hf_config: PreTrainedConfig,
        parallel_kwargs: Dict[str, Any],
        **kwargs,
    ):
        from mcore_bridge import ModelConfig, hf_to_mcore_config
        config_kwargs = hf_to_mcore_config(hf_config)
        config_kwargs.update(kwargs)
        if 'calculate_per_token_loss' not in config_kwargs:
            config_kwargs['calculate_per_token_loss'] = True

        if 'moe_token_dispatcher_type' not in config_kwargs:
            config_kwargs['moe_token_dispatcher_type'] = 'alltoall' if self.variable_seq_lengths else 'allgather'
        model_config = ModelConfig(
            use_cpu_initialization=True,
            params_dtype=self.params_type,
            sequence_parallel=self.sequence_parallel,
            finalize_model_grads_func=finalize_model_grads_for_lora,
            variable_seq_lengths=self.variable_seq_lengths,
            **parallel_kwargs,
            **config_kwargs,
        )
        if Platform.device_prefix() == 'npu':
            # After Twinkle stops feeding the dense 4D causal mask, MindSpeed's
            # patched TE attention should generate its own compressed causal
            # mask. In 0.15.3 that path is gated by ``use_flash_attn`` on the
            # model config itself. If we leave it unset, MindSpeed falls back to
            # the non-flash mask generator and aborts the first 8-card forward
            # with: "Please set micro_batch_size or set use_flash_attn=True in
            # config." Keep the TE flash path enabled and let it synthesize the
            # mask it expects.
            model_config.use_flash_attn = True
        configure_mindspeed_runtime_args(model_config)
        return model_config

    def create_megatron_model(
        self,
        load_weights: bool = True,
    ) -> List[nn.Module]:
        import torch.distributed as dist
        from mcore_bridge import get_mcore_model
        mg_models = get_mcore_model(self.config)

        if dist.is_initialized():
            dist.barrier()

        _models = []
        for _model in mg_models:
            _model = self._move_model_to_gpu(_model)
            _models.append(_model)

        if load_weights:
            # Load weights
            bridge = self.config.bridge
            bridge.load_weights(mg_models, self.model_dir)
        return _models

    @staticmethod
    def _move_model_to_gpu(model: nn.Module) -> nn.Module:
        model = model.to(Platform.get_local_device())
        torch_util.synchronize()
        return model
