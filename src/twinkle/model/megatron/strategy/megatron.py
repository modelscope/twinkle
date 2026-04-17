# Copyright (c) ModelScope Contributors. All rights reserved.
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

    def __init__(
        self,
        model_dir,
        device_mesh: Optional[DeviceMesh] = None,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        seed: int = 42,
        variable_seq_lengths: bool = False,
        config: PreTrainedConfig = None,
        ddp_config: Dict[str, Any] = None,
        **kwargs,
    ):
        import torch.distributed as dist
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
        if Platform.device_prefix() == 'npu' and dist.is_initialized():
            default_pg = dist.distributed_c10d._get_default_group()
            if getattr(default_pg, 'bound_device_id', None) is not None:
                # If the default HCCL PG keeps a bound device id, PyTorch may
                # propagate that binding into later Gloo subgroup creation. That
                # breaks the metrics/object-gather path on NPU, so clear it
                # before Megatron creates its Gloo DP groups.
                default_pg.bound_device_id = None

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

    @property
    def sequence_parallel(self) -> bool:
        """Read from device_mesh so auto-enable in args.py is visible."""
        return getattr(self.device_mesh, 'sequence_parallel', False)

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
        from megatron.core.transformer.module import Float16Module
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        _models = []
        for _model in model:
            # Unwrap DDP first
            while isinstance(_model, (MegatronDDP, TorchDDP, Float16Module)):
                _model = _model.module
            _models.append(_model)
        return _models

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
            wrapped_model = MegatronDDP(
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
        return local_loss, count, {
            'loss': local_loss.detach(),
            'logits': logits.detach(),
            'logps': logps.detach(),
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
