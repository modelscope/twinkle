# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Literal, Optional, Dict, Any

import torch
import torch.nn as nn

from twinkle import DeviceMesh, Platform, torch_util


class MegatronStrategy:

    def __init__(
        self,
        model_dir,
        device_mesh: Optional[DeviceMesh] = None,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        seed: int = 42,
        variable_seq_lengths: bool = False,
        **kwargs,
    ):
        from megatron.core import mpu
        self.device_mesh = device_mesh
        self.use_distributed_optimizer = use_distributed_optimizer
        self.mixed_precision = mixed_precision
        self.model_dir = model_dir
        self.seed = seed
        self.variable_seq_lengths = variable_seq_lengths
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
        mpu.initialize_model_parallel(
            order=self.device_mesh.order,
            **parallel_kwargs,
        )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        model_parallel_cuda_manual_seed(self.seed)
        self.config = self.get_model_config(model_dir, parallel_kwargs, **kwargs)

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
        use_distributed_optimizer: bool = True,
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
        return self._wrap_with_megatron_ddp(model, use_distributed_optimizer)

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

    @staticmethod
    def _wrap_with_megatron_ddp(
        model: List[nn.Module],
        use_distributed_optimizer: bool,
    ) -> List[nn.Module]:
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.module import Float16Module

        wrapped_models = []
        for _model in model:
            config: TransformerConfig = _model.config  # noqa

            if not isinstance(model, Float16Module) and (config.fp16 or config.bf16):
                _model = Float16Module(config, _model)

            ddp_config = DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                use_distributed_optimizer=use_distributed_optimizer,
            )

            wrapped_model = MegatronDDP(
                config=config,
                ddp_config=ddp_config,
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
        model_dir: str,
        parallel_kwargs: Dict[str, Any],
        **kwargs,
    ):
        from mcore_bridge import ModelConfig, hf_to_mcore_config
        from megatron.core.distributed import finalize_model_grads as _native_finalize_model_grads
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        config_kwargs = hf_to_mcore_config(hf_config)
        config_kwargs.update(kwargs)
        if 'calculate_per_token_loss' not in config_kwargs:
            config_kwargs['calculate_per_token_loss'] = True

        if 'moe_token_dispatcher_type' not in config_kwargs:
            config_kwargs['moe_token_dispatcher_type'] = 'alltoall' if self.variable_seq_lengths else 'allgather'

        def finalize_model_grads_for_lora(model, *args, **kwargs):
            from megatron.core.distributed import DistributedDataParallel as MegatronDDP
            from peft import PeftModel as _PeftModel

            # Check if model is DDP-wrapped (has ddp_config)
            # Need to unwrap PeftModel to check the underlying model
            def _get_base_model(m):
                if isinstance(m, _PeftModel):
                    return _get_base_model(m.base_model.model)
                return m

            base_model = _get_base_model(model[0])
            if isinstance(base_model, MegatronDDP) or hasattr(base_model, 'ddp_config'):
                # Use native implementation for DDP models
                return _native_finalize_model_grads(model, *args, **kwargs)

        config = ModelConfig(
            use_cpu_initialization=True,
            params_dtype=self.params_type,
            sequence_parallel=self.sequence_parallel,
            finalize_model_grads_func=finalize_model_grads_for_lora,
            variable_seq_lengths=self.variable_seq_lengths,
            **parallel_kwargs,
            **config_kwargs,
        )
        return config

    def create_megatron_model(
        self,
        load_weights: bool = True,
    ) -> List[nn.Module]:
        from mcore_bridge import get_mcore_model
        import torch.distributed as dist
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