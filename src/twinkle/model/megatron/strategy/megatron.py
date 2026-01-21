# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, Literal, Optional, Tuple, List

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import PeftModel

from twinkle import DeviceMesh, exists, Platform


class MegatronStrategy:

    def __init__(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        expert_tensor_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        params_dtype: Optional[str] = None,
        megatron_args: Optional[Dict[str, Any]] = None,
    ):
        self.device_mesh = device_mesh
        self.etp_size = expert_tensor_parallel_size or self.device_mesh.tp_world_size
        self.sequence_parallel = sequence_parallel
        self.use_distributed_optimizer = use_distributed_optimizer
        self.mixed_precision = mixed_precision
        self._params_dtype = params_dtype
        self._megatron_args = megatron_args or {}
        self._initialized = False
        self._parallel_state = None

    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return

        from megatron.core import parallel_state
        dist.init_process_group(backend='nccl')

        init_kwargs = {
            'tensor_model_parallel_size': self.device_mesh.tp_world_size or 1,
            'pipeline_model_parallel_size': self.device_mesh.pp_world_size or 1,
            'context_parallel_size': self.device_mesh.cp_world_size or 1,
            'virtual_pipeline_model_parallel_size': self.device_mesh.vpp_size or 1,
            'expert_model_parallel_size': self.device_mesh.ep_size or 1,
        }

        if exists('megatron_core>=0.13'):
            init_kwargs['expert_tensor_parallel_size'] = self.etp_size

        parallel_state.initialize_model_parallel(**init_kwargs)

        self._parallel_state = parallel_state
        self._initialized = True

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

    @staticmethod
    @DeprecationWarning
    def _get_transformer_config(model: nn.Module):

        def _valid_config(_config):
            return _config is not None and hasattr(_config, 'tensor_model_parallel_size')

        def _search_config(module: nn.Module, paths: Optional[List[str]] = None):
            paths = paths or ['']
            for path in paths:
                try:
                    sub_module = module.get_submodule(path) if path else module
                    _config = getattr(sub_module, 'config', None)
                    if _valid_config(_config):
                        return _config
                except AttributeError:
                    pass
            return None

        config = _search_config(model, ['base_model', 'model', 'base_model.model'])
        assert config is not None, 'Cannot find valid config on megatron model.'
        return config

    def wrap_model(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        use_distributed_optimizer: bool = True,
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
        if not self._initialized:
            self.initialize()

        if self.device_mesh.world_size <= 1:
            return model, optimizer

        return self._wrap_with_megatron_ddp(model, optimizer,
                                            use_distributed_optimizer)

    @staticmethod
    def _wrap_with_megatron_ddp(
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        use_distributed_optimizer: bool,
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.transformer.module import Float16Module
        from megatron.core.transformer import TransformerConfig
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP

        assert not isinstance(model, PeftModel), 'Cannot wrap peft model.'
        config: TransformerConfig = model.config # noqa
        model_device = next(model.parameters()).device
        if model_device.type == 'cpu':
            model = model.to(Platform.get_local_device())

        if not isinstance(model, Float16Module) and  (config.fp16 or config.bf16):
            model = Float16Module(config, model)

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=use_distributed_optimizer,
        )

        # Wrap with MegatronDDP
        # TODO: multi-tenant ddp
        wrapped_model = MegatronDDP(
            config=config,
            ddp_config=ddp_config,
            module=model,
        )

        # Broadcast params from data parallel src rank
        # In torchrun mode, all ranks enter here simultaneously, so this works
        wrapped_model.broadcast_params()

        return wrapped_model, optimizer

    def get_model_config(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_layers: int,
        ffn_hidden_size: Optional[int] = None,
        num_query_groups: Optional[int] = None,
        num_experts: Optional[int] = None,
        moe_router_topk: int = 2,
        **kwargs,
    ):
        from megatron.core.transformer import TransformerConfig

        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups or num_attention_heads,
            ffn_hidden_size=ffn_hidden_size or 4 * hidden_size,
            use_cpu_initialization=True,
            params_dtype=self.params_type,
            tensor_model_parallel_size=self.device_mesh.tp_world_size or 1,
            pipeline_model_parallel_size=self.device_mesh.pp_world_size or 1,
            context_parallel_size=self.device_mesh.cp_world_size or 1,
            expert_model_parallel_size=self.device_mesh.ep_size or 1,
            sequence_parallel=self.sequence_parallel,
            num_moe_experts=num_experts,
            moe_router_topk=moe_router_topk,
            **kwargs,
        )

        return config

