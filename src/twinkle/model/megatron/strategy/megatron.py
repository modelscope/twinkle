# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Literal, Optional, Tuple, List

import torch
import torch.nn as nn
from peft import PeftModel

from twinkle import DeviceMesh, Platform


class MegatronStrategy:

    def __init__(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        expert_tensor_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        params_dtype: Optional[str] = None,
        **kwargs,
    ):
        self.device_mesh = device_mesh
        self.etp_size = expert_tensor_parallel_size or self.device_mesh.tp_world_size
        self.sequence_parallel = sequence_parallel
        self.use_distributed_optimizer = use_distributed_optimizer
        self.mixed_precision = mixed_precision
        self._params_dtype = params_dtype

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
        model: List[nn.Module],
        optimizer: Optional[torch.optim.Optimizer] = None,
        use_distributed_optimizer: bool = True,
    ) -> Tuple[List[nn.Module], Optional[torch.optim.Optimizer]]:
        if self.device_mesh.world_size <= 1:
            return model, optimizer

        return self._wrap_with_megatron_ddp(model, optimizer,
                                            use_distributed_optimizer)

    def unwrap_model(self, model: List[nn.Module]) -> List[nn.Module]:
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        _models = []
        for _model in model:
            if isinstance(_model, (MegatronDDP, TorchDDP)):
                _models.append(model.module)
            else:
                _models.append(_model)
        return _models

    @staticmethod
    def _wrap_with_megatron_ddp(
        model: List[nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
        use_distributed_optimizer: bool,
    ) -> Tuple[List[nn.Module], Optional[torch.optim.Optimizer]]:
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.transformer.module import Float16Module
        from megatron.core.transformer import TransformerConfig
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP

        wrapped_models = []
        for _model in model:
            assert not isinstance(_model, PeftModel), 'Cannot wrap peft model.'
            config: TransformerConfig = _model.config # noqa

            if not isinstance(model, Float16Module) and  (config.fp16 or config.bf16):
                _model = Float16Module(config, _model)

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
                module=_model,
            )

            # Broadcast params from data parallel src rank
            # In torchrun mode, all ranks enter here simultaneously, so this works
            wrapped_model.broadcast_params()
            wrapped_models.append(wrapped_model)

        return wrapped_models, optimizer

    def split_inputs_for_cp(self, inputs):
        # Calculate padded seq_length based on parallelism requirements
        # 1. For CP > 1: seq_len must be divisible by 2 * cp_size
        # 2. For sequence_parallel with TP > 1: seq_len must be divisible by tp_size
        from megatron.core import parallel_state as mpu
        cp_size = self.device_mesh.cp_world_size
        tp_size = self.device_mesh.tp_world_size
        cp_rank = mpu.get_context_parallel_rank() if cp_size > 1 else 0
        input_ids = inputs.get('input_ids')
        position_ids = inputs.get('position_ids')
        attention_mask = inputs.get('attention_mask')
        batch_labels = inputs.get('labels')

        def split_tensor_for_cp(tensor, dim=-1):
            """
            Split tensor along sequence dimension for Context Parallel.

            With causal masking, split into 2*CP chunks and assign alternating
            chunks to balance workload across CP ranks.
            For CP rank i: chunks [i, 2*CP-1-i]
            """
            if tensor is None or cp_size <= 1:
                return tensor

            if dim < 0:
                dim = (dim + tensor.ndim) % tensor.ndim

            seq_len = tensor.shape[dim]

            # Reshape to [batch, 2*cp_size, seq_per_chunk, ...]
            view_shape = list(tensor.shape)
            view_shape[dim:dim + 1] = [2 * cp_size, seq_len // (2 * cp_size)]
            reshaped = tensor.view(*view_shape)

            # Select chunks [cp_rank, 2*cp_size-1-cp_rank]
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)],
                                 device='cpu',
                                 pin_memory=True).cuda(non_blocking=True)
            selected = reshaped.index_select(dim, index)

            # Reshape back: [batch, 2*seq_per_chunk, ...]
            out_shape = list(tensor.shape)
            out_shape[dim] = seq_len // cp_size
            return selected.reshape(*out_shape)

        # Split tensors for Context Parallel
        # Each CP rank processes a portion of the sequence
        if cp_size > 1:
            # Pad sequence for parallel compatibility
            # 1. For CP > 1: Megatron's RoPE requires seq_len % (2 * cp_size) == 0
            # 2. For sequence_parallel: seq_len must be divisible by TP size
            if input_ids is not None:
                seq_len = input_ids.shape[1]

                # Calculate required divisor based on parallelism settings
                if cp_size > 1:
                    divisor = 2 * cp_size
                elif self.sequence_parallel and tp_size > 1:
                    divisor = tp_size
                else:
                    divisor = 1

                if divisor > 1 and seq_len % divisor != 0:
                    pad_len = divisor - (seq_len % divisor)
                    # Pad input_ids
                    input_ids = torch.nn.functional.pad(input_ids,
                                                        (0, pad_len),
                                                        value=0)
                    # Pad labels if present
                    if batch_labels is not None:
                        batch_labels = torch.nn.functional.pad(batch_labels,
                                                               (0, pad_len),
                                                               value=-100)
                    # Pad attention_mask if present
                    if attention_mask is not None:
                        attention_mask = torch.nn.functional.pad(
                            attention_mask, (0, pad_len), value=0)
                    # Pad position_ids if present
                    if position_ids is not None:
                        position_ids = torch.nn.functional.pad(position_ids,
                                                               (0, pad_len),
                                                               value=0)

            input_ids = split_tensor_for_cp(input_ids, dim=-1)
            position_ids = split_tensor_for_cp(position_ids, dim=-1)
            attention_mask = split_tensor_for_cp(attention_mask, dim=-1)
            batch_labels = split_tensor_for_cp(batch_labels, dim=-1)

        inputs['input_ids'] = input_ids
        inputs['position_ids'] = position_ids
        inputs['attention_mask'] = attention_mask
        inputs['labels'] = batch_labels
        return inputs

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

