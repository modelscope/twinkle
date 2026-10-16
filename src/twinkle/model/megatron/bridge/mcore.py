# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, Dict, List


class MCoreBridgeBackend:
    """The default backend: build a Megatron model through mcore-bridge.

    ``MegatronStrategy.get_model_config`` / ``create_megatron_model`` delegate here, so this is the
    construction path a plain strategy run takes. Every branch matters beyond the forward loss a
    bit-match test can see: backward grad scaling (``calculate_per_token_loss``), MoE dispatch
    (``moe_token_dispatcher_type``), kernel fusion and NPU flash attention are all invisible to a
    dense/GPU/forward-only check, so their correctness is carried by this code rather than a test.
    """
    backend_name = 'mcore-bridge'

    @property
    def is_multimodal(self) -> bool:
        # TODO: MLLM
        return False

    def build_model_config(self, hf_config: Any, parallel_kwargs: Dict[str, Any], strategy: Any, **kwargs) -> Any:
        from mcore_bridge import ModelConfig, hf_to_mcore_config
        from twinkle import Platform
        from twinkle.model.megatron._mindspeed_runtime import configure_mindspeed_runtime_args
        from twinkle.model.megatron.strategy.megatron import finalize_model_grads_for_lora

        # Internal provenance used by MegatronBridgeBackend to distinguish an explicit user option
        # from a compatibility default. mcore's ModelConfig consumes the actual values, not this marker.
        kwargs.pop('_strict_model_kwargs', None)
        config_kwargs = hf_to_mcore_config(hf_config)
        config_kwargs.update(kwargs)
        # per-token-mean grad normalization (mcore default is False; twinkle forces True).
        if 'calculate_per_token_loss' not in config_kwargs:
            config_kwargs['calculate_per_token_loss'] = True
        # MoE dispatch: variable_seq_lengths gates alltoall vs allgather (MoE models only).
        if 'moe_token_dispatcher_type' not in config_kwargs:
            config_kwargs['moe_token_dispatcher_type'] = ('alltoall' if strategy.variable_seq_lengths else 'allgather')
        # Align fusion flags with legacy: mcore's TransformerConfig defaults them False, while legacy
        # Megatron-SWIFT defaults them True and copies that onto ModelConfig. Leaving them unset makes
        # the run use unfused kernels where legacy uses fused ones, which is not numerically equivalent
        # in low precision (notably bias_activation_fusion's SwiGLU path). gradient_accumulation_fusion
        # is excluded on purpose: it hard-fails without the optional APEX extension, whereas legacy
        # falls back to unfused, so forcing it here would break setups legacy tolerates.
        for _fusion_flag in ('bias_activation_fusion', 'masked_softmax_fusion', 'bias_dropout_fusion'):
            config_kwargs.setdefault(_fusion_flag, True)
        # This backend builds the model on CPU and moves the shards to GPU in create_model, so it owns
        # use_cpu_initialization=True. MegatronConfig also exposes a use_cpu_initialization field which
        # forwarded model kwargs can carry into config_kwargs, so passing it explicitly below as well
        # would give ModelConfig two values for the same keyword. Drop the forwarded one and let the
        # bridge's required value win. (The megatron-bridge backend folds config kwargs by hasattr
        # instead and never double-passes, so it needs no such guard.)
        config_kwargs.pop('use_cpu_initialization', None)
        model_config = ModelConfig(
            use_cpu_initialization=True,
            params_dtype=strategy.params_type,
            sequence_parallel=strategy.sequence_parallel,
            finalize_model_grads_func=finalize_model_grads_for_lora,
            variable_seq_lengths=strategy.variable_seq_lengths,
            **parallel_kwargs,
            **config_kwargs,
        )
        # NPU: MindSpeed's patched TE attention needs use_flash_attn to synthesize its own
        # compressed causal mask; unset aborts the first 8-card forward (NPU-only, no GPU effect).
        if Platform.device_prefix() == 'npu':
            model_config.use_flash_attn = True
        configure_mindspeed_runtime_args(model_config)
        return model_config

    def create_model(self, config: Any, model_dir: str, *, load_weights: bool, move_to_gpu) -> List[Any]:
        import torch.distributed as dist
        from mcore_bridge import get_mcore_model

        mg_models = get_mcore_model(config)
        if dist.is_initialized():
            dist.barrier()

        models = [move_to_gpu(m) for m in mg_models]

        if load_weights:
            bridge = config.bridge
            bridge.load_weights(mg_models, model_dir)
        return models
