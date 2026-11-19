# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, Dict, List


class MCoreBridgeBackend:
    """The default backend: build a Megatron model through mcore-bridge.

    Beyond the forward loss, this code carries backward grad scaling
    (``calculate_per_token_loss``), MoE dispatch and fusion-flag/NPU correctness."""
    backend_name = 'mcore-bridge'

    @property
    def is_multimodal(self) -> bool:
        # TODO: MLLM
        return False

    def build_model_config(self, hf_config: Any, parallel_kwargs: Dict[str, Any], strategy: Any, **kwargs) -> Any:
        from mcore_bridge import ModelConfig, hf_to_mcore_config
        from twinkle import Platform
        from twinkle.model.megatron.strategy.megatron import finalize_model_grads_for_lora

        # Internal provenance used by MegatronBridgeBackend to distinguish an explicit user option
        # from a compatibility default. mcore's ModelConfig consumes the actual values, not this marker.
        kwargs.pop('_strict_model_kwargs', None)
        config_kwargs = hf_to_mcore_config(hf_config)
        # Preserve a checkpoint's classification head instead of constructing a
        # vocabulary-sized language-model output layer and skipping score weights.
        if any(name.endswith('ForSequenceClassification')
               for name in (getattr(hf_config, 'architectures', None) or [])):
            if kwargs.get('task_type', 'seq_cls') != 'seq_cls':
                raise ValueError('A sequence-classification checkpoint requires task_type=seq_cls.')
            if kwargs.get('num_labels', hf_config.num_labels) != hf_config.num_labels:
                raise ValueError('num_labels must match the persisted classification head.')
            kwargs.update(task_type='seq_cls', num_labels=hf_config.num_labels,
                          untie_embeddings_and_output_weights=True)
        config_kwargs.update(kwargs)
        # per-token-mean grad normalization (mcore default is False; twinkle forces True).
        if 'calculate_per_token_loss' not in config_kwargs:
            config_kwargs['calculate_per_token_loss'] = True
        # MoE dispatch: variable_seq_lengths gates alltoall vs allgather (MoE models only).
        if 'moe_token_dispatcher_type' not in config_kwargs:
            config_kwargs['moe_token_dispatcher_type'] = ('alltoall' if strategy.variable_seq_lengths else 'allgather')
        # Align fusion flags with legacy (mcore defaults them False); bias_activation_fusion
        # changes SwiGLU arithmetic in low precision. gradient_accumulation_fusion is excluded
        # because it hard-fails without APEX, which legacy tolerates.
        for _fusion_flag in ('bias_activation_fusion', 'masked_softmax_fusion', 'bias_dropout_fusion'):
            config_kwargs.setdefault(_fusion_flag, True)
        # This backend builds on CPU and moves shards to GPU in create_model, so it owns
        # use_cpu_initialization=True; drop any forwarded copy to avoid a duplicate keyword.
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
        # Platform hook: NPU's MindSpeed stack sets use_flash_attn and applies its
        # runtime args here; no-op on every other platform.
        Platform.get_platform().configure_model_config(model_config)
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

    @staticmethod
    def enable_text_vision_skip(model: Any) -> None:
        if getattr(model.config, 'hf_model_type', None) not in ('qwen3_5', 'qwen3_5_moe'):
            return
        visual = getattr(model, 'visual', None)
        if visual is None:
            return
        import torch

        get_inputs_embeds = visual.get_inputs_embeds

        def embed(inputs_embeds, **kwargs):
            if kwargs.get('pixel_values') is not None or kwargs.get('pixel_values_videos') is not None:
                return get_inputs_embeds(inputs_embeds, **kwargs)
            token_types = kwargs.get('mm_token_type_ids')
            if token_types is not None:
                if token_types.shape != kwargs['input_ids'].shape:
                    raise ValueError('mm_token_type_ids must match input_ids for vision embedding.')
                torch._assert_async((token_types == 0).all(), 'Vision placeholders require pixel values.')
            return inputs_embeds

        visual.get_inputs_embeds = embed
