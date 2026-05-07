import torch
from transformers.utils.import_utils import is_flash_linear_attention_available
from typing import Optional

from twinkle.patch import Patch


def _is_qwen35_model(hf_config) -> bool:
    return 'qwen3_5' in getattr(hf_config, 'model_type', '')


def _find_qwen35_classes(module: Optional[torch.nn.Module], hf_config, enable_sp: bool):
    if module is None or enable_sp or not _is_qwen35_model(hf_config):
        return None, None
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet
    except Exception:
        return None, None
    if any(isinstance(submodule, Qwen3_5GatedDeltaNet) for submodule in module.modules()):
        return Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet
    return None, None


def _get_flash_linear_attention_kernels():
    if not is_flash_linear_attention_available():
        raise NotImplementedError(
            'padding_free/packed inputs require flash-linear-attention for GatedDeltaNet. '
            'The native torch GatedDeltaNet implementation does not reset linear-attention state at packed '
            'sequence boundaries. Please install flash-linear-attention or disable padding_free/packing.')
    from fla.modules.convolution import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return causal_conv1d, chunk_gated_delta_rule


def _patch_gdn_kernels_for_cu_seqlens(
    mod: torch.nn.Module,
    *,
    cu_seqlens: torch.Tensor,
    origin_forward,
    forward_args,
    forward_kwargs,
) -> torch.Tensor:
    causal_conv1d, chunk_gated_delta_rule = _get_flash_linear_attention_kernels()
    old_conv_fn = mod.causal_conv1d_fn
    old_chunk_rule = mod.chunk_gated_delta_rule

    def causal_conv1d_wrapper(*args, **kwargs):
        x = kwargs.pop('x')
        output = causal_conv1d(
            *args,
            x=x.transpose(1, 2).contiguous(),
            cu_seqlens=cu_seqlens.to(dtype=torch.int32, device=x.device),
            **kwargs,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.transpose(1, 2).contiguous()

    def chunk_gated_delta_rule_wrapper(query, key, value, **kwargs):
        kwargs['cu_seqlens'] = cu_seqlens.to(dtype=torch.int32, device=query.device)
        return chunk_gated_delta_rule(query, key, value, **kwargs)

    mod.causal_conv1d_fn = causal_conv1d_wrapper
    mod.chunk_gated_delta_rule = chunk_gated_delta_rule_wrapper
    try:
        return origin_forward(mod, *forward_args, **forward_kwargs)
    finally:
        mod.causal_conv1d_fn = old_conv_fn
        mod.chunk_gated_delta_rule = old_chunk_rule


class GatedDeltaNetPaddingFreePatch(Patch):

    def __call__(self, module, *args, **kwargs):
        del args
        Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet = _find_qwen35_classes(
            module,
            kwargs.get('hf_config'),
            bool(kwargs.get('enable_sp', False)),
        )
        if Qwen3_5DecoderLayer is None or Qwen3_5GatedDeltaNet is None:
            return
        if getattr(Qwen3_5GatedDeltaNet, '_twinkle_sp_linear_patched', False):
            return
        module._twinkle_gdn_padding_free_patched = True

        if not getattr(Qwen3_5DecoderLayer, '_twinkle_padding_free_cu_seqlens_patched', False):
            origin_decoder_forward = Qwen3_5DecoderLayer.forward

            def decoder_forward(
                layer,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values=None,
                cache_position: Optional[torch.Tensor] = None,
                **extra_kwargs,
            ):
                if getattr(layer, 'layer_type', None) != 'linear_attention':
                    return origin_decoder_forward(
                        layer,
                        hidden_states=hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        **extra_kwargs,
                    )
                cu_seq_lens_q = extra_kwargs.pop('cu_seq_lens_q', None)
                extra_kwargs.pop('cu_seq_lens_k', None)
                extra_kwargs.pop('max_length_q', None)
                extra_kwargs.pop('max_length_k', None)

                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                hidden_states = layer.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=past_key_values,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    cu_seq_lens_q=cu_seq_lens_q,
                )
                hidden_states = residual + hidden_states

                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states
                return hidden_states

            Qwen3_5DecoderLayer.forward = decoder_forward
            Qwen3_5DecoderLayer._twinkle_padding_free_cu_seqlens_patched = True

        if not getattr(Qwen3_5GatedDeltaNet, '_twinkle_padding_free_gdn_patched', False):
            origin_forward = Qwen3_5GatedDeltaNet.forward

            def forward(
                mod,
                hidden_states: torch.Tensor,
                cache_params=None,
                cache_position=None,
                attention_mask: Optional[torch.Tensor] = None,
                cu_seq_lens_q: Optional[torch.Tensor] = None,
                **extra_kwargs,
            ):
                if cu_seq_lens_q is None:
                    return origin_forward(
                        mod,
                        hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                        **extra_kwargs,
                    )
                return _patch_gdn_kernels_for_cu_seqlens(
                    mod,
                    cu_seqlens=cu_seq_lens_q,
                    origin_forward=origin_forward,
                    forward_args=(hidden_states, ),
                    forward_kwargs={
                        'cache_params': cache_params,
                        'cache_position': cache_position,
                        'attention_mask': attention_mask,
                        **extra_kwargs,
                    },
                )

            Qwen3_5GatedDeltaNet.forward = forward
            Qwen3_5GatedDeltaNet._twinkle_padding_free_gdn_patched = True
