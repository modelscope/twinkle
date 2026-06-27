import torch
import transformers
from importlib import import_module
from packaging.version import Version
from transformers.utils.import_utils import is_flash_linear_attention_available
from typing import Optional

from twinkle.patch import Patch
from twinkle.utils.utils import call_with_supported_kwargs


def _is_qwen35_model(hf_config) -> bool:
    return 'qwen3_5' in getattr(hf_config, 'model_type', '')


def _iter_qwen35_class_pairs():
    class_specs = (
        (
            'transformers.models.qwen3_5.modeling_qwen3_5',
            'Qwen3_5DecoderLayer',
            'Qwen3_5GatedDeltaNet',
        ),
        (
            'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe',
            'Qwen3_5MoeDecoderLayer',
            'Qwen3_5MoeGatedDeltaNet',
        ),
    )
    for module_name, decoder_class_name, gdn_class_name in class_specs:
        try:
            modeling_module = import_module(module_name)
            yield getattr(modeling_module, decoder_class_name), getattr(modeling_module, gdn_class_name)
        except Exception:
            continue


def _find_qwen35_class_pairs(module: Optional[torch.nn.Module], hf_config, enable_sp: bool):
    if module is None or enable_sp or not _is_qwen35_model(hf_config):
        return ()
    class_pairs = []
    for decoder_layer_cls, gated_delta_net_cls in _iter_qwen35_class_pairs():
        if any(isinstance(submodule, gated_delta_net_cls) for submodule in module.modules()):
            class_pairs.append((decoder_layer_cls, gated_delta_net_cls))
    return tuple(class_pairs)


def _get_flash_linear_attention_kernels():
    if not is_flash_linear_attention_available():
        raise NotImplementedError(
            'padding_free/packed inputs require flash-linear-attention for GatedDeltaNet. '
            'The native torch GatedDeltaNet implementation does not reset linear-attention state at packed '
            'sequence boundaries. Please install flash-linear-attention or disable padding_free/packing.')
    from fla.modules.convolution import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return causal_conv1d, chunk_gated_delta_rule


def _get_mindspeed_ops_causal_conv1d():
    from twinkle.kernel.causal_conv1d import causal_conv1d as _ms_causal_conv1d
    return _ms_causal_conv1d


def _needs_chunk_gated_delta_rule_cu_seqlens_patch() -> bool:
    return Version(transformers.__version__) < Version('5.9.0')


def _patch_gdn_kernels_for_cu_seqlens(
    mod: torch.nn.Module,
    *,
    cu_seqlens: torch.Tensor,
    patch_chunk_rule: bool,
    origin_forward,
    forward_args,
    forward_kwargs,
) -> torch.Tensor:
    is_npu = getattr(mod, '_twinkle_npu_patched', False)
    if is_npu:
        from twinkle.kernel.causal_conv1d import npu_causal_conv1d_fn
    else:
        causal_conv1d, chunk_gated_delta_rule = _get_flash_linear_attention_kernels()

    old_conv_fn = mod.causal_conv1d_fn
    old_chunk_rule = mod.chunk_gated_delta_rule

    if is_npu:

        def causal_conv1d_wrapper(*args, **kwargs):
            x = kwargs.pop('x')
            del kwargs['seq_idx']
            del kwargs['backend']

            if len(args) > 0:
                kwargs['weight'] = args[0]
                args = args[1:]
            if len(args) > 0:
                kwargs['bias'] = args[0]
            return npu_causal_conv1d_fn(
                x=x,
                cu_seqlens=cu_seqlens.to(dtype=torch.int32, device=x.device),
                **kwargs,
            )
    else:

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

    if is_npu:

        def chunk_gated_delta_rule_wrapper(query, key, value, **kwargs):
            kwargs['cu_seqlens'] = cu_seqlens.to(dtype=torch.int32, device=query.device)
            return old_chunk_rule(query, key, value, **kwargs)
    else:

        def chunk_gated_delta_rule_wrapper(query, key, value, **kwargs):
            kwargs['cu_seqlens'] = cu_seqlens.to(dtype=torch.int32, device=query.device)
            return chunk_gated_delta_rule(query, key, value, **kwargs)

    mod.causal_conv1d_fn = causal_conv1d_wrapper
    if patch_chunk_rule:
        mod.chunk_gated_delta_rule = chunk_gated_delta_rule_wrapper
    try:
        return call_with_supported_kwargs(origin_forward, mod, *forward_args, **forward_kwargs)
    finally:
        mod.causal_conv1d_fn = old_conv_fn
        if patch_chunk_rule:
            mod.chunk_gated_delta_rule = old_chunk_rule


class GatedDeltaNetPaddingFreePatch(Patch):

    def __call__(self, module, *args, **kwargs):
        del args
        qwen35_class_pairs = _find_qwen35_class_pairs(
            module,
            kwargs.get('hf_config'),
            bool(kwargs.get('enable_sp', False)),
        )
        if not qwen35_class_pairs:
            return
        module._twinkle_gdn_padding_free_patched = True

        for decoder_layer_cls, gated_delta_net_cls in qwen35_class_pairs:
            if getattr(gated_delta_net_cls, '_twinkle_sp_linear_patched', False):
                continue

            if not getattr(decoder_layer_cls, '_twinkle_padding_free_cu_seqlens_patched', False):
                origin_decoder_forward = decoder_layer_cls.forward

                def decoder_forward(
                    layer,
                    hidden_states: torch.Tensor,
                    position_embeddings: tuple[torch.Tensor, torch.Tensor],
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.Tensor] = None,
                    past_key_values=None,
                    cache_position: Optional[torch.Tensor] = None,
                    _origin_decoder_forward=origin_decoder_forward,
                    **extra_kwargs,
                ):
                    if getattr(layer, 'layer_type', None) != 'linear_attention':
                        return call_with_supported_kwargs(
                            _origin_decoder_forward,
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
                        **extra_kwargs,
                    )
                    hidden_states = residual + hidden_states

                    residual = hidden_states
                    hidden_states = layer.post_attention_layernorm(hidden_states)
                    hidden_states = layer.mlp(hidden_states)
                    if isinstance(hidden_states, tuple):
                        hidden_states, _ = hidden_states
                    hidden_states = residual + hidden_states
                    return hidden_states

                decoder_layer_cls.forward = decoder_forward
                decoder_layer_cls._twinkle_padding_free_cu_seqlens_patched = True

            if not getattr(gated_delta_net_cls, '_twinkle_padding_free_gdn_patched', False):
                origin_forward = gated_delta_net_cls.forward
                patch_chunk_rule = _needs_chunk_gated_delta_rule_cu_seqlens_patch()

                def forward(
                    mod,
                    hidden_states: torch.Tensor,
                    cache_params=None,
                    cache_position=None,
                    attention_mask: Optional[torch.Tensor] = None,
                    cu_seq_lens_q: Optional[torch.Tensor] = None,
                    _origin_forward=origin_forward,
                    _patch_chunk_rule=patch_chunk_rule,
                    **extra_kwargs,
                ):
                    if cu_seq_lens_q is None:
                        return call_with_supported_kwargs(
                            _origin_forward,
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
                        patch_chunk_rule=_patch_chunk_rule,
                        origin_forward=_origin_forward,
                        forward_args=(hidden_states, ),
                        forward_kwargs={
                            'cache_params': cache_params,
                            'cache_position': cache_position,
                            'attention_mask': attention_mask,
                            'cu_seq_lens_q': cu_seq_lens_q,
                            **extra_kwargs,
                        },
                    )

                gated_delta_net_cls.forward = forward
                gated_delta_net_cls._twinkle_padding_free_gdn_patched = True
