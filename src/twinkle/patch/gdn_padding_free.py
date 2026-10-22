import torch
from inspect import unwrap
from transformers.utils.import_utils import is_flash_linear_attention_available
from typing import Optional

from twinkle.patch import Patch
from twinkle.utils.utils import call_with_supported_kwargs


def _find_gdn_class_pairs(module: Optional[torch.nn.Module], enable_sp: bool):
    if module is None or enable_sp:
        return ()
    supported_names = {'Qwen3_5GatedDeltaNet', 'Qwen3_5MoeGatedDeltaNet'}
    class_pairs = []
    for layer in module.modules():
        gdn = getattr(layer, 'linear_attn', None)
        if not isinstance(gdn, torch.nn.Module):
            continue
        if type(gdn).__name__ not in supported_names:
            continue
        # Match concrete class names without inheriting support through the MRO.
        # No Transformers internal import path is needed.
        pair = (type(layer), type(gdn))
        if pair not in class_pairs:
            class_pairs.append(pair)
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


def _patch_gdn_kernels_for_cu_seqlens(
    mod: torch.nn.Module,
    *,
    cu_seqlens: torch.Tensor,
    origin_forward,
    forward_args,
    forward_kwargs,
) -> torch.Tensor:
    is_npu = getattr(mod, '_twinkle_npu_patched', False)
    if is_npu:
        from twinkle.kernel.ops.fla.npu import npu_causal_conv1d_fn
    else:
        causal_conv1d, chunk_gated_delta_rule = _get_flash_linear_attention_kernels()

    # Older Transformers stores callables on the instance; newer versions call
    # modeling globals. Prefer instance attributes, including a None conv fallback.
    modeling_globals = unwrap(origin_forward).__globals__
    conv_namespace = mod.__dict__ if 'causal_conv1d_fn' in mod.__dict__ else modeling_globals
    chunk_namespace = mod.__dict__ if 'chunk_gated_delta_rule' in mod.__dict__ else modeling_globals
    chunk_name = 'chunk_gated_delta_rule' if chunk_namespace is mod.__dict__ else 'torch_chunk_gated_delta_rule'
    old_conv_fn = conv_namespace['causal_conv1d_fn']
    old_chunk_rule = chunk_namespace[chunk_name]

    if is_npu:

        def causal_conv1d_wrapper(hidden_states=None, weight=None, bias=None, activation=None, **kwargs):
            if hidden_states is None:
                hidden_states = kwargs.pop('x')
            kwargs.pop('seq_idx', None)
            kwargs.pop('backend', None)
            return npu_causal_conv1d_fn(
                x=hidden_states,
                weight=weight,
                bias=bias,
                activation=activation,
                cu_seqlens=cu_seqlens.to(dtype=torch.int32, device=hidden_states.device),
            )
    else:

        def causal_conv1d_wrapper(hidden_states=None, weight=None, bias=None, activation=None, **kwargs):
            if hidden_states is None:
                hidden_states = kwargs.pop('x')
            kwargs.pop('seq_idx', None)
            kwargs.pop('backend', None)
            output = causal_conv1d(
                x=hidden_states.transpose(1, 2).contiguous(),
                weight=weight,
                bias=bias,
                activation=activation,
                cu_seqlens=cu_seqlens.to(dtype=torch.int32, device=hidden_states.device),
                backend='triton',
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

    conv_namespace['causal_conv1d_fn'] = causal_conv1d_wrapper
    chunk_namespace[chunk_name] = chunk_gated_delta_rule_wrapper
    try:
        return call_with_supported_kwargs(origin_forward, mod, *forward_args, **forward_kwargs)
    finally:
        conv_namespace['causal_conv1d_fn'] = old_conv_fn
        chunk_namespace[chunk_name] = old_chunk_rule


class GatedDeltaNetPaddingFreePatch(Patch):

    @staticmethod
    def is_applicable(module: torch.nn.Module, *, enable_sp: bool = False) -> bool:
        """Detect supported GatedDeltaNet layers by type, rather than model names."""
        return bool(_find_gdn_class_pairs(module, enable_sp))

    def __call__(self, module, *args, **kwargs):
        del args
        gdn_class_pairs = _find_gdn_class_pairs(
            module,
            bool(kwargs.get('enable_sp', False)),
        )
        if not gdn_class_pairs:
            return
        module._twinkle_gdn_padding_free_patched = True

        for decoder_layer_cls, gated_delta_net_cls in gdn_class_pairs:
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
                    if (getattr(layer, 'layer_type', None) != 'linear_attention'
                            or extra_kwargs.get('cu_seq_lens_q') is None):
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

                def forward(
                    mod,
                    hidden_states: torch.Tensor,
                    cache_params=None,
                    cache_position=None,
                    attention_mask: Optional[torch.Tensor] = None,
                    cu_seq_lens_q: Optional[torch.Tensor] = None,
                    _origin_forward=origin_forward,
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
