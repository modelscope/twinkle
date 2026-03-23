# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import importlib.util
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_qwen35
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


try:
    from fla.modules import FusedRMSNormGated as _FLA_FUSED_RMS_NORM_GATED
    from fla.modules.convolution import causal_conv1d as _FLA_CAUSAL_CONV1D_FN
    from fla.modules.convolution import causal_conv1d_update as _FLA_CAUSAL_CONV1D_UPDATE
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _FLA_CHUNK_GATED_DELTA_RULE
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as _FLA_FUSED_RECURRENT_GATED_DELTA_RULE
except ImportError:
    _FLA_FUSED_RMS_NORM_GATED = None
    _FLA_CAUSAL_CONV1D_FN = None
    _FLA_CAUSAL_CONV1D_UPDATE = None
    _FLA_CHUNK_GATED_DELTA_RULE = None
    _FLA_FUSED_RECURRENT_GATED_DELTA_RULE = None

_HAS_CAUSAL_CONV1D = importlib.util.find_spec('causal_conv1d') is not None


def _ensure_text_config(config: Qwen3_5TextConfig) -> Qwen3_5TextConfig:
    if isinstance(config, Qwen3_5TextConfig):
        return config
    raise TypeError(
        'TwinkleQwen3_5 text-only models require transformers.models.qwen3_5.Qwen3_5TextConfig. '
        f'Got {type(config).__name__}.'
    )


def _ensure_linear_attention_fast_path() -> None:
    missing = []
    if _FLA_CAUSAL_CONV1D_FN is None:
        missing.append('fla.modules.convolution.causal_conv1d')
    if _FLA_CHUNK_GATED_DELTA_RULE is None or _FLA_FUSED_RECURRENT_GATED_DELTA_RULE is None:
        missing.append('fla.ops.gated_delta_rule')
    if not _HAS_CAUSAL_CONV1D:
        missing.append('causal-conv1d')
    if missing:
        raise ImportError(
            'TwinkleQwen3_5 linear attention requires flash-linear-attention and causal-conv1d. '
            f'Missing: {", ".join(missing)}'
        )


def _maybe_slice_tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _sp_is_enabled(sequence_parallel_context: Any | None) -> bool:
    return bool(
        sequence_parallel_context is not None
        and getattr(sequence_parallel_context, 'sp_world_size', 1) > 1
        and getattr(sequence_parallel_context, 'sp_group', None) is not None
    )


def _get_sp_rank(sequence_parallel_context: Any | None) -> int:
    if not _sp_is_enabled(sequence_parallel_context):
        return 0
    import torch.distributed as dist

    return dist.get_rank(sequence_parallel_context.sp_group)


def _seq_to_head_shard(tensor: torch.Tensor, sequence_parallel_context: Any | None) -> torch.Tensor:
    if not _sp_is_enabled(sequence_parallel_context):
        return tensor
    from twinkle.model.transformers.strategy.sequence_parallel import _SeqAllToAll

    if tensor.dim() == 3:
        return _SeqAllToAll.apply(sequence_parallel_context.sp_group, tensor.unsqueeze(-1), 2, 1).squeeze(-1)
    return _SeqAllToAll.apply(sequence_parallel_context.sp_group, tensor, 2, 1)


def _head_to_seq_shard(tensor: torch.Tensor, sequence_parallel_context: Any | None) -> torch.Tensor:
    if not _sp_is_enabled(sequence_parallel_context):
        return tensor
    from twinkle.model.transformers.strategy.sequence_parallel import _SeqAllToAll

    if tensor.dim() == 3:
        return _SeqAllToAll.apply(sequence_parallel_context.sp_group, tensor.unsqueeze(-1), 1, 2).squeeze(-1)
    return _SeqAllToAll.apply(sequence_parallel_context.sp_group, tensor, 1, 2)


class TwinkleQwen3_5GatedDeltaNet(hf_qwen35.Qwen3_5GatedDeltaNet):

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        _ensure_linear_attention_fast_path()
        super().__init__(config, layer_idx)
        self.causal_conv1d_fn = _FLA_CAUSAL_CONV1D_FN
        self.causal_conv1d_update = _FLA_CAUSAL_CONV1D_UPDATE or hf_qwen35.causal_conv1d_update
        self.chunk_gated_delta_rule = _FLA_CHUNK_GATED_DELTA_RULE
        self.recurrent_gated_delta_rule = _FLA_FUSED_RECURRENT_GATED_DELTA_RULE
        if _FLA_FUSED_RMS_NORM_GATED is not None and torch.cuda.is_available():
            self.norm = _FLA_FUSED_RMS_NORM_GATED(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )

    def _get_local_conv1d_weight(self, sp_rank: int, local_key_dim: int, local_value_dim: int) -> torch.Tensor:
        w_full = self.conv1d.weight.squeeze(1)
        key_offset = sp_rank * local_key_dim
        value_offset = sp_rank * local_value_dim
        w_q = w_full[key_offset:key_offset + local_key_dim]
        w_k = w_full[self.key_dim + key_offset:self.key_dim + key_offset + local_key_dim]
        w_v = w_full[2 * self.key_dim + value_offset:2 * self.key_dim + value_offset + local_value_dim]
        return torch.cat((w_q, w_k, w_v), dim=0)

    def _apply_varlen_conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        cu_seq_lens_q: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.causal_conv1d_fn is None:
            raise ImportError(
                'TwinkleQwen3_5 linear attention requires fla.modules.convolution.causal_conv1d for prefill/train.'
            )
        output = self.causal_conv1d_fn(
            x=mixed_qkv,
            weight=conv_weight,
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=None,
            backend='triton',
            cu_seqlens=cu_seq_lens_q,
        )
        return _maybe_slice_tensor_output(output)

    def _apply_decode_conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
    ) -> torch.Tensor:
        if self.causal_conv1d_update is None:
            raise ImportError(
                'TwinkleQwen3_5 decode requires a causal_conv1d_update implementation from flash-linear-attention '
                'or causal-conv1d.'
            )
        mixed_qkv_t = mixed_qkv.transpose(1, 2).contiguous()
        output = self.causal_conv1d_update(
            mixed_qkv_t,
            conv_state,
            conv_weight,
            self.conv1d.bias,
            self.activation,
        )
        output = _maybe_slice_tensor_output(output)
        if output.dim() == 2:
            output = output.unsqueeze(1)
        elif output.dim() == 3 and output.shape[1] == conv_weight.shape[0]:
            output = output.transpose(1, 2).contiguous()
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: hf_qwen35.Qwen3_5DynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        sequence_parallel_context: Any | None = None,
    ):
        hidden_states = hf_qwen35.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        else:
            conv_state = None
            recurrent_state = None

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        sp_enabled = _sp_is_enabled(sequence_parallel_context)
        if sp_enabled:
            sp_world_size = int(sequence_parallel_context.sp_world_size)
            if self.num_k_heads % sp_world_size != 0 or self.num_v_heads % sp_world_size != 0:
                raise RuntimeError(
                    'TwinkleQwen3_5 linear attention requires sp_world_size to divide both '
                    f'linear_num_key_heads ({self.num_k_heads}) and linear_num_value_heads ({self.num_v_heads}).'
                )
            local_num_k_heads = self.num_k_heads // sp_world_size
            local_num_v_heads = self.num_v_heads // sp_world_size
            local_key_dim = local_num_k_heads * self.head_k_dim
            local_value_dim = local_num_v_heads * self.head_v_dim

            q_proj, k_proj, v_proj = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q_proj = q_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
            k_proj = k_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
            v_proj = v_proj.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
            q_proj = _seq_to_head_shard(q_proj, sequence_parallel_context)
            k_proj = _seq_to_head_shard(k_proj, sequence_parallel_context)
            v_proj = _seq_to_head_shard(v_proj, sequence_parallel_context)
            b = _seq_to_head_shard(b.reshape(batch_size, seq_len, self.num_v_heads), sequence_parallel_context)
            a = _seq_to_head_shard(a.reshape(batch_size, seq_len, self.num_v_heads), sequence_parallel_context)

            mixed_qkv = torch.cat(
                (
                    q_proj.reshape(batch_size, q_proj.shape[1], local_key_dim),
                    k_proj.reshape(batch_size, k_proj.shape[1], local_key_dim),
                    v_proj.reshape(batch_size, v_proj.shape[1], local_value_dim),
                ),
                dim=-1,
            )
            conv_weight = self._get_local_conv1d_weight(_get_sp_rank(sequence_parallel_context), local_key_dim, local_value_dim)
        else:
            local_num_k_heads = self.num_k_heads
            local_num_v_heads = self.num_v_heads
            local_key_dim = self.key_dim
            local_value_dim = self.value_dim
            b = b.reshape(batch_size, seq_len, self.num_v_heads)
            a = a.reshape(batch_size, seq_len, self.num_v_heads)
            conv_weight = self.conv1d.weight.squeeze(1)

        if use_precomputed_states:
            if conv_state is None:
                raise RuntimeError('Qwen3.5 decode requires initialized convolution state.')
            mixed_qkv = self._apply_decode_conv(mixed_qkv, conv_state, conv_weight)
        else:
            if cache_params is not None:
                cache_params.conv_states[self.layer_idx] = F.pad(
                    mixed_qkv.transpose(1, 2).contiguous(),
                    (self.conv_kernel_size - mixed_qkv.shape[1], 0),
                )
            mixed_qkv = self._apply_varlen_conv(mixed_qkv, conv_weight, cu_seq_lens_q)

        query, key, value = torch.split(mixed_qkv, [local_key_dim, local_key_dim, local_value_dim], dim=-1)
        query = query.reshape(batch_size, query.shape[1], local_num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, key.shape[1], local_num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, value.shape[1], local_num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        if sp_enabled:
            head_offset = _get_sp_rank(sequence_parallel_context) * local_num_v_heads
            head_slice = slice(head_offset, head_offset + local_num_v_heads)
            g = -self.A_log[head_slice].float().exp() * F.softplus(a.float() + self.dt_bias[head_slice])
        else:
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        if use_precomputed_states:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seq_lens_q,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = _head_to_seq_shard(core_attn_out, sequence_parallel_context)
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.value_dim)
        return self.out_proj(core_attn_out)


class TwinkleQwen3_5DecoderLayer(hf_qwen35.Qwen3_5DecoderLayer):

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == 'linear_attention':
            self.linear_attn = TwinkleQwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == 'full_attention':
            self.self_attn = hf_qwen35.Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f'Unsupported Qwen3.5 layer_type={self.layer_type!r}')
        self.mlp = hf_qwen35.Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = hf_qwen35.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = hf_qwen35.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        sequence_parallel_context: Any | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == 'linear_attention':
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                sequence_parallel_context=sequence_parallel_context,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class TwinkleQwen3_5PreTrainedModel(hf_qwen35.Qwen3_5PreTrainedModel):
    config_class = Qwen3_5TextConfig
    _no_split_modules = ['TwinkleQwen3_5DecoderLayer']
    _can_record_outputs = {
        'hidden_states': TwinkleQwen3_5DecoderLayer,
        'attentions': hf_qwen35.Qwen3_5Attention,
    }


class TwinkleQwen3_5TextModel(TwinkleQwen3_5PreTrainedModel):

    def __init__(self, config: Qwen3_5TextConfig):
        config = _ensure_text_config(config)
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [TwinkleQwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = hf_qwen35.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = hf_qwen35.Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self._sequence_parallel_context = None
        self.requires_cu_seq_lens_q = any(layer_type == 'linear_attention' for layer_type in config.layer_types)
        self.post_init()

    def set_sequence_parallel_context(self, context: Any | None) -> None:
        self._sequence_parallel_context = context

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = hf_qwen35.Qwen3_5DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        causal_mask = hf_qwen35.create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        sp_context = self._sequence_parallel_context
        if _sp_is_enabled(sp_context) and self.requires_cu_seq_lens_q and cu_seq_lens_q is None:
            raise ValueError('TwinkleQwen3_5TextModel requires cu_seq_lens_q when sequence parallel is enabled.')

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            layer_mask = linear_attn_mask if decoder_layer.layer_type == 'linear_attention' else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                cu_seq_lens_q=cu_seq_lens_q if decoder_layer.layer_type == 'linear_attention' else None,
                sequence_parallel_context=sp_context if decoder_layer.layer_type == 'linear_attention' else None,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hf_qwen35.Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class TwinkleQwen3_5ForCausalLM(TwinkleQwen3_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {'lm_head.weight': 'model.embed_tokens.weight'}
    _tp_plan = {'lm_head': 'colwise_gather_output'}
    _pp_plan = {'lm_head': (['hidden_states'], ['logits'])}
    _keys_to_ignore_on_load_unexpected = [r'^mtp.*', r'^model\.visual.*']

    def __init__(self, config: Qwen3_5TextConfig):
        config = _ensure_text_config(config)
        super().__init__(config)
        self.model = TwinkleQwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_sequence_parallel_context(self, context: Any | None) -> None:
        self.model.set_sequence_parallel_context(context)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        cu_seq_lens_q: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            cu_seq_lens_q=cu_seq_lens_q,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
