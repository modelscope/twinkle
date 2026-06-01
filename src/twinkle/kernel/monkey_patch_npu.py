"""NPU monkey patches for Ascend hardware acceleration.

Unified entry point::

    >>> from twinkle.kernel.monkey_patch_npu import apply_npu_patch
    >>> if Torch.is_npu_available():
    ...     apply_npu_patch(model)
"""

import importlib
import os
import torch
import torch.nn.functional as F
from torch import nn
from transformers.utils import is_torch_npu_available

from twinkle import get_logger

logger = get_logger(__name__)

_is_torch_npu_available = is_torch_npu_available()
_NPU_PATCH_APPLIED = False

if _is_torch_npu_available:
    import torch_npu

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def import_optional_module(module_name: str):
    """Import a module, returning None if unavailable."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        logger.debug('Failed to import optional module %s: %s', module_name, exc)
        return None


def _resolve_unsqueeze_dim(position_ids=None, unsqueeze_dim=1):
    if isinstance(position_ids, int) and unsqueeze_dim == 1:
        return position_ids
    return unsqueeze_dim


def _is_ep_enabled(model=None) -> bool:
    r"""Check whether Expert Parallelism (EP) is enabled.

    EP is detected via ``device_mesh.ep_size > 1``.
    When EP is active, each rank holds only a subset of expert weights,
    making ``npu_grouped_matmul`` efficient (small contiguous weights).
    """
    device_mesh = getattr(model, 'device_mesh', None)
    if device_mesh is None:
        return False
    return (getattr(device_mesh, 'ep_size', None) or 0) > 1


# =============================================================================
# Section 1: MoE Grouped MatMul (GMM)
# =============================================================================


class GmmFunction(torch.autograd.Function):
    r"""Custom autograd function for NPU grouped matrix multiplication."""

    @staticmethod
    def forward(ctx, x: torch.tensor, group_list: torch.tensor, weight_ekn: torch.tensor):
        group_list = group_list.to(torch.int64)
        ctx.save_for_backward(x, group_list, weight_ekn)
        outputs = torch_npu.npu_grouped_matmul(
            [x],
            [weight_ekn],
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        x, group_list, weight_ekn = ctx.saved_tensors
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight_ekn.transpose(-2, -1).contiguous()],
            bias=None,
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )[0]
        grad_weight = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=group_list,
            group_type=2,
            split_item=3,
            group_list_type=1,
        )[0]
        return grad_input, None, grad_weight.contiguous()


def _grouped_mm_npu(input: torch.tensor, weight_ekn: torch.tensor, offs: torch.tensor) -> torch.tensor:
    counts = torch.empty_like(offs)
    counts[0] = offs[0]
    if offs.numel() > 1:
        counts[1:] = offs[1:] - offs[:-1]
    counts = counts.to(torch.int64)
    return GmmFunction.apply(input, counts, weight_ekn)


def _apply_hf_moe_grouped_mm_patch(model=None) -> None:
    r"""Patch HuggingFace MoE integration to use NPU grouped matmul.

    When Expert Parallelism (EP) is **not** enabled, each rank holds **all**
    expert weights.  ``weight.transpose(-2, -1)`` then produces a large
    non-contiguous view that ``npu_grouped_matmul`` forces to ``.contiguous()``
    (~12.88 GB per MoE layer), creating a bandwidth bottleneck that makes the
    NPU patch **slower** than the native per-expert fallback (~8x overhead).

    Detection logic:
      - ``TWINKLE_NPU_GMM_PATCH`` not set → **skip** the patch by default.
      - ``TWINKLE_NPU_GMM_PATCH=1`` → EP-aware: apply only if EP is enabled
        (each rank has few experts, weights are small and contiguous);
        skip if EP is **not** enabled (avoid ~8x overhead).
      - ``TWINKLE_NPU_GMM_PATCH=0`` → **disable** the patch regardless.
    """
    moe_enabled = _is_env_enabled('TWINKLE_NPU_GMM_PATCH', default=False)

    if not moe_enabled:
        has_native_gmm = hasattr(torch.nn.functional, 'grouped_mm')
        logger.info(
            '[PATCH] TWINKLE_NPU_GMM_PATCH not set: MoE GMM patch skipped by default. '
            'Set TWINKLE_NPU_GMM_PATCH=1 to enable (EP-aware). '
            'Native grouped_mm available: %s.',
            has_native_gmm,
        )
        return

    if not _is_ep_enabled(model):
        has_native_gmm = hasattr(torch.nn.functional, 'grouped_mm')
        logger.info(
            '[PATCH] TWINKLE_NPU_GMM_PATCH=1 but EP not enabled (all experts on each rank) — '
            'skipping _grouped_mm_npu patch to avoid ~8x overhead from '
            'contiguous copies on transposed weights. '
            'Native grouped_mm available: %s.',
            has_native_gmm,
        )
        return

    import transformers.integrations.moe as hf_moe
    hf_moe._grouped_mm = _grouped_mm_npu
    logger.info('[PATCH] transformers.integrations.moe._grouped_mm -> _grouped_mm_npu')


# =============================================================================
# Section 1b: MoE Packed Experts
# =============================================================================


def _normalize_packed_expert_weights(module, input_dtype: torch.dtype, hidden_dim: int):
    """Normalize packed expert weight shapes for NPU grouped matmul."""
    gate_up_proj = module.gate_up_proj.to(input_dtype)
    down_proj = module.down_proj.to(input_dtype)

    if gate_up_proj.shape[1] == hidden_dim:
        gate_up_weight = gate_up_proj
    elif gate_up_proj.shape[2] == hidden_dim:
        gate_up_weight = gate_up_proj.transpose(1, 2)
    else:
        raise RuntimeError(f'Unsupported gate_up_proj shape for NPU MoE patch: {tuple(gate_up_proj.shape)}.')

    if down_proj.shape[2] == hidden_dim:
        down_weight = down_proj
    elif down_proj.shape[1] == hidden_dim:
        down_weight = down_proj.transpose(1, 2)
    else:
        raise RuntimeError(f'Unsupported down_proj shape for NPU MoE patch: {tuple(down_proj.shape)}.')

    return gate_up_weight, down_weight


def _get_cached_expert_weights(self, target_dtype: torch.dtype, hidden_dim: int):
    """Return normalized expert weights with automatic cache invalidation.

    Cache key combines (dtype, gate_version, down_version). This correctly
    handles:
      - Full-parameter training: optimizer in-place updates bump _version
      - LoRA training: frozen weights keep _version stable, cache persists
      - Inference: cache is permanent
      - AMP autocast: separate cache per dtype

    Safety: when weights require gradients, the cache is bypassed to avoid
    breaking the PyTorch autograd graph (non-leaf tensors from .to() cannot
    be reused across forward passes).
    """
    requires_grad = (
        getattr(self.gate_up_proj, 'requires_grad', False) or getattr(self.down_proj, 'requires_grad', False))
    cache_attr = '_npu_expert_cache'
    if not requires_grad and hasattr(self, cache_attr):
        cached_dtype, cached_gate_ver, cached_down_ver, cached = getattr(self, cache_attr)
        if (cached_dtype == target_dtype and cached_gate_ver == self.gate_up_proj._version
                and cached_down_ver == self.down_proj._version):
            return cached

    weights = _normalize_packed_expert_weights(self, target_dtype, hidden_dim)
    if not requires_grad:
        setattr(
            self,
            cache_attr,
            (target_dtype, self.gate_up_proj._version, self.down_proj._version, weights),
        )
    return weights


def npu_packed_moe_experts_forward(
    self,
    hidden_states: torch.Tensor,
    router_indices_or_routing_weights: torch.Tensor,
    routing_weights_or_router_indices: torch.Tensor,
) -> torch.Tensor:
    """Packed MoE experts forward using NPU grouped matmul.

    Compatible with Qwen3-MoE, Qwen3.5-MoE, and any model using packed experts
    with the standard ``(hidden_states, router_indices, routing_weights)`` call convention.
    """
    if router_indices_or_routing_weights.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        router_indices = router_indices_or_routing_weights
        routing_weights = routing_weights_or_router_indices
    else:
        routing_weights = router_indices_or_routing_weights
        router_indices = routing_weights_or_router_indices

    output_shape = hidden_states.shape
    hidden_dim = output_shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)

    if routing_weights.shape != router_indices.shape:
        routing_weights = torch.gather(routing_weights, dim=-1, index=router_indices.to(torch.long))
    routing_weights = routing_weights.to(hidden_states.dtype)
    router_indices = router_indices.to(torch.int32)

    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, router_indices)
    tokens_per_expert = torch.bincount(router_indices.view(-1), minlength=self.num_experts).to(torch.int64)

    # Cached normalized weights: auto-invalidates on weight updates (full-param)
    # and persists when frozen (LoRA / inference).
    gate_up_weight, down_weight = _get_cached_expert_weights(self, hidden_states.dtype, hidden_dim)

    intermediate_hidden_states = GmmFunction.apply(permuted_hidden_states, tokens_per_expert, gate_up_weight)
    intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
    output = GmmFunction.apply(intermediate_activations, tokens_per_expert, down_weight)
    next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
    return next_states.view(*output_shape)


# =============================================================================
# Section 1c: MoE Sparse Block
# =============================================================================


def _topk_from_router_logits(module, hidden_states: torch.Tensor, router_logits: torch.Tensor):
    """Compute top-k routing from router logits (Transformers 4.x style)."""
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, router_indices = torch.topk(routing_weights, module.top_k, dim=-1)
    if getattr(module, 'norm_topk_prob', True):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    return routing_weights, router_indices


def _add_shared_expert(self, hidden_states: torch.Tensor, expert_output: torch.Tensor) -> torch.Tensor:
    """Add shared expert output with sigmoid gating.

    Automatically skips if the module lacks shared_expert / shared_expert_gate.
    """
    if not (hasattr(self, 'shared_expert') and hasattr(self, 'shared_expert_gate')):
        return expert_output

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = (F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output)
    return expert_output + shared_expert_output


def _qwen3_5_moe_forward_transformers_5(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor,
                                        selected_experts: torch.Tensor) -> torch.Tensor:
    """Transformers 5.x path: gate returns (router_logits, routing_weights, selected_experts)."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    expert_output = self.experts(hidden_states, selected_experts, routing_weights)
    expert_output = _add_shared_expert(self, hidden_states, expert_output)
    return expert_output.reshape(batch_size, sequence_length, hidden_dim)


def _qwen3_5_moe_forward_linear_gate(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    """Transformers 4.x path: gate is nn.Linear and returns router logits."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    routing_weights, selected_experts = _topk_from_router_logits(self, hidden_states, router_logits)
    expert_output = self.experts(hidden_states, selected_experts, routing_weights)
    expert_output = _add_shared_expert(self, hidden_states, expert_output)
    return expert_output.reshape(batch_size, sequence_length, hidden_dim)


def npu_qwen3_5_moe_sparse_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """NPU-accelerated SparseMoeBlock forward with dual Transformers version support."""
    hidden_dim = hidden_states.shape[-1]
    gate_output = self.gate(hidden_states.view(-1, hidden_dim))

    if isinstance(gate_output, tuple):
        _, routing_weights, selected_experts = gate_output
        return _qwen3_5_moe_forward_transformers_5(self, hidden_states, routing_weights, selected_experts)

    return _qwen3_5_moe_forward_linear_gate(self, hidden_states, gate_output)


# =============================================================================
# Section 2: Fused Operators (RMSNorm / RoPE / SwiGLU / SDPA)
# =============================================================================


class NpuRMSNorm(nn.Module):
    r"""Fused RMSNorm via ``torch_npu.npu_rms_norm``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        # Detect residual parameterization (e.g. Qwen3.5: scale = 1.0 + weight)
        # once at initialization to avoid CPU-synchronizing Tensor.item() calls.
        self._residual_param = abs(self.weight.data.mean().item()) < 0.3
        if self._residual_param:
            logger.debug('[NPU] NpuRMSNorm using residual parameterization (1.0 + weight)')

    def _get_effective_weight(self, target_dtype: torch.dtype):
        if self._residual_param:
            return (1.0 + self.weight).to(dtype=target_dtype)
        return self.weight.to(dtype=target_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scale = self._get_effective_weight(hidden_states.dtype)
        return torch_npu.npu_rms_norm(hidden_states, scale, epsilon=self.variance_epsilon)[0]

    def extra_repr(self) -> str:
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'


def npu_gated_rms_norm_forward(self, hidden_states, gate=None):
    """NPU forward for Gated RMSNorm.

    The FP32 mode is controlled by ``TWINKLE_NPU_GATED_RMSNorm_FP32``,
    resolved once during patching and stored in ``self._twinkle_force_fp32``.
    """
    input_dtype = hidden_states.dtype
    _eps = getattr(self, 'variance_epsilon', getattr(self, 'eps', 1e-6))

    # Read the cached flag; no env lookup in the hot path.
    force_fp32 = getattr(self, '_twinkle_force_fp32', False)
    if force_fp32:
        hidden_states = hidden_states.to(torch.float32)
        weight = self.weight.float()
        gate = gate.to(torch.float32) if gate is not None else None
    else:
        weight = self.weight

    hidden_states = torch_npu.npu_rms_norm(hidden_states, weight, epsilon=_eps)[0]

    if gate is not None:
        hidden_states = hidden_states * F.silu(gate)

    return hidden_states.to(input_dtype)


def _make_apply_npu_rotary_emb():
    _cached_partial = {}

    def _apply_npu_rotary_emb(q, k, cos, sin):
        rotary_dim = cos.shape[-1]
        query_dim = q.shape[-1]
        shape_key = (rotary_dim, query_dim)

        use_partial = _cached_partial.get(shape_key)
        if use_partial is None:
            use_partial = rotary_dim < query_dim
            _cached_partial[shape_key] = use_partial

        if use_partial:
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin).to(q.dtype)
            k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin).to(k.dtype)
            q_embed = torch.cat([q_embed, q_pass], dim=-1)
            k_embed = torch.cat([k_embed, k_pass], dim=-1)
        else:
            q_embed = torch_npu.npu_rotary_mul(q, cos, sin).to(q.dtype)
            k_embed = torch_npu.npu_rotary_mul(k, cos, sin).to(k.dtype)

        return q_embed, k_embed

    return _apply_npu_rotary_emb


_apply_npu_rotary_emb = _make_apply_npu_rotary_emb()


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Fused RoPE via ``torch_npu.npu_rotary_mul`` with automatic Partial RoPE support."""
    unsqueeze_dim = _resolve_unsqueeze_dim(position_ids, unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return _apply_npu_rotary_emb(q, k, cos, sin)


def npu_apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Multimodal RoPE for Qwen2.5-VL with automatic Partial RoPE support."""
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    return _apply_npu_rotary_emb(q, k, cos, sin)


def npu_swiglu_forward(self, hidden_state):
    """Fused SwiGLU (Qwen-style)."""
    return self.down_proj(
        torch_npu.npu_swiglu(
            torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1),
            dim=-1,
        ))


def npu_sdpa_attention_forward(module,
                               query,
                               key,
                               value,
                               attention_mask,
                               dropout=0.0,
                               scaling=None,
                               is_causal=None,
                               **kwargs):
    r"""SDPA with NPU compatibility fixes."""
    from transformers.integrations.sdpa_attention import repeat_kv
    if hasattr(module, 'num_key_value_groups'):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, :key.shape[-2]]

    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()

    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    if causal_mask is not None and causal_mask.dtype != torch.bool:
        causal_mask = torch.logical_not(causal_mask.bool()).to(query.device)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    return attn_output.transpose(1, 2).contiguous(), None


# =============================================================================
# Section 2c: Flash Linear Attention (FLA) for Qwen3.5
# =============================================================================


def _patch_qwen3_5_fla(model=None) -> None:
    """Enable Flash Linear Attention (FLA) fast path for Qwen3.5 on NPU.

    Controlled by environment variable ``TWINKLE_NPU_FLA`` (default: True).
    """
    if not _is_env_enabled('TWINKLE_NPU_FLA', default=True):
        logger.info('[NPU] [FLA] Disabled by TWINKLE_NPU_FLA environment variable')
        return

    if not _is_torch_npu_available:
        logger.info('[NPU] [FLA] Skip: NPU not available')
        return

    # 1. Force FLA availability flag
    def _is_fla_available() -> bool:
        return True

    for utils_mod_name in ('transformers.utils', 'transformers.utils.import_utils'):
        try:
            utils_mod = importlib.import_module(utils_mod_name)
            setattr(utils_mod, 'is_flash_linear_attention_available', _is_fla_available)
            logger.info(
                '[NPU] [FLA] Patched %s.is_flash_linear_attention_available',
                utils_mod_name,
            )
        except Exception as exc:
            logger.debug('[NPU] [FLA] Failed to patch %s: %s', utils_mod_name, exc)

    # 2. Try MindSpeed Triton FLA backend
    mindspeed_fla = None
    try:
        from .chunk_gated_delta_rule import chunk_gated_delta_rule as _ms_fla
        mindspeed_fla = _ms_fla
        logger.info('[NPU] [FLA] MindSpeed Triton chunk_gated_delta_rule loaded')
    except ImportError as exc:
        logger.warning('[NPU] [FLA] MindSpeed not available: %s', exc)

    # 3. Patch Qwen3.5 modeling modules
    fla_target_modules = [
        'transformers.models.qwen3_5.modeling_qwen3_5',
        'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe',
    ]

    for module_name in fla_target_modules:
        module = import_optional_module(module_name)
        if module is None:
            logger.info('[NPU] [FLA] %s: module not found, skip', module_name)
            continue

        # Only enable FLA flags if we actually have a backend to serve it
        if mindspeed_fla is not None:
            setattr(module, 'is_flash_linear_attention_available', _is_fla_available)
            setattr(module, 'is_fast_path_available', True)

            # Disable CUDA-only fused op
            if hasattr(module, 'FusedRMSNormGated'):
                setattr(module, 'FusedRMSNormGated', None)
                logger.info('[NPU] [FLA] %s: disabled FusedRMSNormGated', module_name)

            # Replace chunk_gated_delta_rule with MindSpeed implementation
            setattr(module, 'chunk_gated_delta_rule', mindspeed_fla)
            logger.info(
                '[NPU] [FLA] Patched %s.chunk_gated_delta_rule -> MindSpeed',
                module_name,
            )
        else:
            logger.warning(
                '[NPU] [FLA] %s: MindSpeed unavailable, FLA flags NOT set',
                module_name,
            )

    # 4. Traverse instantiated model and replace per-layer chunk_gated_delta_rule
    if model is not None and mindspeed_fla is not None:
        # Resolve the underlying PyTorch model from TransformersModel wrapper
        model = getattr(model, 'model', getattr(model, 'module', model))
        if not hasattr(model, 'named_modules'):
            logger.warning('[NPU] [FLA] Model does not support named_modules, skipping instance patch')
            return
        patched_instances = 0
        for _name, _module in model.named_modules():
            if hasattr(_module, 'chunk_gated_delta_rule') and callable(getattr(_module, 'chunk_gated_delta_rule')):
                if _module.chunk_gated_delta_rule is mindspeed_fla:
                    continue

                _module.chunk_gated_delta_rule = mindspeed_fla
                # Mark as NPU-patched to prevent it from being overwritten by SP
                _module._twinkle_npu_patched = True
                patched_instances += 1
                logger.debug(
                    '[NPU] [FLA] Replaced %s(%s).chunk_gated_delta_rule -> MindSpeed',
                    _name,
                    type(_module).__name__,
                )

        if patched_instances > 0:
            logger.info(
                '[NPU] [FLA] Patched %d linear attention instance(s)',
                patched_instances,
            )
        else:
            logger.info('[NPU] [FLA] No linear attention instances found in model')


# =============================================================================
# Section 3: Patching Helpers
# =============================================================================


def _patch_sdpa_forward() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    AttentionInterface._global_mapping['sdpa'] = npu_sdpa_attention_forward
    ALL_ATTENTION_FUNCTIONS['sdpa'] = npu_sdpa_attention_forward
    logger.debug('[NPU] [SDPA] Patched global SDPA attention forward')


def _patch_rmsnorm(module, class_name: str) -> None:
    """Patch RMSNorm class with NPU-optimized implementation."""
    if 'Gated' in class_name:
        orig_cls = getattr(module, class_name)
        setattr(orig_cls, 'forward', npu_gated_rms_norm_forward)

        # Cache the FP32 env flag once at patch time to avoid per-forward overhead.
        orig_cls._twinkle_force_fp32 = os.environ.get('TWINKLE_NPU_GATED_RMSNorm_FP32',
                                                      '0').lower() in ('1', 'true', 'on', 'yes')
        if orig_cls._twinkle_force_fp32:
            logger.info(
                '[NPU] [RMSNorm] %s.%s forced to FP32 mode',
                module.__name__,
                class_name,
            )

        logger.info(
            '[NPU] [RMSNorm] Patched %s.%s.forward -> npu_gated_rms_norm_forward',
            module.__name__,
            class_name,
        )
    else:
        setattr(module, class_name, NpuRMSNorm)
        logger.info(
            '[NPU] [RMSNorm] Patched %s.%s -> NpuRMSNorm',
            module.__name__,
            class_name,
        )


def _patch_rope(module, func_name: str) -> None:
    setattr(module, func_name, npu_apply_rotary_pos_emb)
    logger.debug(
        '[NPU] [RoPE] Patched %s.%s -> npu_apply_rotary_pos_emb',
        module.__name__,
        func_name,
    )


def _patch_swiglu(module, class_name: str) -> None:
    setattr(getattr(module, class_name), 'forward', npu_swiglu_forward)
    logger.debug(
        '[NPU] [MLP] Patched %s.%s.forward -> npu_swiglu_forward',
        module.__name__,
        class_name,
    )


def _patch_moe_sparse_block(module, class_name: str) -> None:
    """Patch SparseMoeBlock forward with NPU-optimized implementation."""
    setattr(getattr(module, class_name), 'forward', npu_qwen3_5_moe_sparse_block_forward)
    logger.info(
        '[NPU] [MoE] Patched %s.%s.forward -> npu_qwen3_5_moe_sparse_block_forward',
        module.__name__,
        class_name,
    )


def _patch_moe_experts(module, class_name: str) -> None:
    """Patch packed Experts forward with NPU grouped matmul."""
    setattr(getattr(module, class_name), 'forward', npu_packed_moe_experts_forward)
    logger.debug(
        '[NPU] [MoE] Patched %s.%s.forward -> npu_packed_moe_experts_forward',
        module.__name__,
        class_name,
    )


# =============================================================================
# Section 4: Environment Control
# =============================================================================


def _is_env_enabled(var_name: str, default: bool = True) -> bool:
    """Check whether an environment variable is enabled.

    Supports: ``1``/``true``/``on``/``yes`` (force on),
    ``0``/``false``/``off``/``no`` (force off),
    unset (use ``default``).
    """
    env = os.environ.get(var_name, '').lower().strip()
    if not env:
        return default
    if env in ('1', 'true', 'on', 'yes'):
        return True
    if env in ('0', 'false', 'off', 'no'):
        logger.info('[NPU] %s=%s: disabled.', var_name, env)
        return False
    return default


# =============================================================================
# Section 5: Unified Patching Logic (Fused Ops)
# =============================================================================


def _apply_all_fused_ops(model=None) -> None:
    """Apply fused ops to supported model families."""
    logger.info('[NPU] === _apply_all_fused_ops ENTERED ===')
    if not _is_torch_npu_available:
        return

    if not _is_env_enabled('TWINKLE_NPU_FUSED_OPS', default=True):
        return

    target_archs = set()
    if model is not None:
        config = getattr(model, 'hf_config', getattr(model, 'config', None))
        archs = getattr(config, 'architectures', None) if config else None
        if archs:
            target_archs = set(archs)
            logger.debug('[NPU] Detected architectures for fused ops: %s', archs)

    logger.info('[NPU] Auto-applying fused ops to supported model families')

    _patch_sdpa_forward()

    model_families = [
        ('transformers.models.qwen3.modeling_qwen3', 'Qwen3', 'Qwen3MLP', 'Qwen3ForCausalLM'),
        ('transformers.models.qwen3_moe.modeling_qwen3_moe', 'Qwen3Moe', 'Qwen3MoeMLP', 'Qwen3MoeForCausalLM'),
        (
            'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl',
            'Qwen2_5_VL',
            'Qwen2MLP',
            'Qwen2_5_VLForConditionalGeneration',
        ),
        (
            'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe',
            'Qwen3_5Moe',
            'Qwen3_5MoeMLP',
            'Qwen3MoeForCausalLM',
        ),
    ]

    modeling_qwen3_5 = import_optional_module('transformers.models.qwen3_5.modeling_qwen3_5')
    if modeling_qwen3_5 is not None:
        model_families.append((
            'transformers.models.qwen3_5.modeling_qwen3_5',
            'Qwen3_5',
            'Qwen3_5MLP',
            'Qwen3_5ForCausalLM',
        ))

    modeling_qwen3_5_moe = import_optional_module('transformers.models.qwen3_5_moe.modeling_qwen3_5_moe')
    if modeling_qwen3_5_moe is not None:
        model_families.append((
            'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe',
            'Qwen3_5Moe',
            'Qwen3_5MoeMLP',
            'Qwen3_5MoeForCausalLM',
        ))

    patched_count = 0
    for module_name, prefix, mlp_name, trigger_arch in model_families:
        try:
            module = importlib.import_module(module_name)

            # RMSNorm
            rmsnorm_cls = f'{prefix}RMSNorm'
            if hasattr(module, rmsnorm_cls):
                _patch_rmsnorm(module, rmsnorm_cls)
                patched_count += 1

            # RoPE
            if hasattr(module, 'apply_rotary_pos_emb'):
                _patch_rope(module, 'apply_rotary_pos_emb')
                patched_count += 1

            # SwiGLU / MLP
            if hasattr(module, mlp_name):
                _patch_swiglu(module, mlp_name)
                patched_count += 1

            experts_cls = f'{prefix}Experts'
            if hasattr(module, experts_cls):
                _patch_moe_experts(module, experts_cls)
                patched_count += 1

            sparse_cls = f'{prefix}SparseMoeBlock'
            if hasattr(module, sparse_cls):
                _patch_moe_sparse_block(module, sparse_cls)
                patched_count += 1

            if prefix == 'Qwen2_5_VL':
                if hasattr(module, 'Qwen2_5_VLMLP'):
                    _patch_swiglu(module, 'Qwen2_5_VLMLP')
                    patched_count += 1
                setattr(module, 'apply_multimodal_rotary_pos_emb', npu_apply_multimodal_rotary_pos_emb)
                logger.debug('[NPU] Patched Qwen2_5_VL multimodal RoPE')

            if prefix == 'Qwen3_5':
                gated_rmsnorm_cls = f'{prefix}GatedRMSNorm'
                if hasattr(module, gated_rmsnorm_cls):
                    _patch_rmsnorm(module, gated_rmsnorm_cls)
                    patched_count += 1
                if hasattr(module, 'Qwen3_5VisionMLP'):
                    _patch_swiglu(module, 'Qwen3_5VisionMLP')
                    patched_count += 1
                if hasattr(module, 'Qwen3_5VisionRMSNorm'):
                    _patch_rmsnorm(module, 'Qwen3_5VisionRMSNorm')
                    patched_count += 1

            if prefix == 'Qwen3_5Moe':
                if hasattr(module, 'Qwen3_5MoeGatedRMSNorm'):
                    _patch_rmsnorm(module, 'Qwen3_5MoeGatedRMSNorm')
                    patched_count += 1

            logger.debug('[NPU] Patched %s fused ops', prefix)
        except ImportError:
            pass

    if not target_archs:
        patched_count += _discover_and_patch_unknown_models()

    _patch_qwen3_5_fla(model)

    logger.info('[NPU] Auto-patched %d components', patched_count)


# =============================================================================
# Section 5b: Dynamic model discovery (no hard-coding)
# =============================================================================


def _discover_and_patch_unknown_models() -> int:
    """Dynamically discover and patch additional transformers model families."""
    patched = 0
    already_patched_modules = {
        'transformers.models.qwen3.modeling_qwen3',
        'transformers.models.qwen3_moe.modeling_qwen3_moe',
        'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl',
        'transformers.models.qwen3_5.modeling_qwen3_5',
        'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe',
    }

    try:
        import transformers.models as models_pkg
    except ImportError:
        return 0

    candidate_modules = []
    for model_name in dir(models_pkg):
        if model_name.startswith('_'):
            continue
        modeling_path = f'transformers.models.{model_name}.modeling_{model_name}'
        if modeling_path not in already_patched_modules:
            candidate_modules.append(modeling_path)

    for module_name in candidate_modules:
        module = import_optional_module(module_name)
        if module is None:
            continue

        has_rmsnorm = any('RMSNorm' in attr_name and isinstance(getattr(module, attr_name, None), type)
                          for attr_name in dir(module))
        has_rope = hasattr(module, 'apply_rotary_pos_emb')
        has_mlp = any(
            attr_name.endswith('MLP') and isinstance(getattr(module, attr_name, None), type)
            for attr_name in dir(module))

        if not (has_rmsnorm or has_rope or has_mlp):
            continue

        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            obj = getattr(module, attr_name, None)
            if not isinstance(obj, type):
                continue

            if 'RMSNorm' in attr_name and issubclass(obj, nn.Module):
                try:
                    _patch_rmsnorm(module, attr_name)
                    patched += 1
                except Exception as exc:
                    logger.debug('[NPU] Failed to patch %s.%s: %s', module_name, attr_name, exc)

            if attr_name.endswith('MLP') and hasattr(obj, 'forward'):
                try:
                    _patch_swiglu(module, attr_name)
                    patched += 1
                except Exception as exc:
                    logger.debug('[NPU] Failed to patch %s.%s: %s', module_name, attr_name, exc)

            if attr_name.endswith('Experts') and hasattr(obj, 'forward'):
                try:
                    _patch_moe_experts(module, attr_name)
                    patched += 1
                except Exception as exc:
                    logger.debug('[NPU] Failed to patch %s.%s: %s', module_name, attr_name, exc)

            if attr_name.endswith('SparseMoeBlock') and hasattr(obj, 'forward'):
                try:
                    _patch_moe_sparse_block(module, attr_name)
                    patched += 1
                except Exception as exc:
                    logger.debug('[NPU] Failed to patch %s.%s: %s', module_name, attr_name, exc)

        if has_rope:
            try:
                _patch_rope(module, 'apply_rotary_pos_emb')
                patched += 1
            except Exception as exc:
                logger.debug('[NPU] Failed to patch %s.apply_rotary_pos_emb: %s', module_name, exc)

        if patched > 0:
            logger.debug('[NPU] Dynamically patched %s', module_name)

    return patched


# =============================================================================
# Section 6: Public API
# =============================================================================


def apply_npu_patch(model=None) -> None:
    """Apply all NPU patches.

    Ascend NPU optimizations applied:
      - MoE grouped_matmul (GMM)
      - RMSNorm fused kernel
      - RoPE fused kernel
      - SwiGLU fused kernel
      - SDPA Attention compatibility fixes
      - Flash Linear Attention (FLA) for Qwen3.5

    When ``model`` is **not** provided, the GMM patch is **skipped** by default
    (EP cannot be detected without a model instance).

    When ``model`` is provided, the GMM patch is evaluated with EP detection:
      - EP enabled → apply GMM patch (efficient on small sharded weights).
      - EP not enabled → skip GMM patch (avoid ~8x contiguous-copy overhead).

    Environment variables:
      - ``TWINKLE_NPU_PATCH``: overall switch (``1``/``0``)
      - ``TWINKLE_NPU_FUSED_OPS``: fused ops switch (``1``/``0``)
      - ``TWINKLE_NPU_GMM_PATCH``: MoE GMM switch (``1``/``0``/unset).
        When unset: skip the patch by default.
        When ``1``: EP-aware — patch is applied **only if EP is enabled**;
        without EP the native grouped_mm or per-expert fallback is used
        (avoiding ~8x overhead from contiguous copies).
        When ``0``: disable the patch regardless.
      - ``TWINKLE_NPU_FLA``: FLA switch (``1``/``0``)
      - ``TWINKLE_NPU_GATED_RMSNorm_FP32``: force FP32 in Gated RMSNorm (``1``/``0``)

    Args:
        model: Optional model instance. If not provided, GMM patch is skipped.
            If provided, GMM patch is evaluated with EP detection on the model.
    """
    global _NPU_PATCH_APPLIED

    if not _is_env_enabled('TWINKLE_NPU_PATCH', default=True):
        return

    if _NPU_PATCH_APPLIED:
        logger.debug('[NPU] Patches already applied, skipping.')
        return

    try:
        import torch_npu
    except ImportError:
        logger.warning('torch_npu not available. Skipping NPU patches.')
        return

    _apply_hf_moe_grouped_mm_patch(model)

    _apply_all_fused_ops(model)

    _NPU_PATCH_APPLIED = True
    logger.info('[NPU] All patches applied successfully')


def register_npu_fused_function_kernels() -> None:
    """Register NPU fused ops as Twinkle function kernels (optional)."""
    if not _is_torch_npu_available:
        return

    from .function import register_function_kernel

    register_function_kernel(
        func_name='apply_rotary_pos_emb',
        target_module='transformers.modeling_rope_utils',
        func_impl=npu_apply_rotary_pos_emb,
        device='npu',
        mode='train',
    )
    register_function_kernel(
        func_name='sdpa_attention_forward',
        target_module='transformers.integrations.sdpa_attention',
        func_impl=npu_sdpa_attention_forward,
        device='npu',
        mode='train',
    )
    logger.info('[NPU] Registered fused function kernels for training')
