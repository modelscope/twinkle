"""NPU monkey patches for Ascend hardware acceleration.

Unified entry point::

    >>> from twinkle.kernel.monkey_patch_npu import apply_npu_patch
    >>> if Torch.is_npu_available():
    ...     apply_npu_patch()  # Enables all patches unconditionally
"""

import importlib
import torch
from torch import nn
from transformers.utils import is_torch_npu_available

from twinkle import get_logger

logger = get_logger(__name__)

_is_torch_npu_available = is_torch_npu_available()
_NPU_PATCH_APPLIED = False

if _is_torch_npu_available:
    import torch_npu

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


def apply_hf_moe_grouped_mm_patch() -> None:
    r"""Patch HuggingFace MoE integration to use NPU grouped matmul."""
    import transformers.integrations.moe as hf_moe
    hf_moe._grouped_mm = _grouped_mm_npu
    logger.info('[PATCH] transformers.integrations.moe._grouped_mm -> _grouped_mm_npu')


# =============================================================================
# Section 2: Fused Operators
# =============================================================================


class NpuRMSNorm(nn.Module):
    r"""Fused RMSNorm via ``torch_npu.npu_rms_norm``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""Fused RoPE via ``torch_npu.npu_rotary_mul``."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return torch_npu.npu_rotary_mul(q, cos, sin), torch_npu.npu_rotary_mul(k, cos, sin)


def npu_apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    r"""Multimodal RoPE for Qwen2.5-VL."""
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    return torch_npu.npu_rotary_mul(q, cos, sin), torch_npu.npu_rotary_mul(k, cos, sin)


def npu_swiglu_forward(self, hidden_state):
    r"""Fused SwiGLU (Qwen-style)."""
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1))


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
# Section 3: Patching Helpers
# =============================================================================


def _patch_sdpa_forward() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    AttentionInterface._global_mapping['sdpa'] = npu_sdpa_attention_forward
    ALL_ATTENTION_FUNCTIONS['sdpa'] = npu_sdpa_attention_forward
    logger.debug('[NPU] Patched SDPA attention forward')


def _patch_rmsnorm(module, class_name: str) -> None:
    setattr(module, class_name, NpuRMSNorm)
    logger.debug(f"[NPU] Patched {module.__name__}.{class_name} -> NpuRMSNorm")


def _patch_rope(module, func_name: str) -> None:
    setattr(module, func_name, npu_apply_rotary_pos_emb)
    logger.debug(f"[NPU] Patched {module.__name__}.{func_name} -> npu_apply_rotary_pos_emb")


def _patch_swiglu(module, class_name: str) -> None:
    setattr(getattr(module, class_name), 'forward', npu_swiglu_forward)
    logger.debug(f"[NPU] Patched {module.__name__}.{class_name}.forward -> npu_swiglu_forward")


# =============================================================================
# Section 4: Unified Patching Logic
# =============================================================================


def _apply_all_fused_ops() -> None:
    r"""Apply fused ops to supported model families unconditionally."""
    if not _is_torch_npu_available:
        return

    logger.info('[NPU] Auto-applying fused ops to supported model families')

    # Patch global SDPA first
    _patch_sdpa_forward()

    # Supported model families: (module_path, class_prefix, mlp_class_name)
    model_families = [
        ('transformers.models.qwen3.modeling_qwen3', 'Qwen3', 'Qwen3MLP'),
        ('transformers.models.qwen3_moe.modeling_qwen3_moe', 'Qwen3Moe', 'Qwen3MoeMLP'),
        ('transformers.models.qwen2_5_vl.modeling_qwen2_5_vl', 'Qwen2_5_VL', 'Qwen2MLP'),
    ]

    patched_count = 0
    for module_name, prefix, mlp_name in model_families:
        try:
            module = importlib.import_module(module_name)

            # RMSNorm
            rmsnorm_cls = f"{prefix}RMSNorm"
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

            # Qwen2.5-VL special cases
            if prefix == 'Qwen2_5_VL':
                if hasattr(module, 'Qwen2_5_VLMLP'):
                    _patch_swiglu(module, 'Qwen2_5_VLMLP')
                    patched_count += 1
                setattr(module, 'apply_multimodal_rotary_pos_emb', npu_apply_multimodal_rotary_pos_emb)
                logger.debug('[NPU] Patched Qwen2_5_VL multimodal RoPE')

            logger.debug(f"[NPU] Patched {prefix} fused ops")
        except ImportError:
            pass  # Model family not installed, skip silently

    logger.info(f"[NPU] Auto-patched {patched_count} components")


# =============================================================================
# Section 5: Public API
# =============================================================================


def apply_npu_patch() -> None:
    r"""Apply all NPU patches unconditionally.

    All Ascend NPU optimizations are applied when running on NPU:
      - MoE grouped_matmul (GMM)
      - RMSNorm fused kernel
      - RoPE fused kernel
      - SwiGLU fused kernel
      - SDPA Attention compatibility fixes

    Safe to call multiple times — patches are only applied once.

    Example::

        >>> # Unified entry — all patches applied
        >>> if Torch.is_npu_available():
        ...     apply_npu_patch()
    """
    global _NPU_PATCH_APPLIED

    if _NPU_PATCH_APPLIED:
        logger.debug('[NPU] Patches already applied, skipping.')
        return

    try:
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
    except ImportError:
        logger.warning('torch_npu not available. Skipping NPU patches.')
        return

    # 1. MoE GMM (always)
    apply_hf_moe_grouped_mm_patch()

    # 2. Fused operators (always, unconditional)
    _apply_all_fused_ops()

    _NPU_PATCH_APPLIED = True
    logger.info('[NPU] All patches applied successfully')


def register_npu_fused_function_kernels() -> None:
    r"""Register NPU fused ops as Twinkle function kernels (optional).

    Integrates with Twinkle's ``kernelize_model()`` so that RoPE and SDPA
    are automatically replaced when running on Ascend devices.
    """
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
