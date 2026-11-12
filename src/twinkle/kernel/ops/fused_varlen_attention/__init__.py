# Copyright (c) ModelScope Contributors. All rights reserved.
"""变长（packed）因果注意力：单算子入口加上 ``transformers`` 注意力接口的适配器。

本算子补齐的是 NPU 侧的融合点：padding-free 训练需要「一条拼接序列 + 累积分段长度 ->
分段因果注意力」，而仓库既有的 padding-free 路径要求 flash-attention 实现。内核底座是
CANN 的 FlashAttention，本包负责分段边界、掩码、autograd 与接口契约。

``torch`` 后端是精确参考实现的 HF 适配器（逐文档 SDPA），用于 CPU/GPU 正确性对照与无 NPU 环境。
两个后端共用 ``adapter`` 中的掩码合成与边界校验，接口语义保持一致。
"""
from __future__ import annotations

from twinkle import get_logger
from ...registry import KernelImpl, is_npu_available, lazy_import, register_op
from .functional import fused_varlen_attention
from .reference import fused_varlen_attention_reference, torch_varlen_attention_forward

logger = get_logger()

#: 上游模型只在 ``flash_attention_2`` 这个实现名下透传 ``cu_seq_lens_q``，
#: 因此适配器挂在该分发名上；此处不做任何可用性伪造，由调用方决定何时安装。
ATTENTION_NAME = 'flash_attention_2'


def install_varlen_attention(model, target, impl) -> None:
    """把昇腾变长实现挂到 ``flash_attention_2`` 分发名上（调用方负责恢复）。"""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    except ImportError:
        logger.warning('[varlen_attention] transformers 注意力接口不可用；跳过安装')
        return
    try:
        AttentionInterface._global_mapping[ATTENTION_NAME] = impl
    except AttributeError:
        logger.warning('[varlen_attention] AttentionInterface._global_mapping 不可用；仅替换全局映射')
    ALL_ATTENTION_FUNCTIONS[ATTENTION_NAME] = impl


register_op(
    'fused_varlen_attention',
    implementations={
        'npu':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.fused_varlen_attention.npu:npu_varlen_attention_forward'),
            available=is_npu_available,
        ),
        'torch':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.fused_varlen_attention.reference:torch_varlen_attention_forward'),
            available=lambda: (True, None),
        ),
    },
    installer=install_varlen_attention,
)

__all__ = [
    'fused_varlen_attention', 'fused_varlen_attention_reference', 'install_varlen_attention',
    'torch_varlen_attention_forward'
]
