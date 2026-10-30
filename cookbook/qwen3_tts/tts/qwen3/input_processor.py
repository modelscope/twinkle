"""Qwen3-TTS 单卡 padded batch processor。

编码样本按时间 pad 并 stack；已 collate 的 dict 直接校验并转换设备/类型。
参考 mel 保留 [B,F,128]，不同参考长度明确拒绝；不走通用 LM 字段过滤管线。
"""

from typing import Dict, List

import numpy as np
import torch

from twinkle.data_format import InputFeature
from twinkle import Platform, remote_function
from twinkle.processor.base import InputProcessor
from twinkle.utils.torch_utils import pad_and_stack_tensors
from .data import normalize_ref_mels

# 序列字段：强制 stack（concat=False）→ [B, maxT, *]
_SEQ_STACK_FIELDS = (
    'input_ids', 'codec_ids', 'codec_0_labels',
    'text_embedding_mask', 'codec_embedding_mask', 'codec_mask',
    'attention_mask', 'position_ids',
)
# mask 字段 stack 后补 channel 维 → [B, maxT, 1]（对齐官方 collate_fn + forward 广播）
_MASK_CHANNEL_FIELDS = ('text_embedding_mask', 'codec_embedding_mask')

# to_transformers_dict 白名单追加（基类 `_keys` 之外的自定义字段）
_EXTRA_WHITELIST = (
    'codec_ids', 'codec_0_labels',
    'text_embedding_mask', 'codec_embedding_mask', 'codec_mask',
    'ref_mels',
)


def _to_tensor(v) -> torch.Tensor:
    """numpy/list → torch；bool → uint8（规避 F.pad 对 bool 的版本兼容问题；值仍是 0/1）。"""
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)
    elif isinstance(v, list):
        v = torch.tensor(v)
    if v.dtype == torch.bool:
        v = v.to(torch.uint8)
    return v


class Qwen3TTSInputProcessor(InputProcessor):
    """Qwen3-TTS 批组织：2D 强制 stack + ref_mels stack + 自定义字段白名单。"""

    padding_map = {
        **InputProcessor.padding_map,
        'codec_ids': 0,
        'codec_0_labels': -100,
        'text_embedding_mask': False,
        'codec_embedding_mask': False,
        'codec_mask': False,
        'ref_mels': 0.0,
    }

    @remote_function()
    def __call__(self, inputs, **kwargs):
        """Single-card, padded TTS: encoded samples or an already collated dict.

        Return one tensor dict. Do not run the generic LM preparation pipeline:
        it iterates dict keys, changes ranks and filters out TTS fields.
        """
        if self.padding_free or kwargs.get('enable_sp', False):
            raise ValueError('Qwen3 TTS supports padded single-card batches only')
        if isinstance(inputs, dict) and getattr(inputs['input_ids'], 'ndim', None) == 3:
            batch = self.to_transformers_dict([inputs])[0]
        else:
            samples = [inputs] if isinstance(inputs, dict) else inputs
            if not samples:
                raise ValueError('Empty TTS batch')
            batch = self.to_transformers_dict(self.collate_fn(samples))[0]
        required = set(_EXTRA_WHITELIST) | {'input_ids', 'attention_mask'}
        missing = required - batch.keys()
        if missing:
            raise ValueError(f'Missing TTS fields: {sorted(missing)}')
        batch = {key: torch.as_tensor(value) for key, value in batch.items()}
        b, t, channels = batch['input_ids'].shape
        if channels != 2 or b == 0 or t == 0:
            raise ValueError('input_ids must have shape [B,T,2]')
        shapes = {'codec_ids': (b, t, 16), 'codec_0_labels': (b, t),
                  'attention_mask': (b, t), 'codec_mask': (b, t),
                  'text_embedding_mask': (b, t, 1), 'codec_embedding_mask': (b, t, 1)}
        for key, shape in shapes.items():
            if tuple(batch[key].shape) != shape:
                raise ValueError(f'{key} must have shape {shape}')
        mel = batch['ref_mels']
        if mel.ndim != 3 or mel.shape[0] != b or mel.shape[1] == 0 or mel.shape[2] != 128:
            raise ValueError('ref_mels must have shape [B,F,128], F > 0')
        if kwargs.get('micro_batch_size') not in (None, b):
            raise ValueError('Split TTS micro-batches in the DataLoader')
        model = kwargs.get('model')
        device = getattr(model, 'device', None)
        if device is None:
            device = Platform.get_local_device()
        for key, value in batch.items():
            dtype = (torch.float32 if key == 'ref_mels' else torch.bool
                     if key in (*_MASK_CHANNEL_FIELDS, 'codec_mask') else torch.long)
            batch[key] = value.to(device=device, dtype=dtype)
        return batch

    def collate_fn(self, inputs, micro_batch_size=None, variable_seq_lengths=False, **kwargs):
        seq_out: Dict[str, torch.Tensor] = {}

        for key in _SEQ_STACK_FIELDS:
            vals: List[torch.Tensor] = []
            for inp in inputs:
                v = inp.get(key)
                if v is None:
                    continue
                v = _to_tensor(v)
                # 撤销 prepare_inputs 对 1D 字段的 unsqueeze（[1,T]→[T]）；
                # 2D 字段（[T,2]/[T,16]）shape[0]=T≠1 不受影响。
                if key not in ('input_ids', 'codec_ids') and v.dim() == 2 and v.shape[0] == 1:
                    v = v.squeeze(0)
                vals.append(v)
            if not vals:
                continue
            pad = int(self.padding_map.get(key, 0))
            if len(vals) == 1:
                stacked = vals[0].unsqueeze(0)                # 保证有 batch 维
            else:
                stacked = pad_and_stack_tensors(vals, pad_value=pad, concat=False)
            if key in _MASK_CHANNEL_FIELDS:
                stacked = stacked.unsqueeze(-1)                # [B, maxT, 1]
            seq_out[key] = stacked

        # 保留每条参考的 batch 归属；无长度掩码时不自动补零。
        ref_mels: List[torch.Tensor] = []
        for inp in inputs:
            v = inp.get('ref_mels')
            if v is None:
                continue
            v = _to_tensor(v)
            ref_mels.append(v)
        if len(ref_mels) != len(inputs):
            raise ValueError('Each sample requires one ref_mel')
        seq_out['ref_mels'] = torch.stack(normalize_ref_mels(ref_mels), dim=0)

        # length：标量元数据，取 batch 内最大
        lengths = [int(inp.get('length')) for inp in inputs if inp.get('length') is not None]
        if lengths:
            seq_out['length'] = max(lengths)

        return [InputFeature(**seq_out)]

    @staticmethod
    def to_transformers_dict(inputs, **kwargs):
        results = []
        # 基类白名单 + 本模型自定义字段（基类 _keys 见 processor/base.py:663-680）
        _keys = {
            'input_ids', 'inputs_embeds', 'attention_mask', 'position_ids', 'labels',
            'completion_mask', 'cu_seq_lens_q', 'cu_seq_lens_k', 'cu_seqlens_q',
            'cu_seqlens_kv', 'max_length_q', 'max_length_k', 'packed_seq_params',
            'routed_experts', 'mm_token_type_ids', 'second_per_grid_ts',
        } | set(InputProcessor.VLM_CONCAT_FIELDS) | set(_EXTRA_WHITELIST)

        for _input in inputs:
            output = {}
            for key in list(_input.keys()):
                if key not in _keys:
                    continue
                value = _input[key]
                if isinstance(value, torch.Tensor) or not isinstance(value, (list, np.ndarray)):
                    output[key] = value
                else:
                    output[key] = np.array(value)
            results.append(InputFeature(**output))
        return results
