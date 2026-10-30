"""Qwen3-TTS 双通道样本构建与 batch 组织。特殊 token 应从目标模型 config 读取。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

NUM_CODE_GROUPS = 16  # Qwen3-TTS 12Hz codec.


@dataclass
class Qwen3TTSDataConfig:
    """Token layout for the 12Hz model.

    The loader supplies actual IDs from the local model configuration; defaults
    document the 1.7B Base layout and should not be assumed for other variants.
    """

    # 文本通道（Qwen3 文本词表段，151xxx，config 顶层 tts_*_token_id）
    tts_pad_token_id: int = 151671
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    # codec 通道（codec 词表段，2xxx，config.talker_config.codec_*_id）
    codec_pad_id: int = 2148
    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    num_code_groups: int = NUM_CODE_GROUPS


def build_assistant_text(text: str) -> str:
    """官方 `_build_assistant_text`：把目标文本包成 assistant 轮（ChatML）。

    真实 text_ids 由 Qwen3 文本 tokenizer 对该串编码、再 `[:, :-5]` 得到。
    本模块只负责布局，text_ids 由上游（Preprocessor/Template）传入。
    """
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_qwen3_sample(
    text_ids: np.ndarray,
    audio_codes: np.ndarray,
    cfg: Qwen3TTSDataConfig,
) -> Dict[str, np.ndarray]:
    """把单条 (text_ids, audio_codes) 组成双通道 InputFeature（未 pad）。

    Args:
        text_ids: 1D int array [L]，L>=4（官方 `[:, :-5]` 后的文本 token，含 assistant 标签）。
        audio_codes: 2D int array [C, num_code_groups]，目标音频多码本。
        cfg: 特殊 token id。

    Returns:
        dict，每项是长度 T=8+L+C 的 numpy 数组（input_ids/codec_ids 是 2D）。
    """
    text_ids = np.asarray(text_ids).reshape(-1)
    audio_codes = np.asarray(audio_codes)
    assert audio_codes.ndim == 2 and audio_codes.shape[1] == cfg.num_code_groups, (
        f"audio_codes shape 应为 [C, {cfg.num_code_groups}]，实得 {audio_codes.shape}"
    )
    L = int(text_ids.shape[0])
    C = int(audio_codes.shape[0])
    assert L >= 4, f"text_ids_len L={L} 过短（需 >=4 以满足 [3:] 非空）"

    T = 8 + L + C
    audio_codec_0 = audio_codes[:, 0]  # [C]
    ncb = cfg.num_code_groups

    input_ids = np.zeros((T, 2), dtype=np.int64)
    codec_ids = np.zeros((T, ncb), dtype=np.int64)
    codec_0_labels = np.full((T,), -100, dtype=np.int64)
    text_embedding_mask = np.zeros((T,), dtype=bool)
    codec_embedding_mask = np.zeros((T,), dtype=bool)
    codec_mask = np.zeros((T,), dtype=bool)
    attention_mask = np.zeros((T,), dtype=np.int64)

    # ---- 文本通道 input_ids[:,0] ----
    input_ids[0:3, 0] = text_ids[0:3]
    input_ids[3:7, 0] = cfg.tts_pad_token_id
    input_ids[7, 0] = cfg.tts_bos_token_id
    input_ids[8 : 8 + L - 3, 0] = text_ids[3:]          # L-3 个
    input_ids[8 + L - 3, 0] = cfg.tts_eos_token_id
    input_ids[8 + L - 2 : 8 + L + C, 0] = cfg.tts_pad_token_id   # [8+L-2 : T]

    # ---- codec 通道 input_ids[:,1] ----
    input_ids[3:8, 1] = [
        cfg.codec_nothink_id,
        cfg.codec_think_bos_id,
        cfg.codec_think_eos_id,
        0,                       # speaker emb 槽（id=0 占位，训练步注入 speaker_emb）
        cfg.codec_pad_id,
    ]
    input_ids[8 : 8 + L - 3, 1] = cfg.codec_pad_id
    input_ids[8 + L - 3, 1] = cfg.codec_pad_id
    input_ids[8 + L - 2, 1] = cfg.codec_bos_id
    input_ids[8 + L - 1 : 8 + L - 1 + C, 1] = audio_codec_0          # [8+L-1 : 8+L-1+C]
    input_ids[8 + L - 1 + C, 1] = cfg.codec_eos_token_id            # pos T-1

    # ---- codec_0_labels（码本0 label，-100 mask） ----
    codec_0_labels[8 + L - 1 : 8 + L - 1 + C] = audio_codec_0
    codec_0_labels[8 + L - 1 + C] = cfg.codec_eos_token_id         # pos T-1

    # ---- codec_ids（全码本，喂 Code predictor） ----
    codec_ids[8 + L - 1 : 8 + L - 1 + C, :] = audio_codes

    # ---- masks ----
    text_embedding_mask[: 8 + L + C] = True                          # [:T]
    codec_embedding_mask[3 : 8 + L + C] = True                       # [3:T]
    codec_embedding_mask[6] = False                                   # speaker emb 槽
    codec_mask[8 + L - 1 : 8 + L - 1 + C] = True                      # [8+L-1 : 8+L+C-1) = C 个目标帧
    attention_mask[: 8 + L + C] = True                                # [:T]

    return {
        'input_ids': input_ids,
        'codec_ids': codec_ids,
        'codec_0_labels': codec_0_labels,
        'text_embedding_mask': text_embedding_mask,
        'codec_embedding_mask': codec_embedding_mask,
        'codec_mask': codec_mask,
        'attention_mask': attention_mask,
        'length': np.int64(T),
    }


def collate_qwen3_batch(
    samples: List[Dict[str, np.ndarray]],
    cfg: Qwen3TTSDataConfig,
    ref_mels: Optional[List[np.ndarray]] = None,
    max_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """把多个 build_qwen3_sample 输出 pad 成一个 batch（对齐官方 collate_fn 的 padding）。

    padding 规则（与官方一致）：
      input_ids/codec_ids -> 0；masks -> False；attention_mask -> 0；codec_0_labels -> -100。
    ref_mels 单条为 [F,128] 或 [1,F,128]，同长度 stack 为 [B,F,128]。
    不同参考长度拒绝合批；使用相同参考音频或按参考长度分批，避免无掩码补零影响池化。
    """
    b = len(samples)
    if max_length is None:
        max_length = int(max(int(s['length']) for s in samples))
    t = max_length
    ncb = cfg.num_code_groups

    input_ids = np.zeros((b, t, 2), dtype=np.int64)
    codec_ids = np.zeros((b, t, ncb), dtype=np.int64)
    codec_0_labels = np.full((b, t), -100, dtype=np.int64)
    text_embedding_mask = np.zeros((b, t), dtype=bool)
    codec_embedding_mask = np.zeros((b, t), dtype=bool)
    codec_mask = np.zeros((b, t), dtype=bool)
    attention_mask = np.zeros((b, t), dtype=np.int64)

    for i, s in enumerate(samples):
        Ti = int(s['length'])
        input_ids[i, :Ti, :] = s['input_ids']
        codec_ids[i, :Ti, :] = s['codec_ids']
        codec_0_labels[i, :Ti] = s['codec_0_labels']
        text_embedding_mask[i, :Ti] = s['text_embedding_mask']
        codec_embedding_mask[i, :Ti] = s['codec_embedding_mask']
        codec_mask[i, :Ti] = s['codec_mask']
        attention_mask[i, :Ti] = s['attention_mask']

    batch = {
        'input_ids': input_ids,
        'codec_ids': codec_ids,
        'codec_0_labels': codec_0_labels,
        'text_embedding_mask': text_embedding_mask[:, :, None],   # [b,t,1] 对齐官方
        'codec_embedding_mask': codec_embedding_mask[:, :, None],  # [b,t,1]
        'codec_mask': codec_mask,
        'attention_mask': attention_mask,
    }
    if ref_mels is not None:
        if len(ref_mels) != b:
            raise ValueError('Each sample requires one ref_mel')
        batch['ref_mels'] = np.stack(normalize_ref_mels(ref_mels), axis=0)
    return batch


def normalize_ref_mels(ref_mels):
    """Validate per-sample mel shapes without padding or merging speakers."""
    normalized = []
    for mel in ref_mels:
        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel[0]
        if mel.ndim != 2 or mel.shape[0] == 0 or mel.shape[1] != 128:
            raise ValueError('Each ref_mel must have shape [F,128] or [1,F,128], F > 0')
        normalized.append(mel)
    if not normalized or any(m.shape != normalized[0].shape for m in normalized):
        raise ValueError('ref_mels must have equal lengths; use a shared reference or length buckets')
    return normalized
