"""Official 24 kHz mel extraction and 16-codebook audio encoding."""

import numpy as np

MEL_DIM = 128
MEL_HOP = 256
MEL_WIN = 1024
MEL_FMAX = 12000.0
TARGET_SR = 24000

def extract_mel(wav_24k: np.ndarray) -> np.ndarray:
    """使用官方 mel 算法，返回单条 [F,128]，不使用替代 fbank。"""
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    wav = np.asarray(wav_24k, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError('Expected mono waveform [samples]')
    with torch.no_grad():
        mel = mel_spectrogram(
            torch.from_numpy(wav).unsqueeze(0), n_fft=MEL_WIN,
            num_mels=MEL_DIM, sampling_rate=TARGET_SR, hop_size=MEL_HOP,
            win_size=MEL_WIN, fmin=0, fmax=MEL_FMAX,
        ).transpose(1, 2)
    return mel[0].cpu().numpy()

def encode_audio_codes(wav_24k: np.ndarray, tokenizer) -> np.ndarray:
    """单条输入必须返回一个 [T,16]；兼容列表、二维张量和单批张量。"""
    result = tokenizer.encode(wav_24k, sr=TARGET_SR)   # wav_24k 已升采样到 24kHz
    codes = result.audio_codes
    if isinstance(codes, (list, tuple)):
        if len(codes) != 1:
            raise ValueError('Single waveform must return exactly one audio_codes entry')
        codes = codes[0]
    if hasattr(codes, 'cpu'):
        codes = codes.cpu().numpy()
    codes = np.asarray(codes)
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]
    if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] != 16:
        raise ValueError(f'Expected nonempty audio_codes [T,16], got {codes.shape}')
    return codes.astype(np.int64)
