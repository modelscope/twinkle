"""Load a hash-verified public regression sample; no download or customer data."""
import hashlib
import json
from pathlib import Path
import numpy as np


def load_speech_fixture(directory):
    import soundfile as sf
    from scipy.signal import resample_poly
    directory = Path(directory)
    metadata = json.loads((directory / 'sample.json').read_text(encoding='utf-8'))
    path = directory / metadata['audio']
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Speech fixture SHA256 mismatch')
    wave, sr = sf.read(path, dtype='float32', always_2d=False)
    if sr != 16000 or wave.ndim != 1 or len(wave) == 0 or not np.isfinite(wave).all():
        raise ValueError('Expected finite mono 16 kHz LibriSpeech fixture')
    if not metadata['text'].strip():
        raise ValueError('Speech fixture transcript is empty')
    # Exact rational rate conversion; preserve the full utterance and transcript.
    wave = resample_poly(wave, 3, 2).astype(np.float32)
    return wave, metadata['text'], metadata
