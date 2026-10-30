"""Validated pre-encoded JSONL and deterministic per-epoch sample plans."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np


class Manifest:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.rows = []
        ids = set()
        digest = hashlib.sha256()
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row['id']
            if not isinstance(key, str) or not key or key in ids:
                raise ValueError('Manifest IDs must be unique nonempty strings')
            ids.add(key)
            if not isinstance(row['text'], str) or not row['text'].strip():
                raise ValueError('Empty/non-string transcript')
            codes = np.asarray(row['audio_codes'])
            if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] != 16 or codes.dtype.kind not in 'iu':
                raise ValueError('audio_codes must be nonempty integer [T,16]')
            if (codes < 0).any() or (codes >= 2048).any():
                raise ValueError('12Hz audio codes must be in [0,2048)')
            mel_path = (self.path.parent / row['ref_mel_path']).resolve(strict=True)
            mel = np.load(mel_path, allow_pickle=False)
            if mel.ndim != 2 or mel.shape[1] != 128 or mel.shape[0] == 0 or not np.isfinite(mel).all():
                raise ValueError('ref mel must be finite [F,128]')
            content = {k: row[k] for k in ('id', 'text', 'audio_codes')}
            content['mel_sha256'] = hashlib.sha256(mel_path.read_bytes()).hexdigest()
            digest.update(json.dumps(content, sort_keys=True, ensure_ascii=False).encode('utf-8') + b'\n')
            self.rows.append({**row, 'ref_mel_path': str(mel_path)})
        if not self.rows:
            raise ValueError('Manifest must not be empty')
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.rows)


def sample_plan(count, epochs, seed, shuffle):
    if count < 1 or epochs < 1:
        raise ValueError('Dataset and epochs must be positive')
    plan = []
    for epoch in range(epochs):
        order = list(range(count))
        if shuffle:
            random.Random(seed + epoch).shuffle(order)  # does not consume training RNG
        plan.append(order)
    return plan


def planned_windows(plan, accumulation_steps):
    if accumulation_steps < 1:
        raise ValueError('accumulation_steps must be positive')
    # Flush a short window at each epoch boundary, never drop or carry it over.
    return [order[i:i + accumulation_steps] for order in plan
            for i in range(0, len(order), accumulation_steps)]


def validate_cursor(progress, plan_windows):
    if not 0 <= progress.optimizer_steps <= len(plan_windows):
        raise ValueError('Optimizer cursor outside sample plan')
    consumed = sum(len(w) for w in plan_windows[:progress.optimizer_steps])
    if progress.pending_micro_steps or progress.micro_steps != consumed or progress.consumed_samples != consumed:
        raise ValueError('Checkpoint sample cursor does not match plan boundary')
