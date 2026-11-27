"""Preserve an explicitly assigned single-speaker train/dev split; never resplit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import unicodedata

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, sr = sf.read(path, dtype='float32')
    if sr not in (16000, 24000) or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Expected finite mono 16/24 kHz audio')
    original = sha(np.asarray(wave, dtype='<f4').tobytes() + str(sr).encode())
    if sr != 24000:
        divisor = math.gcd(sr, 24000)
        wave = resample_poly(wave, 24000 // divisor, sr // divisor).astype(np.float32)
    return wave, original


def audit_split(train_path, dev_path, reference_id):
    splits, seen, speakers = {}, {}, set()
    for split, source in [('train', train_path), ('dev', dev_path)]:
        source = Path(source).resolve(strict=True)
        rows = []
        for line_no, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ('id', 'text', 'group_id', 'speaker_id', 'audio'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'Missing/non-string {key} at {source}:{line_no}')
            audio = (source.parent / row['audio']).resolve(strict=True)
            wave, pcm_hash = read_audio(audio)
            digest = sha(audio.read_bytes())
            text_key = ' '.join(unicodedata.normalize('NFKC', row['text']).casefold().split())
            identities = dict(id=row['id'], group=row['group_id'], text=text_key,
                              file=digest, pcm=pcm_hash,
                              resampled=sha(np.asarray(wave, dtype='<f4').tobytes()))
            for kind, value in identities.items():
                previous = seen.get((kind, value))
                if previous and (kind == 'id' or previous != split):
                    raise ValueError(f'Duplicate ID or cross-split overlap: {kind}')
                seen[kind, value] = split
            speakers.add(row['speaker_id'])
            rows.append(dict(row, audio=str(audio), audio_sha256=digest, pcm_sha256=pcm_hash,
                             source_manifest=str(source), source_line=line_no))
        if not rows:
            raise ValueError(f'{split} must not be empty')
        splits[split] = rows
    if len(speakers) != 1:
        raise ValueError('This preparation entry supports exactly one speaker')
    references = [r for r in splits['train'] if r['id'] == reference_id]
    if len(references) != 1:
        raise ValueError('Reference must be an explicit training sample ID')
    return splits, references[0]


def prepare(train_path, dev_path, output_dir, reference_id, encode, extract_mel):
    splits, reference = audit_split(train_path, dev_path, reference_id)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    ref_wave, _ = read_audio(reference['audio'])
    mel = np.asarray(extract_mel(ref_wave))
    if mel.ndim != 2 or mel.shape[0] == 0 or mel.shape[1] != 128 or not np.isfinite(mel).all():
        raise ValueError('Expected finite reference mel [F,128]')
    np.save(out / 'reference.npy', mel)
    mapping = []
    for split, rows in splits.items():
        encoded = []
        for row in rows:
            if sha(Path(row['audio']).read_bytes()) != row['audio_sha256']:
                raise ValueError('Source audio changed after audit')
            wave, _ = read_audio(row['audio'])
            codes = np.asarray(encode(wave))
            if (codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] != 16 or
                    codes.dtype.kind not in 'iu' or (codes < 0).any() or (codes >= 2048).any()):
                raise ValueError('Expected integer 12Hz codes [T,16] in [0,2048)')
            encoded.append(dict(id=row['id'], text=row['text'], audio_codes=codes.tolist(),
                                ref_mel_path='reference.npy'))
            mapping.append(dict(row, split=split, reference_id=reference_id,
                                output_manifest=f'{split}.jsonl'))
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in encoded), encoding='utf-8')
    report = dict(format_version=1, counts={k: len(v) for k, v in splits.items()},
                  reference_id=reference_id, reference_split='train', mapping=mapping,
                  source_manifest_sha256={k: sha(Path(p).read_bytes()) for k, p in
                                          [('train', train_path), ('dev', dev_path)]},
                  output_sha256={n: sha((out / n).read_bytes()) for n in
                                 ('train.jsonl', 'dev.jsonl', 'reference.npy')})
    (out / 'provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train_jsonl', 'dev_jsonl', 'output_dir', 'reference_id', 'model_path'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    # Audit before loading weights or creating output. No automatic split/filter fallback.
    audit_split(args.train_jsonl, args.dev_jsonl, args.reference_id)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTokenizer
    from audio_features import encode_audio_codes, extract_mel
    codec = Qwen3TTSTokenizer.from_pretrained(str(Path(args.model_path) / 'speech_tokenizer'),
        local_files_only=True, device_map='cpu', torch_dtype=torch.float32)
    codec.model.float().eval()
    with torch.no_grad():
        report = prepare(args.train_jsonl, args.dev_jsonl, args.output_dir, args.reference_id,
                         lambda wave: encode_audio_codes(wave, codec), extract_mel)
    from train_qwen3_tts_lora import model_fingerprint
    report['codec_content_sha256'] = model_fingerprint(Path(args.model_path) / 'speech_tokenizer')
    (Path(args.output_dir) / 'provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('准备完成 / Preparation completed: ' + json.dumps(report['counts']), flush=True)


if __name__ == '__main__':
    main()
