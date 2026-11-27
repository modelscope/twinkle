"""Manifest-based single-NPU bf16 LoRA training with boundary-only resume.

Each JSONL row: id, text, audio_codes [T,16], ref_mel_path [F,128] .npy.
Output directory must be new; resume writes to a new run directory.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def model_fingerprint(path):
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in ('.safetensors', '.json', '.txt'))
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('Local model must contain safetensors weights')
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        with file.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--train_jsonl', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--resume_from')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--grad_accum', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_decay', type=float, default=1.0, help='StepLR gamma per optimizer update; default keeps LR constant')
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_dropout', type=float, default=0.1)
    ap.add_argument('--save_interval', type=int, default=1, help='Optimizer updates between checkpoints')
    ap.add_argument('--stop_after_updates', type=int, default=0, help='Absolute update limit; 0 runs all epochs')
    args = ap.parse_args(argv)
    if min(args.epochs, args.grad_accum, args.lora_r, args.save_interval) < 1 or args.stop_after_updates < 0:
        ap.error('Counts must be positive (stop_after_updates may be 0)')
    if not math.isfinite(args.lr_decay) or not 0 < args.lr_decay <= 1:
        ap.error('lr_decay must be in (0, 1]')
    if not math.isfinite(args.lr) or not args.lr > 0 or not 0 <= args.lora_dropout < 1:
        ap.error('lr must be positive and dropout in [0,1)')
    return args


def train():
    args = parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import random
    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    import twinkle
    from twinkle import DeviceMesh
    from peft import LoraConfig
    from tts.qwen3.manifest import Manifest, sample_plan, planned_windows, validate_cursor
    from tts.qwen3.runtime import load_twinkle_model, data_config, ADAPTER
    from tts.qwen3.tokenization import load_text_tokenizer
    from tts.qwen3.data import build_assistant_text, build_qwen3_sample, collate_qwen3_batch
    from tts.qwen3.input_processor import Qwen3TTSInputProcessor
    from tts.qwen3.loss import Qwen3TTSLoss
    from tts.qwen3.training import Progress, train_window, save_boundary, resume_boundary

    manifest = Manifest(args.train_jsonl)
    plan = sample_plan(len(manifest), args.epochs, args.seed, args.shuffle)
    schedule = planned_windows(plan, args.grad_accum)
    path = Path(args.model_path).resolve(strict=True)
    print('计算模型内容指纹 / Hashing local model files', flush=True)
    config = dict(format_version=1, manifest=manifest.fingerprint, model=model_fingerprint(path),
        plan=plan, epochs=args.epochs, accumulation_steps=args.grad_accum, seed=args.seed,
        shuffle=args.shuffle, lr=args.lr, lora_r=args.lora_r, lora_dropout=args.lora_dropout,
        targets=['q_proj', 'v_proj'], scheduler=f'StepLR(1,{args.lr_decay})', tokenizer_fix=True)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    torch.npu.set_device(0)
    device_mesh = DeviceMesh.from_sizes(fsdp_size=1, dp_size=1)
    twinkle.initialize(mode='local', global_device_mesh=device_mesh)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    model = load_twinkle_model(path, device_mesh)
    model.model.speech_tokenizer = None  # manifest is already codec-encoded
    model.add_adapter_to_model(ADAPTER, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
        target_modules=['q_proj', 'v_proj'], lora_dropout=args.lora_dropout, bias='none'),
        gradient_accumulation_steps=1)
    model.set_processor(Qwen3TTSInputProcessor, adapter_name=ADAPTER)
    model.set_loss(Qwen3TTSLoss(), adapter_name=ADAPTER)
    model.set_optimizer('AdamW', lr=args.lr, weight_decay=0.0, adapter_name=ADAPTER)
    model.set_lr_scheduler('StepLR', step_size=1, gamma=args.lr_decay, adapter_name=ADAPTER)
    model._lazy_wrap_model()
    cfg = data_config(model.hf_config)
    tokenizer = load_text_tokenizer(path)
    suffix = tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)

    def batch(index):
        row = manifest.rows[index]
        ids = tokenizer.encode(build_assistant_text(row['text']))
        if not suffix or ids[-len(suffix):] != suffix:
            raise ValueError('Unexpected assistant suffix')
        sample = build_qwen3_sample(np.asarray(ids[:-len(suffix)]), np.asarray(row['audio_codes']), cfg)
        mel = np.load(row['ref_mel_path'], allow_pickle=False)
        raw = collate_qwen3_batch([sample], cfg, [mel])
        return model.optimizer_group[ADAPTER].processor(raw, model=model.strategy.unwrap_model(model.model))

    progress = Progress(args.grad_accum)
    if args.resume_from:
        progress = resume_boundary(model, args.resume_from, ADAPTER, fingerprint, args.grad_accum)
    validate_cursor(progress, schedule)
    final_checkpoint = None
    trace = out / 'trace.jsonl'
    with trace.open('w', encoding='utf-8') as log:
        for indices in schedule[progress.optimizer_steps:]:
            if args.stop_after_updates and progress.optimizer_steps >= args.stop_after_updates:
                break
            losses = train_window(model, [batch(i) for i in indices], progress, ADAPTER)
            record = dict(micro_steps=progress.micro_steps, optimizer_steps=progress.optimizer_steps,
                consumed_samples=progress.consumed_samples, sample_ids=[manifest.rows[i]['id'] for i in indices],
                losses=losses, learning_rates=model.optimizer_group[ADAPTER].lr_scheduler.get_last_lr())
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
            log.flush()
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if progress.optimizer_steps % args.save_interval == 0:
                final_checkpoint = save_boundary(model, progress, out,
                    f'checkpoint-update-{progress.optimizer_steps}', ADAPTER, fingerprint)
        if progress.optimizer_steps and (final_checkpoint is None or
                final_checkpoint.name != f'checkpoint-update-{progress.optimizer_steps}'):
            final_checkpoint = save_boundary(model, progress, out,
                f'checkpoint-update-{progress.optimizer_steps}', ADAPTER, fingerprint)
    (out / 'result.json').write_text(json.dumps(dict(pid=os.getpid(), checkpoint=str(final_checkpoint) if final_checkpoint else None,
        micro_steps=progress.micro_steps, optimizer_steps=progress.optimizer_steps,
        consumed_samples=progress.consumed_samples, complete=progress.optimizer_steps == len(schedule)),
        indent=2), encoding='utf-8')
    print('本次运行完成 / Run completed', flush=True)


if __name__ == '__main__':
    train()
