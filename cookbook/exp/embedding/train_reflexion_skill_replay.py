"""Replay-train the reflexion skill model from prebuilt exact RFT data.

Use ``build_reflexion_skill_data.py`` first to create ``skill_dataset.jsonl``. This
script trains only the skill model from those frozen records; it does not run vLLM
rollouts, leak checks, or rubric diagnosis.

Launch:
    python cookbook/exp/embedding/train_reflexion_skill_replay.py \
        --data ./output/reflexion_skill_data/skill_dataset.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import train_reflexion_skill_rft as rft


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='./output/reflexion_skill_data/skill_dataset.jsonl')
    p.add_argument('--output-dir', default='./output/reflexion_skill_replay')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--sft-batch-size', type=int, default=8)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--grpo-epsilon', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--save-rounds', type=int, default=50)
    return p.parse_args()


def _load_chunks(path: str) -> List[List[Dict[str, Any]]]:
    chunks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    fallback_chunk = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('record_type') == 'config':
                continue
            for key in ('problem', 'response', 'advantage'):
                if key not in row:
                    raise ValueError(f'{path}:{line_no} missing required field {key!r}')
            ci = int(row.get('chunk', fallback_chunk))
            chunks[ci].append(row)
            if 'chunk' not in row and len(chunks[ci]) >= 64:
                fallback_chunk += 1
    return [chunks[k] for k in sorted(chunks) if chunks[k]]


def _init_model(args: argparse.Namespace, total_updates: int):
    train_mesh = rft.DeviceMesh.from_sizes(
        world_size=rft.TRAIN_GPUS, dp_size=rft.TRAIN_DP, fsdp_size=rft.TRAIN_FSDP)
    device_groups = [
        rft.DeviceGroup(name='train', ranks=list(range(rft.TRAIN_GPUS)), device_type='GPU'),
    ]
    rft.twinkle.initialize(mode='ray', nproc_per_node=rft.TRAIN_GPUS, groups=device_groups,
                           lazy_collect=False)
    model = rft.TransformersModel(model_id=rft.GEN_MODEL_ID, device_mesh=train_mesh,
                                  remote_group='train', ddp_config={'find_unused_parameters': False})
    from twinkle.patch.no_split_modules import NoSplitModulesPatch
    model.apply_patch(NoSplitModulesPatch({'Qwen3DecoderLayer'}))
    model.set_template(rft.Template, model_id=rft.GEN_MODEL_ID,
                       enable_thinking=True, max_length=args.max_model_len,
                       truncation_strategy='delete')
    model.set_processor(rft.InputProcessor, padding_free=False)
    model.set_loss('GRPOLoss', epsilon=args.grpo_epsilon)
    model.set_optimizer('AdamW', lr=args.lr)
    model.set_lr_scheduler('CosineWarmupScheduler', num_warmup_steps=10,
                           num_training_steps=max(1, total_updates))
    return model


def main() -> None:
    args = _build_args()
    if args.sft_batch_size % rft.TRAIN_DP != 0:
        raise ValueError(f'--sft-batch-size ({args.sft_batch_size}) must be a multiple '
                         f'of the training dp size ({rft.TRAIN_DP})')
    chunks = _load_chunks(args.data)
    if not chunks:
        raise ValueError(f'no train records found in {args.data}')
    os.makedirs(args.output_dir, exist_ok=True)
    total_updates = len(chunks) * args.epochs
    model = _init_model(args, total_updates)
    log_path = os.path.join(args.output_dir, 'train_log.jsonl')
    cfg = {'record_type': 'config', 'mode': 'offline_replay', 'data': args.data,
           'chunks': len(chunks), 'epochs': args.epochs, 'lr': args.lr,
           'sft_batch_size': args.sft_batch_size}
    rounds = 0
    with open(log_path, 'w', encoding='utf-8') as tlog:
        tlog.write(json.dumps(cfg, ensure_ascii=False) + '\n')
        for epoch in range(args.epochs):
            for ci, samples in enumerate(chunks):
                log = rft._train_chunk(model, None, samples, args)
                rounds += 1
                log.update({'record_type': 'train_round', 'round': rounds,
                            'epoch': epoch, 'chunk': ci})
                tlog.write(json.dumps(log, ensure_ascii=False) + '\n')
                tlog.flush()
                sys.stderr.write(
                    f'[replay-rft] e{epoch} c{ci}: n={log["n_samples"]} '
                    f'micro={log["n_micro_batches"]} metric={log.get("metric")}\n')
                if rounds % args.save_rounds == 0:
                    model.save(f'skill-rft-replay-{rounds}', output_dir=args.output_dir)
    model.save('skill-rft-replay-final', output_dir=args.output_dir)
    sys.stderr.write(f'[replay-rft] done: {rounds} updates; log -> {log_path}\n')


if __name__ == '__main__':
    main()
