# Copyright (c) ModelScope Contributors. All rights reserved.
"""Train StartLux dense/MoE decision adapters through Twinkle's public API."""
import argparse
import json
import math
import os
import torch
from decision_training import (audit_splits, audit_trainable, candidate_loss, collate, copy_metadata, encode_records,
                               install_head, lora_targets)
from pathlib import Path
from peft import LoraConfig
from transformers import AutoTokenizer

import twinkle
from twinkle import DeviceMesh
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.kernel.ops.fla.npu import apply_qwen3_5_fla
from twinkle.loss.base import Loss
from twinkle.metric.accuracy import Accuracy
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor


class DecisionProcessor(InputProcessor):

    def __init__(self, pad_token_id, **kwargs):
        super().__init__(padding_free=False, **kwargs)
        self.pad_token_id = pad_token_id

    def __call__(self, inputs, **kwargs):
        batch = collate(inputs, self.pad_token_id)
        device = next(kwargs['model'].parameters()).device
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class DecisionLoss(Loss):
    require_logits = True
    require_logps = False

    def __call__(self, inputs, outputs, **kwargs):
        loss = candidate_loss(outputs['logits'], inputs['attention_mask'], inputs['labels'])
        return {'loss': loss.mean(), 'num_tokens': 0}

    def micro_batch_scale(self, inputs, indices):
        return len(indices) / len(inputs)


class DecisionAccuracy(Accuracy):
    """Count one prediction per question, rather than language-model tokens."""

    def accumulate(self, inputs, outputs, **kwargs):
        logits = outputs.get('logits')
        if logits is None:
            return
        positions = inputs['attention_mask'].sum(-1) - 1
        selected = logits[torch.arange(logits.shape[0], device=logits.device), positions]
        targets = inputs['labels']
        predictions = selected.masked_fill(targets < 0, -torch.inf).argmax(-1)
        self.total_correct += (predictions == targets.argmax(-1)).sum().item()
        self.total_count += targets.shape[0]

    def calculate(self):
        result = super().calculate()
        if 'accuracy' in result:
            result['soft_target_mode_agreement'] = result.pop('accuracy')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--save-steps', type=int, default=50)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=2, help='Micro-batch size per data rank')
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora-scope', choices=['attention', 'language-linear'], default='attention')
    parser.add_argument('--resume')
    args = parser.parse_args()
    if Path(args.output).exists() and not args.resume:
        raise FileExistsError('Use a fresh output directory')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    mesh = DeviceMesh.from_sizes(fsdp_size=world, dp_size=1, device_type='npu')
    twinkle.initialize(mode='local', nproc_per_node=world, global_device_mesh=mesh, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train, train_ids = encode_records(args.train, tokenizer)
    dev, dev_ids = encode_records(args.dev, tokenizer)
    audit_splits(train_ids, dev_ids)
    generator = torch.Generator().manual_seed(42)
    global_batch_size = args.batch_size * mesh.data_world_size
    ordered = []
    while len(ordered) < args.steps * args.accum * global_batch_size:
        ordered.extend(train[i] for i in torch.randperm(len(train), generator=generator).tolist())
    ordered = ordered[:args.steps * args.accum * global_batch_size]
    loader = DataLoader(
        dataset=Dataset(dataset_meta=DatasetMeta(data=ordered)),
        device_mesh=mesh,
        batch_size=global_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True)
    apply_qwen3_5_fla()
    model = TransformersModel(
        model_id=args.model,
        device_mesh=mesh,
        dtype=torch.bfloat16,
        mixed_precision='bf16',
        attn_implementation='sdpa',
        strategy='native_fsdp')
    install_head(model.model, args.model, tokenizer)
    apply_qwen3_5_fla(model)
    model._default_tokenizer = tokenizer
    model.add_adapter_to_model(
        'decision',
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=lora_targets(model.model, args.lora_scope)))
    audit = audit_trainable(model.model)
    model.set_processor(DecisionProcessor, pad_token_id=tokenizer.pad_token_id)
    model.set_loss(DecisionLoss)
    model.set_optimizer('AdamW', lr=args.lr, weight_decay=0., foreach=False)
    if args.resume:
        progress = model.resume_from_checkpoint(args.resume, adapter_name='decision')
        loader.resume_from_checkpoint(progress['consumed_train_samples'])
    group = model.optimizer_group['decision']
    for status in (group.train_status, group.eval_status):
        status.metrics = [
            DecisionAccuracy(mesh, group._dp_group) if isinstance(metric, Accuracy) else metric
            for metric in status.metrics
        ]
    if rank == 0:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    def evaluate():
        losses = []
        for row in dev:
            model.forward_only(inputs=[row])
            losses.append(model.calculate_loss(is_training=False))
        return math.fsum(losses) / len(losses)

    before = evaluate()
    for batch in loader:
        result = model.forward_backward(inputs=batch, gradient_accumulation_steps=args.accum)
        if not math.isfinite(float(result['loss'])):
            raise RuntimeError('Non-finite training loss')
        norm = model.clip_grad_norm(max_grad_norm=1., gradient_accumulation_steps=args.accum)
        if norm is not None and not math.isfinite(float(norm)):
            raise RuntimeError('Non-finite gradient norm')
        model.step(gradient_accumulation_steps=args.accum)
        model.zero_grad(gradient_accumulation_steps=args.accum)
        model.lr_step(gradient_accumulation_steps=args.accum)
        if rank == 0 and group.cur_step % args.accum == 0:
            print(
                json.dumps({
                    'step': group.cur_step // args.accum,
                    'loss': float(result['loss']),
                    'grad_norm': float(norm)
                }),
                flush=True)
        if group.cur_step % (args.accum * args.save_steps) == 0:
            model.save(
                name=f'checkpoint-{group.cur_step // args.accum}',
                output_dir=args.output,
                adapter_name='decision',
                save_optimizer=True,
                consumed_train_samples=loader.get_state()['consumed_train_samples'])
    after = evaluate()
    checkpoint = model.save(
        name='checkpoint-final',
        output_dir=args.output,
        adapter_name='decision',
        save_optimizer=True,
        consumed_train_samples=loader.get_state()['consumed_train_samples'])
    if rank == 0:
        copy_metadata(args.model, checkpoint)
        Path(args.output, 'validation.json').write_text(
            json.dumps(
                {
                    'before_loss': before,
                    'after_loss': after,
                    'steps': group.cur_step // args.accum,
                    'checkpoint': str(checkpoint),
                    'training_args': vars(args),
                    'trainable': audit,
                    'scope': 'held-out validation loss; independent test accuracy is required'
                },
                indent=2))


if __name__ == '__main__':
    main()
