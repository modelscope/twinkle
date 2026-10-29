# Copyright (c) ModelScope Contributors. All rights reserved.
"""Train and evaluate typed decisions through Twinkle's Transformers backend."""
import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer

import twinkle
from twinkle.loss.decision import DecisionLoss
from twinkle.metric.decision import DecisionAccuracy
from twinkle.model import TransformersModel
from twinkle.model.decision.encoding import DecisionEncoder
from twinkle.model.decision.model import activate_decision_head, attach_decision_head, decision_lora_config, initialize_missing_adapter_head
from twinkle.preprocessor.decision import (ClefPreprocessor, DecisionAugmentConfig, JevPreprocessor,
                                           OmniJevPreprocessor, OpenJevPreprocessor, StartLuxPreprocessor)
from twinkle.processor.decision import DecisionInputProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'infer'])
    parser.add_argument('--model', required=True, help='Local base or merged checkpoint directory')
    parser.add_argument('--family', choices=['generic', 'jev', 'clef', 'omnijev', 'startlux'], default='generic')
    parser.add_argument('--head-dir', help='Original decision head artifacts, required for released models')
    parser.add_argument('--adapter', help='Released or trained PEFT adapter directory')
    parser.add_argument('--resume', help='Resume optimizer and model state from a training checkpoint')
    parser.add_argument('--data', required=True, help='Local JSONL; use disjoint files for training and testing')
    parser.add_argument('--open-jev-format', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--loss', choices=['scoring', 'distill_rps', 'smooth_brier', 'ce_rps'])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--lora-targets', nargs='+', default=['q_proj', 'v_proj'])
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    parser.add_argument('--shuffle-fields', action='store_true')
    parser.add_argument('--drop-instructions', type=float, default=0)
    parser.add_argument('--subsample-options', type=int, default=0)
    args = parser.parse_args()
    if args.resume and (args.action != 'train' or args.adapter):
        raise ValueError('Use --resume for training recovery; --adapter is a separate weights-only initialization')
    if args.family == 'startlux' and (args.shuffle_fields or args.drop_instructions or args.subsample_options):
        raise ValueError('StartLux augmentation is not supported; preserve the official renderer')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    if args.steps < 1:
        raise ValueError('steps must be positive')
    twinkle.initialize(mode='local')
    if args.family in ('omnijev', 'jev'):
        processor = AutoProcessor.from_pretrained(args.model)
    else:
        processor = AutoTokenizer.from_pretrained(args.model)
    tokenizer = getattr(processor, 'tokenizer', processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer needs a pad or EOS token')
    wrapper = TransformersModel(model_id=args.model, dtype=getattr(torch, args.dtype),
                                mixed_precision='no' if args.dtype == 'float32' else 'bf16',
                                attn_implementation='sdpa')
    attach_decision_head(wrapper.model, processor, args.family, args.head_dir or args.model)
    if args.adapter:
        from peft import PeftConfig, set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        adapter = PeftConfig.from_pretrained(args.adapter)
        adapter.inference_mode = args.action != 'train'
        factory_head = 'scoring_head' not in (adapter.modules_to_save or [])
        adapter.modules_to_save = list(dict.fromkeys(list(adapter.modules_to_save or []) + ['scoring_head']))
        wrapper.add_adapter_to_model('decision', adapter)
        state = load_peft_weights(args.adapter, device='cpu')
        if factory_head:
            state = initialize_missing_adapter_head(wrapper.model, state, 'decision')
        restored = set_peft_model_state_dict(wrapper.model, state, adapter_name='decision')
        missing_lora = [key for key in restored.missing_keys if 'lora_' in key]
        if missing_lora:
            raise ValueError(f'Missing adapter weights: {missing_lora}')
        if restored.unexpected_keys:
            raise ValueError(f'Unexpected adapter keys: {restored.unexpected_keys}')
    else:
        wrapper.add_adapter_to_model('decision', decision_lora_config(
            r=8, lora_alpha=16, target_modules=args.lora_targets))
    activate_decision_head(wrapper.model)
    wrapper._default_tokenizer = processor
    wrapper.set_processor(DecisionInputProcessor, pad_token_id=tokenizer.pad_token_id,
                          padding_side='right' if args.action == 'train' else 'left')
    mode = args.loss or {'generic': 'scoring', 'jev': 'distill_rps', 'clef': 'smooth_brier',
                        'omnijev': 'ce_rps', 'startlux': 'scoring'}[args.family]
    wrapper.set_loss(DecisionLoss, mode=mode)
    if args.action == 'train':
        wrapper.set_optimizer('AdamW', lr=args.lr, foreach=False)
    start_step = 0
    if args.resume:
        progress = wrapper.resume_from_checkpoint(args.resume, adapter_name='decision')
        start_step = int(progress['consumed_train_samples'])
        if start_step >= args.steps:
            raise ValueError('steps must exceed the already completed step count')
    # Token-level accuracy is not defined for a distribution-valued label.
    group = wrapper.optimizer_group['decision']
    for state in (group.train_status, group.eval_status):
        state.metrics = [DecisionAccuracy(wrapper.device_mesh, group._dp_group)]
    config = None
    if args.action == 'train':
        config = DecisionAugmentConfig(drop_instructions=args.drop_instructions,
            subsample_options=args.subsample_options, shuffle_fields=args.shuffle_fields)
    pre_cls = {'generic': JevPreprocessor, 'jev': JevPreprocessor, 'clef': ClefPreprocessor,
               'omnijev': OmniJevPreprocessor, 'startlux': StartLuxPreprocessor}[args.family]
    source = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    preprocessor = pre_cls(augment_config=config)
    if args.open_jev_format:
        if args.family == 'startlux':
            raise ValueError('Open-Jev normalization supports generic/JEV/Clef/OmniJev schemas')
        preprocessor = OpenJevPreprocessor(pre_cls(), augment_config=config)
    normalized, _ = preprocessor(source)
    encoder = DecisionEncoder(processor, args.family, args.max_length)
    rows = [encoder(row) for row in normalized]
    if not rows:
        raise ValueError('Dataset has no decision records')
    output.mkdir(parents=True)
    (output / 'run.json').write_text(json.dumps(vars(args), indent=2))
    if args.action == 'train':
        if any('target_probs' not in row for row in rows):
            raise ValueError('All training records require targets')
        for step in range(start_step, args.steps):
            result = wrapper.forward_backward(inputs=[rows[step % len(rows)]])
            if not math.isfinite(float(result['loss'])):
                raise RuntimeError('Non-finite decision loss')
            norm = wrapper.clip_grad_norm(max_grad_norm=1.)
            if norm is not None and not math.isfinite(float(norm)):
                raise RuntimeError('Non-finite decision gradient')
            wrapper.step(); wrapper.zero_grad()
            print(json.dumps({'step': step + 1, 'loss': float(result['loss'])}), flush=True)
        wrapper.save(name='final', output_dir=str(output), adapter_name='decision', save_optimizer=True,
                     consumed_train_samples=args.steps)
    else:
        metric = DecisionAccuracy()
        with (output / 'predictions.jsonl').open('w') as stream:
            for row in rows:
                result = wrapper.forward_only(inputs=[row], return_logits=True)
                status = group.eval_status
                metric.accumulate(status.inputs, status.outputs)
                scores = result['logits'].masked_fill(~result['option_mask'], -torch.inf)
                probs = scores.float().softmax(-1).cpu().tolist()
                stream.write(json.dumps({'probabilities': probs}) + '\n')
        (output / 'metrics.json').write_text(json.dumps(metric.calculate(), indent=2))


if __name__ == '__main__':
    main()
