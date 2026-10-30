# Copyright (c) ModelScope Contributors. All rights reserved.
"""Train a three-class CS model with backbone and classification-head LoRA.

From the repository root:
    python cookbook/client/twinkle/prepare_sequence_model.py --tiny
    twinkle-server launch --config cookbook/client/twinkle/task_server_transformers.yaml
    python cookbook/client/twinkle/seq_cls.py

For Megatron with the tiny model, use task_server_megatron.yaml and pass --backend megatron.
For Qwen3.5, prepare with --model ms://Qwen/Qwen3.5-4B --mcore-native, launch
with task_server_megatron_qwen35.yaml, and keep the tokenizer in output/cs-models/seq_cls.
The base checkpoint must contain a fixed three-output head;
this recipe does not train a full head through modules_to_save.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from peft import LoraConfig
from transformers import AutoTokenizer

from twinkle import init_twinkle_client

# Class IDs match the head width in the service YAML: animal=0, fruit=1, vehicle=2.
SAMPLES = [
    ('cat animal', 0), ('dog pet', 0), ('apple fruit', 1), ('banana sweet', 1),
    ('car vehicle', 2), ('boat water', 2), ('cat pet', 0), ('dog animal', 0),
    ('orange fruit', 1), ('apple sweet', 1), ('train vehicle', 2), ('boat vehicle', 2),
]


def encoded_batch(tokenizer):
    rows = []
    for text, class_id in SAMPLES:
        row = dict(tokenizer(text, truncation=True, max_length=128))
        row['labels'] = [class_id]
        row['position_ids'] = list(range(len(row['input_ids'])))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=['transformers', 'megatron'], default='transformers')
    parser.add_argument('--base-url', default=os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000'))
    parser.add_argument('--api-key', default=os.environ.get('TWINKLE_SERVER_TOKEN', 'classification-example'))
    parser.add_argument('--model', default='cs-seq_cls', help='Public model alias in the server YAML')
    parser.add_argument('--tokenizer', default='output/cs-models/seq_cls')
    parser.add_argument('--steps', type=int, default=7, help='Backward batches; save on a GA=2 update boundary')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=2e-5)
    parser.add_argument('--result-json', help='Optional local file for model outputs and training progress')
    args = parser.parse_args()
    if args.steps < 3 or args.steps % 2 == 0:
        parser.error('--steps must be odd and at least 3 so GA=2 has no unsaved pending gradients')
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.atol) or args.atol < 0:
        parser.error('lr must be positive; atol must be finite and nonnegative')
    batch = encoded_batch(AutoTokenizer.from_pretrained(args.tokenizer))
    client = init_twinkle_client(base_url=args.base_url, api_key=args.api_key)
    model = None
    targets = (['q_proj', 'v_proj', 'score'] if args.backend == 'transformers' else
               ['linear_qkv', 'output_layer'])
    forward_kwargs = {'task': 'seq_cls', 'return_logits': True}
    if args.backend == 'megatron':
        forward_kwargs.update(micro_batch_size=6, compute_loss=False)
    try:
        model = client.model(args.model)
        model.add_adapter_to_model('classification', LoraConfig(r=4, lora_alpha=8, target_modules=targets),
                                   gradient_accumulation_steps=2)
        model.set_template('Template', max_length=128)
        model.set_processor('InputProcessor', padding_side='right')
        # Cross entropy consumes one integer class ID per input sequence.
        model.set_loss('SeqClsLoss', problem_type='single_label_classification', num_labels=3)
        model.set_optimizer('Adam', lr=args.lr)
        before = model.forward_only(inputs=batch, **forward_kwargs).result['logits']
        losses = []
        for _ in range(args.steps):
            kwargs = {'task': 'seq_cls', 'gradient_accumulation_steps': 2}
            if args.backend == 'megatron':
                kwargs['micro_batch_size'] = 6
            result = model.forward_backward(inputs=batch, **kwargs).result
            # Call after every backward; the backend gates updates at its GA boundary.
            model.clip_grad_and_step(max_grad_norm=1.0)
            losses.append(float(result['loss']))
        trained = model.forward_only(inputs=batch, **forward_kwargs).result['logits']
        saved = model.save('classification-trained', save_optimizer=True,
                           consumed_train_samples=args.steps * len(batch))
        model.remove_adapter()
        model = None

        # Restore both LoRA (including score-head LoRA) and optimizer, not a new random head.
        model = client.model(args.model)
        model.add_adapter_to_model('restored', LoraConfig(r=4, lora_alpha=8, target_modules=targets),
                                   gradient_accumulation_steps=2)
        model.set_template('Template', max_length=128)
        model.set_processor('InputProcessor', padding_side='right')
        model.set_loss('SeqClsLoss', problem_type='single_label_classification', num_labels=3)
        model.set_optimizer('Adam', lr=args.lr)
        progress = model.resume_from_checkpoint(saved.twinkle_path)
        restored = model.forward_only(inputs=batch, **forward_kwargs).result['logits']
        np.testing.assert_allclose(restored, trained, rtol=0, atol=args.atol)
        for _ in range(2):
            kwargs = {'task': 'seq_cls', 'gradient_accumulation_steps': 2}
            if args.backend == 'megatron':
                kwargs['micro_batch_size'] = 6
            model.forward_backward(inputs=batch, **kwargs)
            model.clip_grad_and_step()
        continued = model.forward_only(inputs=batch, **forward_kwargs).result['logits']
        result = {'task': 'seq_cls', 'backend': args.backend, 'losses': losses,
                  'checkpoint': saved.twinkle_path, 'checkpoint_dir': saved.checkpoint_dir,
                  'progress': progress, 'before': before, 'trained': trained,
                  'restored': restored, 'continued': continued}
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(result, indent=2))
        print(json.dumps({'losses': losses, 'checkpoint': saved.twinkle_path, 'progress': progress}))  # noqa: T201
    finally:
        try:
            if model is not None:
                model.remove_adapter()
        finally:
            client.close()


if __name__ == '__main__':
    main()
