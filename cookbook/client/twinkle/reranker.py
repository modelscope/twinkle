# Copyright (c) ModelScope Contributors. All rights reserved.
"""Train a pointwise reranker over HTTP with LoRA on the backbone and score head.

From the repository root:
    python cookbook/client/twinkle/prepare_sequence_model.py --tiny
    twinkle-server launch --config cookbook/client/twinkle/task_server_transformers.yaml
    python cookbook/client/twinkle/reranker.py

For Megatron with the tiny model, use task_server_megatron.yaml and pass --backend megatron.
For Qwen3.5, prepare with --model ms://Qwen/Qwen3.5-4B --mcore-native, launch
with task_server_megatron_qwen35.yaml, and keep the tokenizer in output/cs-models/reranker.
The base checkpoint must have a fixed one-output score head.
Each input is one query/document sequence; labels are binary relevance targets.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from peft import LoraConfig
from transformers import AutoTokenizer

from twinkle import init_twinkle_client

# Wire labels are integer 0/1; PointwiseRerankerLoss casts them to the score dtype.
PAIRS = [
    ('cat animal', 'cat pet', 1), ('cat animal', 'plane sky', 0),
    ('dog animal', 'dog pet', 1), ('dog animal', 'stock market', 0),
    ('apple fruit', 'apple sweet', 1), ('apple fruit', 'car vehicle', 0),
    ('boat vehicle', 'boat water', 1), ('boat vehicle', 'banana fruit', 0),
    ('cat animal', 'cat pet', 1), ('cat animal', 'plane sky', 0),
    ('dog animal', 'dog pet', 1), ('dog animal', 'stock market', 0),
]


def encoded_batch(tokenizer):
    rows = []
    for query, document, relevance in PAIRS:
        row = dict(tokenizer('Query: ' + query + '\nDocument: ' + document,
                             truncation=True, max_length=128))
        row['labels'] = [relevance]
        row['position_ids'] = list(range(len(row['input_ids'])))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=['transformers', 'megatron'], default='transformers')
    parser.add_argument('--base-url', default=os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000'))
    parser.add_argument('--api-key', default=os.environ.get('TWINKLE_SERVER_TOKEN', 'reranker-example'))
    parser.add_argument('--model', default='cs-reranker', help='Public model alias in the server YAML')
    parser.add_argument('--tokenizer', default='output/cs-models/reranker')
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
    forward_kwargs = {'task': 'reranker', 'return_logits': True}
    if args.backend == 'megatron':
        forward_kwargs.update(micro_batch_size=6, compute_loss=False)
    try:
        # score/output_layer LoRA is explicitly selected, not a full shared trainable head.
        model = client.model(args.model)
        model.add_adapter_to_model('reranker', LoraConfig(r=4, lora_alpha=8, target_modules=targets),
                                   gradient_accumulation_steps=2)
        model.set_template('Template', max_length=128)
        model.set_processor('InputProcessor', padding_side='right')
        model.set_loss('PointwiseRerankerLoss')
        model.set_optimizer('Adam', lr=args.lr)
        before = model.forward_only(inputs=batch, **forward_kwargs).result['logits']
        losses = []
        for _ in range(args.steps):
            kwargs = {'task': 'reranker', 'gradient_accumulation_steps': 2}
            if args.backend == 'megatron':
                kwargs['micro_batch_size'] = 6
            result = model.forward_backward(inputs=batch, **kwargs).result
            # Call after every backward; the backend gates updates at its GA boundary.
            model.clip_grad_and_step(max_grad_norm=1.0)
            losses.append(float(result['loss']))
        trained = model.forward_only(inputs=batch, **forward_kwargs).result['logits']
        saved = model.save('reranker-trained', save_optimizer=True,
                           consumed_train_samples=args.steps * len(batch))
        model.remove_adapter()
        model = None

        # A cold worker loads the same persisted base head before restoring its LoRA.
        model = client.model(args.model)
        model.add_adapter_to_model('restored', LoraConfig(r=4, lora_alpha=8, target_modules=targets),
                                   gradient_accumulation_steps=2)
        model.set_template('Template', max_length=128)
        model.set_processor('InputProcessor', padding_side='right')
        model.set_loss('PointwiseRerankerLoss')
        model.set_optimizer('Adam', lr=args.lr)
        progress = model.resume_from_checkpoint(saved.twinkle_path)
        restored = model.forward_only(inputs=batch, **forward_kwargs).result['logits']
        np.testing.assert_allclose(restored, trained, rtol=0, atol=args.atol)
        for _ in range(2):
            kwargs = {'task': 'reranker', 'gradient_accumulation_steps': 2}
            if args.backend == 'megatron':
                kwargs['micro_batch_size'] = 6
            model.forward_backward(inputs=batch, **kwargs)
            model.clip_grad_and_step()
        continued = model.forward_only(inputs=batch, **forward_kwargs).result['logits']
        result = {'task': 'reranker', 'backend': args.backend, 'losses': losses,
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
