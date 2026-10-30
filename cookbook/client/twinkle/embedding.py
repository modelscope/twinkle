# Copyright (c) ModelScope Contributors. All rights reserved.
"""Train an embedding adapter over HTTP, save it and continue after cold recreation.

From the repository root:
    python cookbook/client/twinkle/prepare_sequence_model.py --tiny
    twinkle-server launch --config cookbook/client/twinkle/task_server_transformers.yaml
    python cookbook/client/twinkle/embedding.py

For Megatron with the tiny model, use task_server_megatron.yaml and run this example
with --backend megatron. For Qwen3.5, prepare with --model ms://Qwen/Qwen3.5-4B
--mcore-native, launch with task_server_megatron_qwen35.yaml, and keep the tokenizer
in output/cs-models/embedding. No sampler service is required.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from peft import LoraConfig
from transformers import AutoTokenizer

from twinkle import init_twinkle_client

# Each group is (query, positive, negative). Labels mark group starts, not relevance.
TRIPLETS = [
    ('cat animal', 'cat pet', 'plane sky'),
    ('dog animal', 'dog pet', 'stock market'),
    ('apple fruit', 'apple sweet', 'car vehicle'),
    ('boat vehicle', 'boat water', 'banana fruit'),
]


def encoded_batch(tokenizer):
    rows = []
    for query, positive, negative in TRIPLETS:
        for text, boundary in [(query, 1), (positive, 0), (negative, 0)]:
            row = dict(tokenizer(text, truncation=True, max_length=128))
            row['labels'] = [boundary]
            row['position_ids'] = list(range(len(row['input_ids'])))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=['transformers', 'megatron'], default='transformers')
    parser.add_argument('--base-url', default=os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000'))
    parser.add_argument('--api-key', default=os.environ.get('TWINKLE_SERVER_TOKEN', 'embedding-example'))
    parser.add_argument('--model', default='cs-embedding', help='Public model alias in the server YAML')
    parser.add_argument('--tokenizer', default='output/cs-models/embedding')
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
    targets = ['q_proj', 'v_proj'] if args.backend == 'transformers' else ['linear_qkv']
    forward_kwargs = {'task': 'embedding', 'return_logits': True}
    if args.backend == 'megatron':
        forward_kwargs.update(micro_batch_size=6, compute_loss=False)
    try:
        # Allocate a tenant handle on the server's frozen causal text backbone.
        model = client.model(args.model)
        model.add_adapter_to_model('embedding', LoraConfig(r=4, lora_alpha=8, target_modules=targets),
                                   gradient_accumulation_steps=2)
        model.set_template('Template', max_length=128)
        model.set_processor('InputProcessor', padding_side='right')
        model.set_loss('InfonceLoss', temperature=0.1, use_batch=True)
        model.set_optimizer('Adam', lr=args.lr)
        before = model.forward_only(inputs=batch, **forward_kwargs).result['embeddings']
        losses = []
        for _ in range(args.steps):
            kwargs = {'task': 'embedding', 'gradient_accumulation_steps': 2}
            if args.backend == 'megatron':
                # DP=2 gives six rows per rank, preserving two complete triplets.
                kwargs['micro_batch_size'] = 6
            result = model.forward_backward(inputs=batch, **kwargs).result
            # Call after every backward; the backend gates updates at its GA boundary.
            model.clip_grad_and_step(max_grad_norm=1.0)
            losses.append(float(result['loss']))
        trained = model.forward_only(inputs=batch, **forward_kwargs).result['embeddings']
        saved = model.save('embedding-trained', save_optimizer=True,
                           consumed_train_samples=args.steps * len(batch))
        model.remove_adapter()
        model = None

        # Reload the same persisted base, then restore adapter and optimizer together.
        model = client.model(args.model)
        model.add_adapter_to_model('restored', LoraConfig(r=4, lora_alpha=8, target_modules=targets),
                                   gradient_accumulation_steps=2)
        model.set_template('Template', max_length=128)
        model.set_processor('InputProcessor', padding_side='right')
        model.set_loss('InfonceLoss', temperature=0.1, use_batch=True)
        model.set_optimizer('Adam', lr=args.lr)
        progress = model.resume_from_checkpoint(saved.twinkle_path)
        restored = model.forward_only(inputs=batch, **forward_kwargs).result['embeddings']
        np.testing.assert_allclose(restored, trained, rtol=0, atol=args.atol)
        for _ in range(2):
            kwargs = {'task': 'embedding', 'gradient_accumulation_steps': 2}
            if args.backend == 'megatron':
                kwargs['micro_batch_size'] = 6
            model.forward_backward(inputs=batch, **kwargs)
            model.clip_grad_and_step()
        continued = model.forward_only(inputs=batch, **forward_kwargs).result['embeddings']
        result = {'task': 'embedding', 'backend': args.backend, 'losses': losses,
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
