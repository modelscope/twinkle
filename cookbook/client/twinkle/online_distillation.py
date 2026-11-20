# Copyright (c) ModelScope Contributors. All rights reserved.
"""Learn online distillation with a small, reproducible multiplication task."""
from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import ExitStack
from dataclasses import asdict
from itertools import combinations_with_replacement, cycle, islice
from uuid import uuid4

import numpy as np
from twinkle import init_twinkle_client
from twinkle.data_format import SamplingParams
from twinkle.processor.trajectory import response_positions

STUDENT_NAME = 'distill-student'
TEACHER_NAME = 'distill-teacher'
MAX_LENGTH = 256
TEMPLATE_CLASS = 'swift.dev.template.Template'
LORA = {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0, 'target_modules': 'all-linear'}
LOSS = {'gkd': 'GKDLoss', 'opd-rl': 'GRPOLoss', 'opsd': 'OPSDLoss'}


def question(pair):
    left, right = pair
    return {
        'messages': [
            {'role': 'system', 'content': 'Answer with just the final integer, without explanation.'},
            {'role': 'user', 'content': f'What is {left} times {right}?'},
        ]
    }


def worked_solution(pair):
    left, right = pair
    tens, ones = divmod(right, 10)
    tens *= 10
    return f'{left} × {tens} + {left} × {ones} = {left * tens} + {left * ones} = {left * right}.'


def sample(sampler, pairs, policy, seed, *, temperature=1.0, logprobs=None):
    # SamplingParams defaults leave top-p/top-k and repetition penalties neutral.
    params = SamplingParams(max_tokens=16, temperature=temperature, logprobs=logprobs, seed=seed)
    responses = sampler.sample([question(pair) for pair in pairs], params, adapter_uri=policy)
    return [response.sequences[0] for response in responses]


def training_feature(feature, tokens):
    positions = response_positions(feature)
    if (not tokens or [feature['labels'][i] for i in positions] != tokens
            or [feature['input_ids'][i + 1] for i in positions] != tokens):
        raise RuntimeError('Training labels must match the sampled response token IDs.')
    return {key: feature[key] for key in ('input_ids', 'labels', 'attention_mask', 'completion_mask') if key in feature}


def response_scores(model, rows, token_rows, **kwargs):
    inputs = [dict(row, episode_id=str(index), response_token_ids=tokens)
              for index, (row, tokens) in enumerate(zip(rows, token_rows, strict=True))]
    result = model.forward_only(inputs=inputs, output_fields=['logps'], output_frame='response', **kwargs).result
    if (result.get('episode_ids') != [row['episode_id'] for row in inputs]
            or result.get('response_token_ids') != token_rows
            or (result.get('field_frames') or {}).get('logps') != 'response'):
        raise RuntimeError('Teacher scores do not match the sampled reply rows and token IDs.')
    scores = result['logps']
    for tokens, row in zip(token_rows, scores, strict=True):
        if len(row) != len(tokens) or not np.isfinite(row).all():
            raise RuntimeError('Expected one finite teacher logprob per sampled token.')
    return scores


def distillation_signal(method, model, teacher, pairs, sequences, features):
    if method == 'gkd':
        # Full vocabulary scores make the HTTP payload grow with the batch size.
        result = teacher.forward_only(inputs=features, return_logits=True).result
        batch_size = len(features)
        length = max(len(feature['input_ids']) for feature in features)
        logits = np.asarray(result['logits'], dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[None, ...]
        if logits.ndim != 3 or logits.shape[:2] != (batch_size, length):
            raise RuntimeError(
                f'Expected teacher logits shaped [{batch_size}, {length}, vocab_size], got {logits.shape}.')
        return {'teacher_logits': logits.tolist()}

    token_rows = [seq.tokens for seq in sequences]
    if method == 'opd-rl':
        from twinkle_agentic.rollout.ledger import sampled_token_logprobs

        old_logps = [sampled_token_logprobs(seq.tokens, seq.logprobs) for seq in sequences]
        teacher_logps = response_scores(teacher, features, token_rows)
        # Pure OPD-RL: base task advantage is zero, so A_t = teacher_logp - rollout_logp.
        advantages = [[target - old for target, old in zip(target_row, old_row, strict=True)]
                      for target_row, old_row in zip(teacher_logps, old_logps, strict=True)]
        return {'advantages': advantages, 'old_logps': old_logps}

    # OPSD: the current student reads the solution and scores its own original reply.
    prompts = [question(pair) for pair in pairs]
    for prompt, pair in zip(prompts, pairs, strict=True):
        prompt['messages'][-1]['content'] += f'\nReference solution: {worked_solution(pair)}'
    return {'teacher_logps': response_scores(model, prompts, token_rows, continuation_token_ids=token_rows)}


def evaluate(sampler, tokenizer, pairs, policy, label, seed):
    # Fixed held-out questions, greedy sampling and no solution in the prompt.
    sequences = sample(sampler, pairs, policy, seed, temperature=0.0)
    if tokenizer is None and any(seq.text is None for seq in sequences):
        raise RuntimeError('OPSD evaluation requires decoded assistant text from the sampler.')
    texts = [(seq.text if tokenizer is None else tokenizer.decode(seq.tokens, skip_special_tokens=True)).strip()
             for seq in sequences]
    correct = sum(text == str(a * b) for (a, b), text in zip(pairs, texts, strict=True))
    print(json.dumps({
        'eval': label, 'correct': correct, 'total': len(pairs), 'accuracy': correct / len(pairs),
        'mean_output_tokens': sum(len(seq.tokens) for seq in sequences) / len(pairs),
        'examples': [{'question': f'{a} × {b}', 'answer': text, 'expected': a * b}
                     for (a, b), text in zip(pairs[:3], texts[:3], strict=True)],
    }), flush=True)


def train(method, steps, eval_size, seed, model_name, student_path=None, teacher_path=None, *, batch_size=None):
    # Use unordered operand pairs so '17 × 23' and '23 × 17' cannot leak across splits.
    pairs = list(combinations_with_replacement(range(11, 30), 2))
    random.Random(seed).shuffle(pairs)
    evaluation, training = pairs[:eval_size], cycle(pairs[eval_size:])
    if batch_size is None:
        batch_size = 1 if method == 'gkd' else 2
    tokenizer = server_spec = None
    if method != 'opsd':
        from swift.dev.builders.template import build_template_spec
        from swift.dev.config import ModelConfig, TemplateConfig
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(student_path)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        special_ids = ('bos_token_id', 'eos_token_id', 'pad_token_id')
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab() or any(
                getattr(tokenizer, key) != getattr(teacher_tokenizer, key) for key in special_ids):
            raise ValueError('The student and external teacher must share the same token IDs.')

        spec = build_template_spec(
            TemplateConfig(template='qwen3_5', max_length=MAX_LENGTH, padding_side='right', enable_thinking=False),
            ModelConfig(model=student_path, model_type='qwen3_5', torch_dtype='bfloat16'))
        server_spec = asdict(spec)
        server_spec['model']['model_id_or_dir'] = os.getenv('TWINKLE_DISTILL_SERVER_STUDENT_PATH', student_path)

    run = f'{method}-{uuid4().hex[:8]}'
    with init_twinkle_client() as client, ExitStack() as adapters:
        model, sampler = client.model(model_name), client.sampler(model_name)
        model.add_adapter_to_model(run, LORA, gradient_accumulation_steps=1)
        adapters.callback(model.remove_adapter, run)
        if server_spec is not None:
            model.set_template(TEMPLATE_CLASS, template_spec=server_spec)
            sampler.set_template(TEMPLATE_CLASS, template_spec=server_spec)
        model.set_processor('InputProcessor', padding_side='right')
        model.set_loss(LOSS[method])
        model.set_optimizer('AdamW', lr=1e-4, weight_decay=0.0)
        teacher = None
        if method != 'opsd':
            teacher = client.model(TEACHER_NAME)
            teacher.add_adapter_to_model(f'{run}-teacher', None)
            adapters.callback(teacher.remove_adapter, f'{run}-teacher')
            teacher.set_processor('InputProcessor', padding_side='right')

        policy = model.save(f'{run}-policy-0', save_optimizer=False).twinkle_path
        evaluate(sampler, tokenizer, evaluation, policy, 'before', seed)
        for step in range(1, steps + 1):
            batch = list(islice(training, batch_size))
            sequences = sample(sampler, batch, policy, seed + step, logprobs=1 if method == 'opd-rl' else None)
            features = [training_feature(seq.new_input_feature, seq.tokens) for seq in sequences]
            signal = distillation_signal(method, model, teacher, batch, sequences, features)
            model.forward_backward(inputs=features, **signal)
            model.clip_grad_and_step()
            metrics = model.calculate_metric(is_training=True).result
            policy = model.save(f'{run}-policy-{step}', save_optimizer=False).twinkle_path
            print(json.dumps({'method': method, 'step': step, 'policy': policy, 'metrics': metrics}), flush=True)
        evaluate(sampler, tokenizer, evaluation, policy, 'after', seed)
        checkpoint = model.save(f'{run}-resume', save_optimizer=True, consumed_train_samples=steps * batch_size)
        print(json.dumps({'resume_from': checkpoint.twinkle_path}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=LOSS, default='opsd')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--eval-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default=STUDENT_NAME, help='Student model name published by the server')
    parser.add_argument('--batch-size', type=int, help='Training examples per step (default: 1 for GKD, 2 otherwise)')
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.eval_size <= 64:
        parser.error('steps must be positive and eval-size must be between 1 and 64')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('batch-size must be positive')
    student_path = os.getenv('TWINKLE_DISTILL_STUDENT_PATH')
    teacher_path = os.getenv('TWINKLE_DISTILL_TEACHER_PATH')
    if args.method != 'opsd' and not student_path:
        parser.error('Set TWINKLE_DISTILL_STUDENT_PATH to the local Qwen3.5-0.8B directory.')
    if args.method != 'opsd' and not teacher_path:
        parser.error('Set TWINKLE_DISTILL_TEACHER_PATH to the local Qwen3.5-2B directory.')
    train(args.method, args.steps, args.eval_size, args.seed, args.model, student_path, teacher_path,
          batch_size=args.batch_size)
