"""Build exact reflexion skill RFT data without updating the skill model.

The output ``skill_dataset.jsonl`` uses the same schema as the online trainer's
per-chunk training records, while ``gen_records.jsonl`` keeps the full trace for
inspection. This lets expensive rollout/scoring be run once, then reused for
offline replay experiments.

Launch:
    python cookbook/exp/embedding/build_reflexion_skill_data.py \
        --total-problems 3200 --base-success-frac 0.3
"""
import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from cookbook.exp.embedding.eval_gpqa_rag import load_aops, load_math
from twinkle.sampler import vLLMSampler

import twinkle
from twinkle import DeviceGroup, DeviceMesh

import train_reflexion_skill_rft as rft
from twinkle_agentic.verifier import LeakVerifier


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--total-problems', type=int, default=3200,
                   help='Final number of problems selected into generated chunks.')
    p.add_argument('--base-success-frac', type=float, default=0.3,
                   help='Target fraction of selected problems solved by the frozen base.')
    p.add_argument('--output-dir', default='./output/reflexion_skill_data')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--seed', type=int, default=42)

    advanced = p.add_argument_group('advanced knobs, usually leave unchanged')
    advanced.add_argument('--dataset', choices=('aops', 'math'), default='aops')
    advanced.add_argument('--n', type=int, default=0,
                          help='Raw train-pool size; 0 derives it from --total-problems.')
    advanced.add_argument('--numeric-only', action=argparse.BooleanOptionalAction, default=True)
    advanced.add_argument('--eval-size', type=int, default=128)
    advanced.add_argument('--chunk-size', type=int, default=16)
    advanced.add_argument('--balance', action=argparse.BooleanOptionalAction, default=True)
    advanced.add_argument('--balance-loop-frac', type=float, default=0.5)
    advanced.add_argument('--balance-max-draws-mult', type=int, default=8)
    advanced.add_argument('--n-skills', type=int, default=8)
    advanced.add_argument('--view-b-frac', type=float, default=0.5)
    advanced.add_argument('--skill-retries', type=int, default=2)
    advanced.add_argument('--skill-gen-temperature', type=float, default=1.0)
    advanced.add_argument('--skill-gen-top-p', type=float, default=1.0)
    advanced.add_argument('--skill-gen-top-k', type=int, default=-1)
    advanced.add_argument('--max-model-len', type=int, default=16384)
    advanced.add_argument('--max-tokens', type=int, default=8192)
    advanced.add_argument('--skill-max-tokens', type=int, default=8192)
    advanced.add_argument('--leak-workers', type=int, default=16)
    advanced.add_argument('--rubric-workers', type=int, default=16)
    advanced.add_argument('--format-in-reward', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    _resolve_args(args)
    return args


def _resolve_args(args: argparse.Namespace) -> None:
    if args.total_problems <= 0:
        raise ValueError('--total-problems must be positive')
    if args.chunk_size <= 0:
        raise ValueError('--chunk-size must be positive')
    if not 0.0 <= args.base_success_frac <= 1.0:
        raise ValueError('--base-success-frac must be in [0, 1]')
    args.chunks = math.ceil(args.total_problems / args.chunk_size)
    args.balance_success_frac = args.base_success_frac
    if args.n <= 0:
        args.n = max(args.total_problems + args.eval_size,
                     math.ceil(args.total_problems * 1.5))


def _init_samplers(args: argparse.Namespace):
    model = 'ms://Qwen/Qwen3-4B'
    device_groups = [
        DeviceGroup(name='skill_sampler', ranks=list(range(0, 4)), device_type='GPU'),
        DeviceGroup(name='base_sampler', ranks=list(range(4, 8)), device_type='GPU'),
    ]
    twinkle.initialize(mode='ray', nproc_per_node=8, groups=device_groups,
                           lazy_collect=False)
    skill_sampler = vLLMSampler(
        model_id='Qwen/Qwen3-4B',
        engine_args={'gpu_memory_utilization': 0.8,
                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
        device_mesh=DeviceMesh.from_sizes(world_size=4, dp_size=4),
        remote_group='skill_sampler')
    skill_sampler.set_template('Template', model_id=model,
                               enable_thinking=True, max_length=args.max_model_len)
    base_sampler = vLLMSampler(
        model_id=model,
        engine_args={'gpu_memory_utilization': 0.8,
                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
        device_mesh=DeviceMesh.from_sizes(world_size=4, dp_size=4),
        remote_group='base_sampler')
    base_sampler.set_template('Template', model_id=model,
                              enable_thinking=True, max_length=args.max_model_len)
    return base_sampler, skill_sampler, 4, 4


def _write_jsonl_row(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = _build_args()
    records, eval_records, data_stats = rft._load_records(args)
    rft._validate_run_config(args, records)
    os.makedirs(args.output_dir, exist_ok=True)
    data_path = os.path.join(args.output_dir, 'skill_dataset.jsonl')
    gen_path = os.path.join(args.output_dir, 'gen_records.jsonl')
    eval_path = os.path.join(args.output_dir, 'eval_holdout.jsonl')
    for path in (data_path, gen_path, eval_path):
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f'{path} exists; pass --overwrite to replace it')

    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('OPENAI_API_KEY')):
        sys.stderr.write('[build-rft-data] WARNING: no LLM backup env; leak/rubric checks degrade\n')

    base_sampler, skill_sampler, base_dp, skill_dp = _init_samplers(args)
    leak = LeakVerifier(sampler=None, answer_only=True)
    checker = rft._build_rubric_checker()
    pool = rft._ProblemPool(records, args.seed)
    rubric_cache: Dict[str, str] = {}

    cfg = {
        'record_type': 'config', 'mode': 'offline_data_build', 'model': 'Qwen/Qwen3-4B',
        'dataset': args.dataset, 'n': len(records), 'eval_n': len(eval_records),
        'total_problems': args.total_problems, 'seed': args.seed,
        'numeric_only': args.numeric_only,
        'raw_loaded': data_stats['raw_loaded'], 'numeric_dropped': data_stats['numeric_dropped'],
        'chunks': args.chunks, 'chunk_size': args.chunk_size, 'n_skills': args.n_skills,
        'view_b_frac': args.view_b_frac, 'balance': args.balance,
        'base_success_frac': args.base_success_frac,
        'balance_success_frac': args.balance_success_frac,
        'reward': 'greedy_binary(correct)', 'advantage': 'group_relative',
        'format_in_reward': args.format_in_reward, 'started': int(time.time()),
    }
    total_groups = 0
    with open(gen_path, 'w', encoding='utf-8') as gen_f, \
            open(data_path, 'w', encoding='utf-8') as data_f, \
            open(eval_path, 'w', encoding='utf-8') as eval_f:
        for handle in (gen_f, data_f, eval_f):
            _write_jsonl_row(handle, cfg)
        for rec in eval_records:
            _write_jsonl_row(eval_f, {'record_type': 'eval_holdout', **rec})
        eval_f.flush()

        selected = 0
        for ci in range(args.chunks):
            remaining = args.total_problems - selected
            if remaining <= 0:
                break
            original_chunk_size = args.chunk_size
            args.chunk_size = min(original_chunk_size, remaining)
            try:
                chunk, balance = rft._draw_chunk(pool, base_sampler, base_dp, args)
                full, summary, groups = rft.process_chunk(
                    base_sampler, skill_sampler, leak, chunk, ci, base_dp, skill_dp,
                    args, checker, rubric_cache)
            finally:
                args.chunk_size = original_chunk_size
            summary['balance'] = balance
            for rec in full:
                _write_jsonl_row(gen_f, rec)
            _write_jsonl_row(gen_f, summary)
            gen_f.flush()
            for row in groups:
                _write_jsonl_row(data_f, {'chunk': ci, **row})
            data_f.flush()
            total_groups += len(groups)
            selected += len(chunk)
            sys.stderr.write(
                f'[build-rft-data] g{ci}: problems={selected}/{args.total_problems} '
                f'train={len(groups)} total={total_groups} '
                f'acc={summary["avg_baseline_pass_on_hard"]:.2f}->{summary["avg_withskill_pass"]:.2f} '
                f'lift={summary["avg_lift"]:+.3f}\n')

    sys.stderr.write(f'[build-rft-data] done: train records -> {data_path}; trace -> {gen_path}\n')


if __name__ == '__main__':
    main()
