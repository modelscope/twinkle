"""Build exact reflexion skill RFT data without updating the skill model.

The output ``skill_dataset.jsonl`` uses the same schema as the online trainer's
per-chunk training records, while ``gen_records.jsonl`` keeps the full trace for
inspection. This lets expensive rollout/scoring be run once, then reused for
offline replay experiments.

Launch:
    python cookbook/exp/embedding/build_reflexion_skill_data.py --chunks 100
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import train_reflexion_skill_rft as rft
from twinkle_agentic.verifier import LeakVerifier


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('aops', 'math'), default='aops')
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--numeric-only', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--eval-size', type=int, default=128)
    p.add_argument('--chunks', type=int, default=100)
    p.add_argument('--chunk-size', type=int, default=16)
    p.add_argument('--balance', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--balance-success-frac', type=float, default=0.4)
    p.add_argument('--balance-loop-frac', type=float, default=0.5)
    p.add_argument('--balance-max-draws-mult', type=int, default=8)
    p.add_argument('--n-skills', type=int, default=8)
    p.add_argument('--view-b-frac', type=float, default=0.5)
    p.add_argument('--skill-retries', type=int, default=2)
    p.add_argument('--skill-gen-temperature', type=float, default=1.0)
    p.add_argument('--skill-gen-top-p', type=float, default=1.0)
    p.add_argument('--skill-gen-top-k', type=int, default=-1)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--skill-max-tokens', type=int, default=8192)
    p.add_argument('--leak-workers', type=int, default=16)
    p.add_argument('--rubric-workers', type=int, default=16)
    p.add_argument('--format-in-reward', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--output-dir', default='./output/reflexion_skill_data')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def _init_samplers(args: argparse.Namespace):
    skill_dp = rft.SKILL_SAMPLER_GPUS
    base_dp = rft.BASE_SAMPLER_GPUS
    total_gpus = skill_dp + base_dp
    if total_gpus <= 0:
        raise ValueError('SKILL_SAMPLER_GPUS + BASE_SAMPLER_GPUS must be positive')
    device_groups = [
        rft.DeviceGroup(name='skill_sampler', ranks=list(range(0, skill_dp)), device_type='GPU'),
        rft.DeviceGroup(name='base_sampler', ranks=list(range(skill_dp, total_gpus)), device_type='GPU'),
    ]
    rft.twinkle.initialize(mode='ray', nproc_per_node=total_gpus, groups=device_groups,
                           lazy_collect=False)
    skill_sampler = rft.vLLMSampler(
        model_id=rft.GEN_MODEL_ID,
        engine_args={'gpu_memory_utilization': rft.GEN_GPU_MEM,
                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
        device_mesh=rft.DeviceMesh.from_sizes(world_size=skill_dp, dp_size=skill_dp),
        remote_group='skill_sampler')
    skill_sampler.set_template(rft.Template, model_id=rft.GEN_MODEL_ID,
                               enable_thinking=True, max_length=args.max_model_len)
    base_sampler = rft.vLLMSampler(
        model_id=rft.GEN_MODEL_ID,
        engine_args={'gpu_memory_utilization': rft.GEN_GPU_MEM,
                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
        device_mesh=rft.DeviceMesh.from_sizes(world_size=base_dp, dp_size=base_dp),
        remote_group='base_sampler')
    base_sampler.set_template(rft.Template, model_id=rft.GEN_MODEL_ID,
                              enable_thinking=True, max_length=args.max_model_len)
    return base_sampler, skill_sampler, base_dp, skill_dp


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
        'record_type': 'config', 'mode': 'offline_data_build', 'model': rft.GEN_MODEL_ID,
        'dataset': args.dataset, 'n': len(records), 'eval_n': len(eval_records),
        'seed': args.seed, 'numeric_only': args.numeric_only,
        'raw_loaded': data_stats['raw_loaded'], 'numeric_dropped': data_stats['numeric_dropped'],
        'chunks': args.chunks, 'chunk_size': args.chunk_size, 'n_skills': args.n_skills,
        'view_b_frac': args.view_b_frac, 'balance': args.balance,
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

        for ci in range(args.chunks):
            chunk, balance = rft._draw_chunk(pool, base_sampler, base_dp, args)
            full, summary, groups = rft.process_chunk(
                base_sampler, skill_sampler, leak, chunk, ci, base_dp, skill_dp,
                args, checker, rubric_cache)
            summary['balance'] = balance
            for rec in full:
                _write_jsonl_row(gen_f, rec)
            _write_jsonl_row(gen_f, summary)
            gen_f.flush()
            for row in groups:
                _write_jsonl_row(data_f, {'chunk': ci, **row})
            data_f.flush()
            total_groups += len(groups)
            sys.stderr.write(
                f'[build-rft-data] g{ci}: train={len(groups)} total={total_groups} '
                f'acc={summary["avg_baseline_pass_on_hard"]:.2f}->{summary["avg_withskill_pass"]:.2f} '
                f'lift={summary["avg_lift"]:+.3f}\n')

    sys.stderr.write(f'[build-rft-data] done: train records -> {data_path}; trace -> {gen_path}\n')


if __name__ == '__main__':
    main()
