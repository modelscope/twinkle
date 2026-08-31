# Copyright (c) ModelScope Contributors. All rights reserved.
"""Held-out evaluation for agentic RSI: pass rate on tasks the trainer never saw.

Episodes are built and scored by :mod:`episode`, the same module ``challenge.py``
takes ``solver_harness`` from, so a number reported here is measured against the
opening a task's n_pass was measured against -- an eval that constructed episodes
differently would measure a different agent.

What it adds on top of collection is only what collection does not need: several
attempts per task (a single attempt at temperature 1 is a coin flip, not a rate),
no optimizer, and weights read off disk rather than held by a live trainer.

Weights are named by ``--model-id`` and nothing else. ``train.py`` trains every
parameter and saves a whole model, so the trained side of a comparison is a
checkpoint directory in exactly the place the base model's name goes.

Usage::

    # baseline
    python cookbook/rsi/agentic/eval.py --tasks output/.../eval_tasks.jsonl \\
        --label base --out output/.../eval_base.jsonl

    # after training
    python cookbook/rsi/agentic/eval.py --tasks output/.../eval_tasks.jsonl \\
        --model-id output/rsi_agentic/<tag>/ckpt/model --label trained \\
        --out output/.../eval_trained.jsonl

Both runs must use the same ``--tasks``, ``--rollouts-per-task`` and sampling
parameters, or the comparison is not one.
"""
import argparse
import json
import os
import shutil
import statistics
import sys

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episode import (SandboxConfig, boot_episodes, load_tasks,  # noqa: E402,I100,I202
                     score_episodes)

logger = get_logger()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tasks', required=True, help='task jsonl (challenge.py or structured)')
    p.add_argument('--model-id', default='ms://Qwen/Qwen3-4B',
                   help='base model name, or a checkpoint directory saved by train.py')
    p.add_argument('--label', default='eval', help='name for this measurement in the log')
    p.add_argument('--sampler-gpus', type=int, default=4)
    p.add_argument('--max-model-len', type=int, default=32768)

    p.add_argument('--rollouts-per-task', type=int, default=4,
                   help='attempts per task; the pass rate is over these')
    p.add_argument('--episodes-per-wave', type=int, default=16,
                   help='sandboxes alive at once; keep at or below RSI_ENV_CONCURRENCY')
    p.add_argument('--max-turns', type=int, default=20)
    # Both have to equal what challenge.py built the tasks under, and neither was
    # reachable from the command line before: the rollout was constructed with the
    # class default for the first and with --max-model-len for the second, while
    # challenge.py passes --stop-after-stuck-turns and leaves the token cap unset.
    # The stuck cutoff is the one that bites -- it ended 50 to 88 of each
    # iteration's ~550 attempts -- so an eval that leaves it at 0 measures an agent
    # that is allowed to repeat itself forever, against tasks whose n_pass was
    # measured on an agent that was not.
    p.add_argument('--stop-after-stuck-turns', type=int, default=2,
                   help="consecutive no-progress turns that end the tool phase; "
                        "challenge.py's default is 2, 0 disables the cutoff")
    p.add_argument('--max-trajectory-tokens', type=int, default=0,
                   help='cap on the whole trajectory; 0 leaves it unset, which is '
                        'what challenge.py does')
    # Has to equal the challenger's --solver-max-tokens and --propose-max-tokens.
    # An eval that gives the model less room than the run that built the tasks is
    # measuring the budget, not the model: at 4096, 15 of 50 attempts ended on
    # stop_reason=length with an untouched workspace.
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.95)

    p.add_argument('--out', default='', help='per-episode results jsonl')
    # The per-episode row says how an attempt ended but not what it did, and a
    # rate of 0 has two very different causes that only the conversation tells
    # apart: the model worked and got it wrong, or it answered in prose and never
    # touched a tool. 71 of 96 attempts in one run ended within two turns, which
    # is unreadable without this.
    p.add_argument('--dump-messages', default='',
                   help='jsonl of the full conversation per episode, for reading attempts')
    p.add_argument('--keep-workspaces', action='store_true')
    return p.parse_args()


def build_sampler(args):
    twinkle.initialize(
        mode='ray', nproc_per_node=args.sampler_gpus, lazy_collect=False,
        groups=[DeviceGroup(name='sampler', ranks=list(range(args.sampler_gpus)),
                            device_type='GPU')])
    sampler = vLLMSampler(
        model_id=args.model_id,
        engine_args={'gpu_memory_utilization': 0.8,
                     'max_model_len': args.max_model_len},
        device_mesh=DeviceMesh.from_sizes(world_size=args.sampler_gpus,
                                          dp_size=args.sampler_gpus),
        remote_group='sampler',
    )
    sampler.set_template('Template', model_id=args.model_id, enable_thinking=True,
                         max_length=args.max_model_len)
    return sampler


def main():
    args = parse_args()
    tasks = load_tasks(args.tasks)
    cfg = SandboxConfig.from_env()
    logger.info(f'[eval:{args.label}] {len(tasks)} tasks x {args.rollouts_per_task} '
                f'attempts, weights={args.model_id}')
    logger.info(f'[eval:{args.label}] sandboxes: template={cfg.template} api={cfg.api_url}')

    sampler = build_sampler(args)
    template = Template(args.model_id, max_length=args.max_model_len, enable_thinking=True)
    template.truncation_strategy = 'delete'
    rollout = MultiTurnRollout(
        sampler=sampler,
        template=template,
        sampling_params=SamplingParams(max_tokens=args.max_tokens, num_samples=1, logprobs=1,
                                       temperature=args.temperature, top_p=args.top_p),
        max_turns=args.max_turns,
        stop_after_stuck_turns=args.stop_after_stuck_turns,
        max_trajectory_tokens=args.max_trajectory_tokens or None,
    )

    # One flat list of attempts, so a wave is a fixed number of sandboxes no
    # matter how the attempts distribute over tasks.
    attempts = [(task, rep) for task in tasks for rep in range(args.rollouts_per_task)]
    results = []
    n_boot_failed = 0
    scratch = os.path.join('output', 'rsi_agentic', f'_eval_{args.label}')
    msg_dump = (open(args.dump_messages, 'w', encoding='utf-8')
                if args.dump_messages else None)

    for start in range(0, len(attempts), args.episodes_per_wave):
        wave = attempts[start:start + args.episodes_per_wave]
        wave_tasks = [task for task, _ in wave]
        try:
            episodes = boot_episodes(wave_tasks, cfg)
        except Exception as e:  # noqa
            # Reported, never silently dropped: an eval that quietly measures
            # fewer episodes than it claims is worse than one that admits a gap.
            n_boot_failed += len(wave)
            logger.warning(f'[eval:{args.label}] wave at {start} failed to boot: {e}')
            continue
        harnesses = [ep[0] for ep in episodes]
        envs = [ep[1] for ep in episodes]
        tool_managers = [ep[2] for ep in episodes]
        trajectories = [ep[3] for ep in episodes]
        wave_dir = os.path.join(scratch, f'wave{start:04d}')
        try:
            outs = rollout(trajectories, harness=harnesses, tool_manager=tool_managers)
            rewards = score_episodes(wave_tasks, envs, outs, wave_dir, cfg)
        finally:
            for env in envs:
                env.close()
            if not args.keep_workspaces:
                shutil.rmtree(wave_dir, ignore_errors=True)

        for (task, rep), out, reward in zip(wave, outs, rewards):
            labels = out.get('labels') or []
            if msg_dump is not None:
                msg_dump.write(json.dumps({
                    'id': task.get('id'),
                    'rep': rep,
                    'reward': reward,
                    'turns': int(out.get('turns') or 0),
                    'stop_reason': out.get('stop_reason'),
                    'query': task.get('query'),
                    'messages': out.get('messages') or [],
                }, ensure_ascii=False, default=str) + '\n')
                msg_dump.flush()
            results.append({
                'id': task.get('id'),
                'rep': rep,
                'reward': reward,
                'turns': int(out.get('turns') or 0),
                'stop_reason': out.get('stop_reason'),
                'truncated': bool(out.get('truncated')),
                'completion_tokens': sum(1 for label in labels if label != -100),
            })
        done = len(results)
        rate = sum(r['reward'] for r in results) / done if done else 0.0
        logger.info(f'[eval:{args.label}] {done}/{len(attempts)} episodes, '
                    f'mean reward so far {rate:.3f}')

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        logger.info(f'[eval:{args.label}] wrote {len(results)} episodes -> {args.out}')
    if msg_dump is not None:
        msg_dump.close()
        logger.info(f'[eval:{args.label}] conversations -> {args.dump_messages}')

    report(args, tasks, results, n_boot_failed)


def report(args, tasks, results, n_boot_failed):
    """Print what the run measured, including what it failed to measure."""
    if not results:
        logger.warning(f'[eval:{args.label}] no episodes completed; nothing to report')
        return
    per_task = {}
    for row in results:
        per_task.setdefault(row['id'], []).append(row['reward'])

    rewards = [row['reward'] for row in results]
    mean_reward = statistics.fmean(rewards)
    task_rates = [statistics.fmean(v) for v in per_task.values()]
    solved_always = sum(1 for r in task_rates if r >= 1.0)
    solved_never = sum(1 for r in task_rates if r <= 0.0)
    turns = [row['turns'] for row in results]
    stops = {}
    for row in results:
        stops[row['stop_reason']] = stops.get(row['stop_reason'], 0) + 1

    logger.info(
        f'[eval:{args.label}] === {len(results)} episodes over {len(per_task)} tasks '
        f'({args.rollouts_per_task} attempts each) ===')
    logger.info(f'[eval:{args.label}] pass rate (mean reward)   : {mean_reward:.4f}')
    logger.info(f'[eval:{args.label}] per-task rate mean/median : '
                f'{statistics.fmean(task_rates):.4f} / {statistics.median(task_rates):.4f}')
    logger.info(f'[eval:{args.label}] tasks always/never solved : '
                f'{solved_always}/{solved_never} of {len(per_task)}')
    logger.info(f'[eval:{args.label}] turns mean/max            : '
                f'{statistics.fmean(turns):.1f} / {max(turns)}')
    logger.info(f'[eval:{args.label}] truncated episodes        : '
                f'{sum(1 for r in results if r["truncated"])}')
    logger.info(f'[eval:{args.label}] stop reasons              : {stops}')
    if n_boot_failed:
        logger.warning(f'[eval:{args.label}] {n_boot_failed} episodes never ran '
                       f'(sandbox boot failed) and are excluded from every number above')
    if len(tasks) != len(per_task):
        logger.warning(f'[eval:{args.label}] {len(tasks) - len(per_task)} of {len(tasks)} tasks '
                       f'produced no episode at all')


if __name__ == '__main__':
    main()
