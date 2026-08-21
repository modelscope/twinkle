# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI self-play entry point — the challenger/solver data loop, no prepare/refine.

Two stages live in ``twinkle_agentic.rsi`` (one model, Qwen3-4B, plays both roles):

    1 challenge  twinkle_agentic.rsi.rsi_challenge  (ray+GPU)  [seed] -> flows + tests
    2 rl         twinkle_agentic.rsi.rsi_rl         (ray+GPU)  flows  -> trained model

Stage 1 asks the model to invent (or vary a seed into) self-contained Python
problems, RUNS its reference solution to get the ground truth, turns that into
asserts, then keeps only the problems the same model solves sometimes-but-not-
always (0 < pass < N). Stage 3 trains on those with ``RSI_SOLVER_MODE``:
``grpo`` (feed the sandbox error back as a tool turn and continue) or ``opsd``
(a teacher that saw the reference solution distills the student).

Why this launches SUBPROCESSES instead of importing and calling (same reason as
run_rsi.py):
  * ``rsi_rl`` runs ``CLI.from_args()`` and ``swanlab.init()`` at IMPORT time, so
    merely importing it would parse this launcher's argv and start a run.
  * the two stages need different ray topologies (sampler-only vs trainer+sampler)
    and cannot share one ray init in-process.

This launcher invents no parameters: it wires stage 1's default outputs into
stage 2's inputs (through each script's own env vars) and forwards any extra
flags straight through to the selected stage.

Examples
--------
Run one stage at a time (extra flags after the known ones are forwarded):

    # from scratch (no seed dataset)
    python cookbook/rsi/run_rsi_selfplay.py --step challenge
    # from a seed dataset (challenger writes variants of its queries)
    python cookbook/rsi/run_rsi_selfplay.py --step challenge --seed data/seed.jsonl
    # train — grpo (default) or opsd; twinkle CLI knobs are forwarded as extras
    python cookbook/rsi/run_rsi_selfplay.py --step rl --mode grpo \
        --model.model_id ms://Qwen/Qwen3-4B --infra.model_gpus 4 --infra.sampler_gpus 4

Run the whole chain with default paths (each stage still a fresh process):

    python cookbook/rsi/run_rsi_selfplay.py --step all --mode grpo
"""
import argparse
import os
import subprocess
import sys

# Default paths chain stage 1 into stage 2. These mirror the defaults baked into
# each stage's own env-var config, kept here so --step all wires up with no flags.
DEFAULT_FLOWS = 'output/rsi/challenge_flows.jsonl'   # rsi_challenge RSI_CH_OUT_FLOWS / rsi_rl RSI_STD_FLOWS
DEFAULT_TESTS = 'output/rsi/challenge_tests.jsonl'   # rsi_challenge RSI_CH_OUT_TESTS / rsi_rl RSI_TESTS

MODULES = {
    'challenge': 'twinkle_agentic.rsi.rsi_challenge',
    'rl': 'twinkle_agentic.rsi.rsi_rl',
}
ORDER = ['challenge', 'rl']


def _run(module: str, argv: list, env: dict) -> None:
    """Run ``python -m module argv...`` as a child process, streaming its output.

    Raises on non-zero exit so --step all stops at the first failing stage
    instead of silently feeding a broken artifact into the next stage.
    """
    cmd = [sys.executable, '-m', module] + argv
    print(f'\n[run_rsi_selfplay] $ {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, env=env, check=True)


def _argv_for(step: str, a: argparse.Namespace, extra: list) -> tuple:
    """Build (argv, env) for one stage. ``extra`` is forwarded verbatim so each
    stage's own flags (twinkle CLI knobs for rl) still work."""
    env = dict(os.environ)
    if step == 'challenge':
        # rsi_challenge is configured purely through RSI_CH_* env vars (no CLI).
        env['RSI_CH_OUT_FLOWS'] = a.flows
        env['RSI_CH_OUT_TESTS'] = a.tests
        if a.seed:
            env['RSI_CH_SEED'] = a.seed
        return list(extra), env
    if step == 'rl':
        # rsi_rl reads flows/tests and the solver mode from env vars; the
        # model/infra/rl knobs arrive through `extra` (twinkle CLI).
        env['RSI_STD_FLOWS'] = a.flows
        env['RSI_TESTS'] = a.tests
        env['RSI_SOLVER_MODE'] = a.mode
        return list(extra), env
    raise SystemExit(f'[run_rsi_selfplay] 未知 step: {step}')


def main():
    parser = argparse.ArgumentParser(
        description='RSI self-play launcher — run one stage (validate) or the whole chain.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--step', required=True, choices=ORDER + ['all'],
                        help='Which stage to run (or "all" for challenge->rl).')
    parser.add_argument('--seed', default='',
                        help='Optional seed dataset (jsonl) for challenge; empty = invent from scratch.')
    parser.add_argument('--flows', default=DEFAULT_FLOWS,
                        help='challenge output flows / rl standard-flow input.')
    parser.add_argument('--tests', default=DEFAULT_TESTS,
                        help='challenge output tests / rl code-round asserts input.')
    parser.add_argument('--mode', default='grpo', choices=['grpo', 'opsd'],
                        help='rl solver mode (RSI_SOLVER_MODE).')
    a, extra = parser.parse_known_args()

    if a.step == 'all':
        if extra:
            # For 'all' the extras are ambiguous (which stage?); refuse rather than
            # forward a flag to a stage that does not accept it.
            raise SystemExit(f'[run_rsi_selfplay] --step all 不接受透传参数 {extra}；'
                             '请逐个 --step 跑并各自带参数')
        for step in ORDER:
            argv, env = _argv_for(step, a, [])
            _run(MODULES[step], argv, env)
        print('\n[run_rsi_selfplay] all stages done.', flush=True)
        return

    argv, env = _argv_for(a.step, a, extra)
    _run(MODULES[a.step], argv, env)


if __name__ == '__main__':
    main()
