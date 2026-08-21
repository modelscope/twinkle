# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI entry point — run one stage on its own, or the whole chain.

Three stages, one model (Qwen3-4B) playing both roles:

    1 prepare    cookbook/rsi/prepare.py        (CPU)      raw    -> seeds
    2 challenge  cookbook/rsi/code/challenge.py (ray+GPU)  [seeds]-> flows + tests
    3 rl         cookbook/rsi/rl.py             (ray+GPU)  flows  -> trained model

``prepare`` only cleans a dataset into seed material and is optional: the
challenger invents problems from nothing when given no seeds. ``challenge`` asks
the model for a problem plus a reference solution, RUNS that solution to get the
ground truth, turns it into asserts, then keeps only the problems the same model
solves sometimes-but-not-always -- a group that all passes or all fails gives
GRPO a zero gradient. ``rl`` trains on what survived, in ``grpo`` mode (feed the
sandbox error back as a tool turn and let it continue) or ``opsd`` (a teacher
that was shown the reference solution distills the student).

Why this launches SUBPROCESSES instead of importing and calling:
  * ``rl`` runs ``CLI.from_args()`` and ``swanlab.init()`` at IMPORT time, so
    merely importing it would parse this launcher's argv and start a run;
  * the stages need different ray topologies (sampler-only vs trainer+sampler)
    and cannot share one ray init in-process.

This launcher invents no parameters: it wires each stage's default output into
the next stage's input and forwards any extra flags straight through.

Examples
--------
    # clean a dataset into seeds (optional)
    python cookbook/rsi/run_rsi.py --step prepare --raw data/raw.jsonl

    # invent problems: from nothing, or seeded by the file above
    python cookbook/rsi/run_rsi.py --step challenge --keep-target 500
    python cookbook/rsi/run_rsi.py --step challenge --seeds output/rsi/subset.jsonl

    # train; twinkle CLI knobs are forwarded as extras
    python cookbook/rsi/run_rsi.py --step rl --mode grpo \
        --model-id ms://Qwen/Qwen3-4B --model-gpus 4 --sampler-gpus 4

    # whole chain with default paths (each stage still a fresh process)
    python cookbook/rsi/run_rsi.py --step all --raw data/raw.jsonl
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Default paths chain one stage into the next. They mirror the defaults each
# stage ships with, kept here so --step all wires up with no flags.
DEFAULT_SEEDS = 'output/rsi/subset.jsonl'            # prepare --output / challenge --seed-file
DEFAULT_FLOWS = 'output/rsi/challenge_flows.jsonl'   # challenge --out-flows / rl RSI_STD_FLOWS
DEFAULT_TESTS = 'output/rsi/challenge_tests.jsonl'   # challenge --out-tests / rl RSI_TESTS

SCRIPTS = {
    'prepare': os.path.join(HERE, 'prepare.py'),
    'challenge': os.path.join(HERE, 'code', 'challenge.py'),
    'rl': os.path.join(HERE, 'rl.py'),
}
ORDER = ['prepare', 'challenge', 'rl']


def _run(script: str, argv: list, env: dict) -> None:
    """Run ``python script argv...`` as a child process, streaming its output.

    Raises on non-zero exit so --step all stops at the first failing stage
    instead of silently feeding a broken artifact into the next one.
    """
    cmd = [sys.executable, script] + argv
    print(f'\n[run_rsi] $ {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, env=env, check=True)


def _argv_for(step: str, a: argparse.Namespace, extra: list) -> tuple:
    """Build (argv, env) for one stage. ``extra`` is forwarded verbatim so each
    stage's own flags (challenger knobs, twinkle CLI knobs, ...) still work."""
    env = dict(os.environ)
    if step == 'prepare':
        if not a.raw:
            raise SystemExit('[run_rsi] --step prepare 需要 --raw 指向原始数据源')
        argv = ['--input', a.raw, '--output', a.seeds, '--num-proc', str(a.num_proc)]
        if a.dropped_log:
            argv += ['--dropped-log', a.dropped_log]
        return argv + extra, env
    if step == 'challenge':
        argv = ['--out-flows', a.flows, '--out-tests', a.tests]
        if a.seeds_given:
            argv += ['--seed-file', a.seeds]
        if a.keep_target:
            argv += ['--keep-target', str(a.keep_target)]
        return argv + extra, env
    if step == 'rl':
        # rl reads flows/tests and the solver mode from env vars; the
        # model/infra/rl knobs arrive through `extra` (twinkle CLI).
        env['RSI_STD_FLOWS'] = a.flows
        env['RSI_TESTS'] = a.tests
        env['RSI_SOLVER_MODE'] = a.mode
        return list(extra), env
    raise SystemExit(f'[run_rsi] 未知 step: {step}')


def main():
    parser = argparse.ArgumentParser(
        description='RSI launcher — run one stage (validate) or the whole chain.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--step', required=True, choices=ORDER + ['all'],
                        help='Which stage to run (or "all" for prepare->challenge->rl).')
    parser.add_argument('--raw', default='', help='Raw data source for prepare (local path or ms:// id).')
    parser.add_argument('--seeds', default=DEFAULT_SEEDS,
                        help='prepare output / challenge seed pool. Passed to challenge only '
                             'when given explicitly or when the chain produced it.')
    parser.add_argument('--flows', default=DEFAULT_FLOWS, help='challenge output flows / rl input.')
    parser.add_argument('--tests', default=DEFAULT_TESTS, help='challenge output tests / rl code asserts.')
    parser.add_argument('--keep-target', type=int, default=0,
                        help="How many problems challenge should keep (0 = the script's own default).")
    parser.add_argument('--mode', default='grpo', choices=['grpo', 'opsd'],
                        help='rl solver mode (RSI_SOLVER_MODE).')
    parser.add_argument('--num-proc', type=int, default=4, help='Parallel workers for prepare.')
    parser.add_argument('--dropped-log', default='', help='Optional dropped-row log for prepare.')
    a, extra = parser.parse_known_args()
    # A seed pool is only handed to the challenger when it was asked for: passing
    # the default path silently would turn "invent from scratch" into "vary
    # whatever happens to be left in output/rsi/ from an earlier run".
    a.seeds_given = '--seeds' in sys.argv

    if a.step == 'all':
        if extra:
            # For 'all' the extras are ambiguous (which stage?); refuse rather than
            # forward a flag to a stage that does not accept it.
            raise SystemExit(f'[run_rsi] --step all 不接受透传参数 {extra}；请逐个 --step 跑并各自带参数')
        if not a.raw:
            raise SystemExit('[run_rsi] --step all 需要 --raw 指向原始数据源')
        # prepare ran, so its output exists and the challenger should use it.
        a.seeds_given = True
        for step in ORDER:
            argv, env = _argv_for(step, a, [])
            _run(SCRIPTS[step], argv, env)
        print('\n[run_rsi] all stages done.', flush=True)
        return

    argv, env = _argv_for(a.step, a, extra)
    _run(SCRIPTS[a.step], argv, env)


if __name__ == '__main__':
    main()
