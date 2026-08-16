# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI pipeline entry point — run any single stage (or the whole chain) so each
step can be validated in isolation.

The four stages live in ``twinkle_agentic.rsi`` and each already has its own CLI:

    1 prepare   twinkle_agentic.rsi.rsi_prepare   (CPU)      raw   -> subset
    2 refine    twinkle_agentic.rsi.rsi_refine    (API)      subset-> flows
    3 rl        twinkle_agentic.rsi.rsi_rl        (ray+GPU)  flows -> executor LoRA
    4 distill   twinkle_agentic.rsi.rsi_distill   (ray+GPU)  dump  -> role LoRA

Why this launches SUBPROCESSES instead of importing and calling:
  * ``rsi_rl`` runs ``CLI.from_args()`` and ``swanlab.init()`` at IMPORT time, so
    merely importing it would parse this launcher's argv and start a run.
  * ``rl`` and ``distill`` need DIFFERENT ray topologies (MultiLora+sampler vs a
    single TransformersModel group); they cannot share one ray init in-process.
Running each stage as its own ``python -m ...`` process side-steps both — and is
exactly what "run each step separately to validate" needs.

This launcher invents no parameters: it only wires the default output of one
stage into the input of the next (reusing each script's own default paths) and
forwards any extra flags straight through to the selected stage.

Examples
--------
Validate one stage at a time (extra flags after the known ones are forwarded):

    python cookbook/rsi/run_rsi.py --step prepare --raw data/raw.jsonl
    python cookbook/rsi/run_rsi.py --step refine --teacher-model qwen3-235b-a22b-instruct-2507
    python cookbook/rsi/run_rsi.py --step rl     --model.model_id ms://Qwen/Qwen3-4B --infra.model_gpus 4
    python cookbook/rsi/run_rsi.py --step distill --dump output/rsi/dump/refine.jsonl --adapter refine

Run the whole chain with default paths (each stage still a fresh process):

    python cookbook/rsi/run_rsi.py --step all --raw data/raw.jsonl
"""
import argparse
import os
import subprocess
import sys

# Default paths chain one stage into the next. These mirror the defaults baked
# into each stage's own CLI, kept here so --step all wires up with no flags.
DEFAULT_SUBSET = 'output/rsi/subset.jsonl'          # rsi_prepare --output
DEFAULT_FLOWS = 'output/rsi/standard_flows.jsonl'   # rsi_refine  --output / rsi_rl RSI_STD_FLOWS
DEFAULT_DUMP = 'output/rsi/dump/refine.jsonl'       # llm_backup LLM_BACKUP_DUMP_PATH / rsi_distill --input

MODULES = {
    'prepare': 'twinkle_agentic.rsi.rsi_prepare',
    'refine': 'twinkle_agentic.rsi.rsi_refine',
    'rl': 'twinkle_agentic.rsi.rsi_rl',
    'distill': 'twinkle_agentic.rsi.rsi_distill',
}
ORDER = ['prepare', 'refine', 'rl', 'distill']


def _run(module: str, argv: list, env: dict) -> None:
    """Run ``python -m module argv...`` as a child process, streaming its output.

    Raises on non-zero exit so --step all stops at the first failing stage
    instead of silently feeding a broken artifact into the next stage.
    """
    cmd = [sys.executable, '-m', module] + argv
    print(f'\n[run_rsi] $ {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, env=env, check=True)


def _argv_for(step: str, a: argparse.Namespace, extra: list) -> tuple:
    """Build (argv, env) for one stage. ``extra`` is forwarded verbatim so each
    stage's own flags (teacher creds, twinkle CLI knobs, ...) still work."""
    env = dict(os.environ)
    if step == 'prepare':
        if not a.raw:
            raise SystemExit('[run_rsi] --step prepare 需要 --raw 指向原始数据源')
        argv = ['--input', a.raw, '--output', a.subset, '--num-proc', str(a.num_proc)]
        if a.dropped_log:
            argv += ['--dropped-log', a.dropped_log]
        return argv + extra, env
    if step == 'refine':
        return ['--input', a.subset, '--output', a.flows] + extra, env
    if step == 'rl':
        # rsi_rl reads the standard-flow path from an env var, not a flag;
        # model/infra/rl knobs arrive through `extra` (twinkle CLI).
        env['RSI_STD_FLOWS'] = a.flows
        return list(extra), env
    if step == 'distill':
        # rsi_distill accepts --input/--adapter and also honours these env vars.
        env['RSI_DUMP_PATH'] = a.dump
        argv = ['--input', a.dump]
        if a.adapter:
            argv += ['--adapter', a.adapter]
        return argv + extra, env
    raise SystemExit(f'[run_rsi] 未知 step: {step}')


def main():
    parser = argparse.ArgumentParser(
        description='RSI pipeline launcher — run one stage (validate) or the whole chain.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--step', required=True, choices=ORDER + ['all'],
                        help='Which stage to run (or "all" for prepare->refine->rl->distill).')
    parser.add_argument('--raw', default='', help='Raw data source for prepare (local path or ms:// id).')
    parser.add_argument('--subset', default=DEFAULT_SUBSET, help='prepare output / refine input.')
    parser.add_argument('--flows', default=DEFAULT_FLOWS, help='refine output / rl standard-flow input.')
    parser.add_argument('--dump', default=os.environ.get('LLM_BACKUP_DUMP_PATH', DEFAULT_DUMP),
                        help='llm_backup dump JSONL / distill input.')
    parser.add_argument('--adapter', default='', help='Adapter name for distill (default: derived from dump name).')
    parser.add_argument('--num-proc', type=int, default=int(os.environ.get('RSI_NUM_PROC', '4')),
                        help='Parallel workers for prepare.')
    parser.add_argument('--dropped-log', default='', help='Optional dropped-row log for prepare.')
    a, extra = parser.parse_known_args()

    if a.step == 'all':
        if extra:
            # For 'all' the extras are ambiguous (which stage?); refuse rather than
            # forward a flag to a stage that does not accept it.
            raise SystemExit(f'[run_rsi] --step all 不接受透传参数 {extra}；请逐个 --step 跑并各自带参数')
        for step in ORDER:
            if step == 'prepare' and not a.raw:
                raise SystemExit('[run_rsi] --step all 需要 --raw 指向原始数据源')
            argv, env = _argv_for(step, a, [])
            _run(MODULES[step], argv, env)
        print('\n[run_rsi] all stages done.', flush=True)
        return

    argv, env = _argv_for(a.step, a, extra)
    _run(MODULES[a.step], argv, env)


if __name__ == '__main__':
    main()
