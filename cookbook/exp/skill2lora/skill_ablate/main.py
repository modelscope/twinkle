# Copyright (c) ModelScope Contributors. All rights reserved.
"""Entry point: run one ablation experiment by name (E1..E12) or explicit knobs.

Usage:
    python -m skill_ablate.main --exp E5 --seam-parquet-dir /root/data/seam \
        --output-dir output.ablate12/E5_rl_ab_off_pitfall

Defaults mirror train_skill_v2._build_args so reused v2 primitives behave identically; only
the ablation-specific knobs are added (--exp / --max-updates / --eval-every-updates /
--improve-skill-temperature / --skill-char-limit / --pool-max / --rubric-global-dir).
"""
import argparse
import sys

import train_skill_v2 as v2

from .config import METHODS, STYLES, THINKINGS, get_spec, ExpSpec
from .trainer import run_experiment


def _build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- experiment selection: either --exp E5, or explicit --method/--thinking/--style ---
    p.add_argument('--exp', default='', help='experiment name E1..E12 (fills method/think/style)')
    p.add_argument('--method', choices=METHODS, default=None)
    p.add_argument('--thinking', choices=THINKINGS, default=None)
    p.add_argument('--style', choices=STYLES, default=None)

    # --- data (mirror v2) ---
    p.add_argument('--dataset', choices=('aops', 'math'), default='aops')
    p.add_argument('--n', type=int, default=0)
    p.add_argument('--exclude-data-ids', default='')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--numeric-only', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--eval-size', type=int, default=128)
    p.add_argument('--seam-parquet-dir', type=str, default='')
    p.add_argument('--deepmath-dir', type=str, default='',
                   help='DeepMath-103K parquet dir; when set, overrides --seam-parquet-dir/--dataset '
                        'and uses the difficulty-stratified split (eval/train same level mix).')
    p.add_argument('--min-level', type=int, default=0,
                   help='train-only difficulty floor for DeepMath (eval keeps full-level mix). '
                        '0 = off. E1/E5 audit: level<=5 is all-pass dominated (zero gradient); '
                        'recommended 6.')

    # --- eval口径 (4 rollouts × T=0.5, 与旧臂 E1-E13 同口径, 2026-07-28 拍板回退) ---
    # 曾短暂改为 SEAM val 口径(1×greedy)，为保持与已完成 6 臂可横比而回退；需要 SEAM 口径时
    # 显式传 --eval-rollouts 1 --eval-skill-temperature 0.0。
    p.add_argument('--eval-rollouts', type=int, default=4)
    p.add_argument('--eval-skill-temperature', type=float, default=0.5)

    # --- skill-gen / rollout (mirror v2) ---
    p.add_argument('--chunk-size', type=int, default=16)
    p.add_argument('--n-skills', type=int, default=8)
    p.add_argument('--skill-gen-temperature', type=float, default=1.0)
    p.add_argument('--skill-gen-top-p', type=float, default=1.0)
    p.add_argument('--skill-gen-top-k', type=int, default=-1)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--skill-max-tokens', type=int, default=None,
                   help='default per-experiment: 8192 (think) / 4096 (nothink); an explicit '
                        'value here wins over the experiment default.')
    p.add_argument('--align-mode', choices=('v2', 'seam'), default='v2')
    p.add_argument('--len-budget', type=int, default=None,
                   help='regen skill length target (chars). Default per style (ablation stats '
                        '#9): narrative~1100 / pitfall~300. Used to pick the regen survivor.')

    # --- rubric / regen / distill (mirror v2 + new) ---
    p.add_argument('--passatk-k', type=int, default=8)
    p.add_argument('--passatk-skill-temp', type=float, default=1.0)
    p.add_argument('--passatk-skill-top-p', type=float, default=1.0)
    p.add_argument('--passatk-m', type=int, default=2)
    p.add_argument('--rubric-workers', type=int, default=16)
    p.add_argument('--improve-skill-temperature', type=float, default=0.5,
                   help='temperature for the single first-pass skill in opsd / improve_sft.')
    p.add_argument('--skill-char-limit', type=int, default=4096,
                   help='hard char cap for SFT-seed skills (skill_quality_analysis.md #15-1/#18).')

    # --- GRPO / optim (mirror v2) ---
    p.add_argument('--sft-batch-size', type=int, default=16,
                   help='batch = TRAIN_DP multiple; also the SFT-pool draw size.')
    p.add_argument('--train-micro-batch', type=int, default=0,
                   help='forward_backward micro size; 0 = follow --sft-batch-size. think/8192 '
                        'experiments auto-halve to 8 (fp32 master + 8k-token logits OOM guard); '
                        'gradient is micro-normalized so this is mathematically equivalent.')
    p.add_argument('--ppo-mini-batch-size', type=int, default=0)
    p.add_argument('--grpo-epsilon', type=float, default=0.2)
    p.add_argument('--adv-clip', type=float, default=0.0)
    p.add_argument('--kl-beta', type=float, default=0.001)
    p.add_argument('--lr', type=float, default=1e-6,
                   help='stable 1e-6, no warmup / no decay (ablation spec).')
    p.add_argument('--sft-weight', type=float, default=1.0,
                   help='advantage magnitude for SFT samples (ablation spec: 1.0).')
    p.add_argument('--drop-zero-adv', action='store_true',
                   help='reserved single-point ablation (接口方案 #10): drop zero-advantage '
                        'candidates from RL training batches instead of keeping them in the '
                        'token-mean denominator (SEAM口径). Default off; NOT part of E1-E12.')

    # --- run control (new) ---
    p.add_argument('--max-updates', type=int, default=50,
                   help='stop after this many PARAMETER UPDATES (the "step" unit).')
    p.add_argument('--eval-every-updates', type=int, default=5)
    p.add_argument('--save-every-updates', type=int, default=0,
                   help='save a weights-only checkpoint (<exp>-u<N>) every N updates; 0 = only '
                        'the final save. lr is constant so optimizer state is NOT saved.')
    p.add_argument('--resume-from', default='',
                   help='checkpoint dir name under --output-dir (e.g. E1-final / E1-u50) or an '
                        'absolute path. Loads weights into skill_model, restores updates/chunk '
                        'position from train_state.json (falls back to DONE.json), appends to '
                        'the record files, and bypasses the DONE.json skip guard. Raise '
                        '--max-updates beyond the restored count to actually continue.')
    p.add_argument('--pool-max', type=int, default=2048,
                   help='SFT pool per-queue cap (drop oldest; bounds majority backlog).')
    p.add_argument('--rubric-global-dir', default='',
                   help="dir for the cross-experiment global rubric cache "
                        "(default: parent of --output-dir).")

    # --- output / logging ---
    p.add_argument('--output-dir', default='./output.ablate12/exp')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--force', action='store_true',
                   help='rerun even if <output-dir>/DONE.json marks this experiment complete.')
    p.add_argument('--swanlab-project', default='twinkle')

    args = p.parse_args(argv)

    # resolve the experiment spec
    if args.exp:
        spec = get_spec(args.exp)
    else:
        if not (args.method and args.thinking and args.style):
            p.error('provide --exp E5, or all of --method/--thinking/--style')
        spec = ExpSpec(name='Ex', method=args.method, thinking=args.thinking, style=args.style)

    # sanity: Ray dp rule — sft/eval batch must divide TRAIN_DP
    if args.sft_batch_size % v2.TRAIN_DP != 0:
        p.error(f'--sft-batch-size ({args.sft_batch_size}) must be a multiple of TRAIN_DP ({v2.TRAIN_DP})')
    if args.train_micro_batch and args.train_micro_batch % v2.TRAIN_DP != 0:
        p.error(f'--train-micro-batch ({args.train_micro_batch}) must be a multiple of TRAIN_DP ({v2.TRAIN_DP})')
    if args.chunk_size < 1:
        p.error('--chunk-size must be >= 1')
    args.rubric_global_dir = args.rubric_global_dir or None
    return args, spec


def main(argv=None):
    args, spec = _build_args(argv)
    sys.stderr.write(f'[ablate] running {spec.name}: view={spec.view} method={spec.method} '
                     f'thinking={spec.thinking} style={spec.style} loss={spec.loss} '
                     f'smt={spec.skill_max_tokens} -> {args.output_dir}\n')
    run_experiment(args, spec)


if __name__ == '__main__':
    main()
