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
import dataclasses
import sys

import code_task
import train_skill_v2 as v2

from .config import METHODS, STYLES, TASKS, THINKINGS, get_spec, ExpSpec
from .trainer import run_experiment


def _build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- experiment selection: either --exp E5, or explicit --method/--thinking/--style ---
    p.add_argument('--exp', default='', help='experiment name E1..E12 (fills method/think/style)')
    p.add_argument('--method', choices=METHODS, default=None)
    p.add_argument('--thinking', choices=THINKINGS, default=None)
    p.add_argument('--executor-thinking', choices=THINKINGS, default=None,
                   help="executor(base_sampler) 的 thinking；默认取实验自己的 spec。'off' 是 "
                        'E19/E20 的核心变量：think 的 executor 在 BigCodeBench 上 34-50% 的 '
                        'rollout 陷入字面死循环撞满预算，关掉后截断归零、裸解与 rubric 增量都更高。')
    p.add_argument('--style', choices=STYLES, default=None)

    # --- data (mirror v2) ---
    p.add_argument('--dataset', choices=('aops', 'math'), default='aops')
    p.add_argument('--task', choices=TASKS, default=None,
                   help="task family; default = the experiment's own spec.task. 'code' switches "
                        'the whole pipeline to BigCodeBench: executor / skill-gen / rubric prompts, '
                        'unit-test judging instead of \\boxed{} matching, and --bcb-parquet as the '
                        'data source (--deepmath-dir / --seam-parquet-dir / --dataset are ignored).')
    p.add_argument('--bcb-parquet', default=code_task.DEFAULT_PARQUET,
                   help='BigCodeBench parquet (task=code only).')
    p.add_argument('--test-workers', type=int, default=24,
                   help='task=code: thread pool for the unit-test subprocesses. Judging one '
                        'rollout starts a python subprocess (1-3s typical), and a chunk needs '
                        'hundreds of them — serial judging costs more wall clock than the GPU '
                        'rollouts themselves.')
    p.add_argument('--test-timeout', type=int, default=60,
                   help='task=code: wall-clock cap per unit-test run (seconds).')
    p.add_argument('--code-selftest', action=argparse.BooleanOptionalAction, default=True,
                   help='task=code: drop tasks whose OWN canonical solution fails their unit '
                        'tests (sandbox-undecidable, not a model failure; ~7.5% measured). '
                        'Result is cached next to the rubric cache and reused across arms.')
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
    p.add_argument('--eval-min-level', type=int, default=0,
                   help='difficulty floor for the EVAL split too (0 = off, full-level mix as in '
                        'E1-E16). Set = --min-level for arms whose readout only carries '
                        'information on problems the bare executor fails (E17): the full mix is '
                        'only ~26%% bare-wrong vs ~43%% at level>=6, so the same eval budget buys '
                        '1.6x the effective sample AND matches the train distribution. Changes '
                        'the eval set -> not comparable to arms run with 0.')

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
    # E14+ 稠密 reward：logp_rl 用 reward_rollouts/temperature 做 executor 采样，找正确伪 GT S；
    # 之后不再 rollout 判 reward，而是算 Δ mean logP_executor(S | problem + skill)。显式传值覆盖 spec。
    p.add_argument('--reward-rollouts', type=int, default=None)
    p.add_argument('--reward-temperature', type=float, default=None)
    # leak 一律不进 reward（用户既定要求，反复强调）：该指标假阳性极高——DeepMath 的 gold 约一半是
    # 单字符，raw leak 率 0.53-0.65 而加 >=2 字符门后只剩 0.05-0.12，5-10 倍虚高；把它当 reward 项会
    # 用噪声主导组内排序，并且 -1.0 量级的离群值会抬高组 std、连带压小其余候选的 advantage。
    # leak 只保留监控口径（_leak_split -> leak/correct_rate、leak/wrong_rate）。默认 0 = 关闭。
    p.add_argument('--logp-leak-penalty', type=float, default=0.0,
                   help='E14/E15 logp reward leak penalty. DEFAULT 0 = OFF: leak is a '
                        'monitoring-only metric by project rule, never a reward term. The '
                        'unparseable-skill floor stays fixed at -1.0 regardless.')
    # --- E16 passrate_hinge reward knobs -------------------------------------------------
    # 标定依据：.tmp_analysis/reward_shape_calib.py（E4 11155 条 rollout + DeepMath r1_solution_1）。
    # 关键发现：本数据集上长度本身无害——P(对|token) 在 5500 以下平在 0.96-0.98，5500-7500 微降到
    # 0.90，7500 以后断崖到 0.226（每个 difficulty 层内同形）；且 GT 参考解比模型正确答案更长
    # （p50 4377 vs 3444，GT p90 9816 已超 8192 预算）。所以“越短越好”在这里是错的，必须给死区。
    p.add_argument('--reward-trunc-penalty', type=float, default=0.12,
                   help='E16 alpha_len: per-rollout length penalty weight. eff *= (1 - this * '
                        'len_pen), len_pen = ((tok - lo) / (budget - lo)) ** pow, 0 below lo. '
                        'Sized so the TOTAL deduction (1+kappa)*(1-eff) stays under one pass_rate '
                        'quantum (1/M): the length signal may break ties but must never override '
                        'pass_rate, which is what eval actually measures. Its absolute size barely '
                        'matters anyway — inside an all-fail group A=(R-mean)/std rescales the '
                        'spread to unit size, so the CURVE SHAPE carries the information.')
    p.add_argument('--reward-trunc-lo', type=int, default=5500,
                   help='E16 length dead zone: rollouts under this many executor tokens are not '
                        'penalized at all. Calibrated: P(correct|tok) is flat 0.96-0.98 below 5500.')
    p.add_argument('--reward-len-pow', type=float, default=2.0,
                   help='E16 length ramp exponent (>1 = convex, marginal penalty grows with '
                        'length, concentrating it in the last ~1300 tokens before the budget).')
    p.add_argument('--reward-loop-penalty', type=float, default=0.04,
                   help='E16 beta_loop: self-revision marker penalty weight. Deliberately small '
                        '— inside a fixed token band ~85%% of the raw marker effect is just the '
                        'mechanical "longer output has more markers" correlation, and the same '
                        'one-pass-quantum budget is shared with the length term.')
    p.add_argument('--reward-loop-lo', type=float, default=2.0,
                   help='E16 marker density (per 1k tokens) below which no loop penalty applies.')
    p.add_argument('--reward-loop-hi', type=float, default=9.0,
                   help='E16 marker density at which the loop penalty saturates at 1.0.')
    p.add_argument('--reward-ineff-kappa', type=float, default=0.10,
                   help='E16 kappa: reward -= this * mean(1 - eff). Keeps FAILING candidates '
                        'separable (a pure product collapses them all to 0 = E4 pathology). Small '
                        'on purpose: in an all-fail group A=(R-mean)/std rescales the spread back '
                        'to unit size anyway, and a large value would reverse pass_rate ordering.')
    p.add_argument('--reward-leak-gate', type=float, default=0.0,
                   help='E16 reward leak penalty. DEFAULT 0 = OFF: leak is a monitoring-only '
                        'metric by project rule, never a reward term (see --logp-leak-penalty).')
    p.add_argument('--base-tok-floor', type=int, default=5000,
                   help='E16 problem filter: only train problems whose baseline (no-skill) greedy '
                        'output exceeds this many tokens (probe: base_tok vs skill lift +0.62, the '
                        'strongest problem-side signal). 0 disables the filter.')
    # --- E17 reflexion 臂 -----------------------------------------------------------------
    p.add_argument('--reflexion-k', type=int, default=24,
                   help='E17 batch alignment: train on EXACTLY this many bare-wrong problems per '
                        'chunk. The chunk is drawn at --chunk-size, bare-solved, and wrong ones '
                        'are taken (with rubric backfill) until K is reached; dynamic '
                        'over-drawing is impossible because resume replays fixed-size pool '
                        'draws. Group count per update must be constant or the step size, the '
                        'noise floor and the online SNR readout all move with the draw. '
                        'DEFAULT 24 matches the 23.46 groups/update E16 actually ran (1173 '
                        'total): cumulative evidence scales as sqrt(N) and E16 only reached '
                        '1.9 sigma on any text-level direction, so a smaller K has no '
                        'discriminating power left. Needs --chunk-size ~5x (bare error rate '
                        '0.329 measured over the FULL level>=6 pool, 527/1600).')
    p.add_argument('--align-mode', choices=('v2', 'seam'), default='v2')
    # --- E18 rejection_sft 臂 ---------------------------------------------------------------
    p.add_argument('--e18-accumulate', type=int, default=16,
                   help='E18: fire one SFT update only after this many accepted (rejection-'
                        'sampled) skills have accumulated in the pool. Winners are also '
                        'appended to <output-dir>/e18_sft_dataset.jsonl for offline reuse. '
                        'DEFAULT 16 (2026-07-30 拍板) = one sft_batch_size, so a chunk of 32 '
                        '(bare error rate 0.329 -> ~10 accepted) fires roughly every other '
                        'chunk and --max-updates 50 is reachable; 128 would need ~15 chunks '
                        'per update.')
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
    # 对初始策略的锚。漂移分析（.tmp_analysis/why_no_correction.py 等）显示：“executor 能干净收束”
    # 是初始 skill 分布自带的脆弱属性，reward 里能反对它被磨耗的可迁移成分只有 11%，因此
    # 锚本身就是一个直接对症的旋钮（选择压力只能在组内排序，锚才能提供恢复力）。
    # 2026-07-29 拍板：默认 0.001 -> 0.01。⚠️ 注意锚是无方向的刹车，它同等抵制那个方向
    # 正确的 +0.078 收束签名比较信号；取值未经标定（既往臂全部跑在 0.001，无可用对照）。
    p.add_argument('--kl-beta', type=float, default=0.01,
                   help='KL anchor to the reference (initial, never-synced) policy. Raised from '
                        '0.001 to 0.01 on 2026-07-29 to oppose the intrinsic drift that erodes '
                        'executor termination. Recorded in the config fingerprint. Note: the '
                        'anchor is undirected -- it also brakes the (weak) useful gradient.')
    p.add_argument('--run-tag', default='',
                   help='suffix for the swanlab experiment name, to tell apart variants that share '
                        'the same ExpSpec (e.g. RUN_TAG=kl01 for the --kl-beta 0.01 arm).')
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
    # 显式 --task 覆盖 spec（ExpSpec 是 frozen dataclass）；用于临时把某个数学臂放到 code 上跑。
    if args.task and args.task != spec.task:
        sys.stderr.write(f'[ablate] WARNING: --task {args.task} overrides {spec.name} '
                         f'spec.task={spec.task}\n')
        spec = dataclasses.replace(spec, task=args.task)
    args.task = spec.task
    # executor thinking 同理（frozen dataclass -> replace）。v2.build_* 读 args.executor_thinking。
    if args.executor_thinking and args.executor_thinking != spec.executor_thinking:
        sys.stderr.write(f'[ablate] WARNING: --executor-thinking {args.executor_thinking} '
                         f'overrides {spec.name} spec.executor_thinking='
                         f'{spec.executor_thinking}\n')
        spec = dataclasses.replace(spec, executor_thinking=args.executor_thinking)
    args.executor_thinking = spec.executor_thinking

    # sanity: Ray dp rule — sft/eval batch must divide TRAIN_DP
    if args.sft_batch_size % v2.TRAIN_DP != 0:
        p.error(f'--sft-batch-size ({args.sft_batch_size}) must be a multiple of TRAIN_DP ({v2.TRAIN_DP})')
    if args.train_micro_batch and args.train_micro_batch % v2.TRAIN_DP != 0:
        p.error(f'--train-micro-batch ({args.train_micro_batch}) must be a multiple of TRAIN_DP ({v2.TRAIN_DP})')
    if args.chunk_size < 1:
        p.error('--chunk-size must be >= 1')
    if spec.method == 'reflexion':
        if args.reflexion_k < 1:
            p.error('--reflexion-k must be >= 1')
        if args.reflexion_k > args.chunk_size:
            p.error(f'--reflexion-k ({args.reflexion_k}) > --chunk-size ({args.chunk_size}): the '
                    f'aligned batch can never be filled')
        # 2-sigma 余量检查：错题数 ~ Binom(chunk, p_wrong)。余量不够时组数会随 chunk 抽样抖，
        # 而组数恒定是本臂的硬要求。⭐ math: 0.329 = E16 全量题池实测（527/1600）；切勿用
        # 0.446，那是 base_tok>5000 筛选后子集的错误率（偏难），而本臂 floor=0。
        # ⭐ code: 0.56 —— 不是 probe 的 0.715。probe 的 pass 0.285 是 nothink + max_tokens 4096
        # 的读数；本臂 executor 是 think=on + 8192，同一批题实测裸错率只有 18/32=0.5625
        # （2026-07-31 dry run：训练侧 10/16、eval 侧 8/16）。用 0.715 会把 chunk 需求算小
        # 一半：P(48 道里凑不满 24) 在 0.715 下是 0.000，在 0.5625 下是 0.154。
        # ⭐ executor nothink（E19/E20）另算：code nothink 探针实测 pass 0.378 -> p_wrong 0.622；
        #   math nothink 实测 0.31（首个 E19 run 的 c0：64 道里 20 道错，baseline acc≈0.69），
        #   与 think 的 0.329 基本相同 —— 曾按 0.85 保守估是错的，会把 chunk 需求算小一半。
        if spec.executor_thinking == 'off':
            _p_wrong = 0.622 if spec.task == 'code' else 0.31
        else:
            _p_wrong = 0.5625 if spec.task == 'code' else 0.329
        _mu = args.chunk_size * _p_wrong
        _sd = (args.chunk_size * _p_wrong * (1 - _p_wrong)) ** 0.5
        if args.reflexion_k > _mu - 2 * _sd:
            sys.stderr.write(
                f'[ablate] WARNING: --chunk-size {args.chunk_size} gives {_mu:.1f}+-{_sd:.1f} '
                f'wrong problems, less than 2 sigma of headroom over --reflexion-k '
                f'{args.reflexion_k}; expect signal/k_short > 0 and a drifting group count. '
                f'Use --chunk-size >= {int((args.reflexion_k + 2 * _sd) / _p_wrong) + 1}.\n')
    if spec.method == 'rejection_sft':
        if args.e18_accumulate < 1:
            p.error('--e18-accumulate must be >= 1')
        # 池 batch 整体交给 _train_batch，drop_last 到 TRAIN_DP 倍数会静默丢尾部真样本；
        # 强制倍数关系把丢样本量钉在 0。
        if args.e18_accumulate % v2.TRAIN_DP != 0:
            p.error(f'--e18-accumulate ({args.e18_accumulate}) must be a multiple of '
                    f'TRAIN_DP ({v2.TRAIN_DP})')
    args.rubric_global_dir = args.rubric_global_dir or None
    return args, spec


def main(argv=None):
    args, spec = _build_args(argv)
    sys.stderr.write(f'[ablate] running {spec.name}: task={spec.task} view={spec.view} '
                     f'method={spec.method} thinking={spec.thinking} style={spec.style} '
                     f'executor_thinking={spec.executor_thinking} '
                     f'loss={spec.loss} smt={spec.skill_max_tokens} -> {args.output_dir}\n')
    run_experiment(args, spec)


if __name__ == '__main__':
    main()
