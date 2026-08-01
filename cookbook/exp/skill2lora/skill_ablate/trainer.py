# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unified training loop for one ablation experiment.

step unit = PARAMETER UPDATE (skill_quality_analysis.md #2): a chunk may yield 0/1/more
updates; we stop at ``--max-updates`` and eval every ``--eval-every-updates`` updates. All
eval is query-only via v2 ``run_greedy_eval`` (T=0.5 × 4 rollouts, no rubric) — the
knowledge-transfer probe for view A. Records/log schema mirror v2 (gen/eval/train_log jsonl,
per-problem rollout rows land in gen_records.jsonl).

swanlab step axis: ONE global axis = chunk index for train AND eval curves (eval also logs
``eval/updates_done`` so the update count is recoverable); mixing chunk/update axes in one
experiment made curves incomparable.
"""
import json
import os
import sys
import time
from typing import Any

import train_skill_v2 as v2

from .config import ExpSpec
from .data import load_deepmath_records
from .data_code import load_code_records
from .eval_reflexion import run_reflexion_eval
from .methods import MethodContext, build_method
from .pool import SamplePool
from .rubric_cache import build_rubric_cache

try:
    import swanlab
except ImportError:
    swanlab = None


def _rubric_scope(method: str) -> str:
    """Bare-problem lines (rl/sft) share a GLOBAL cache; with-skill lines (opsd/improve) use
    a per-experiment LOCAL cache. bnpo needs no rubric."""
    if method in ('rl_ab', 'rl_err', 'sft', 'reflexion', 'rejection_sft'):
        return 'global'
    if method in ('opsd', 'improve_sft', 'logp_rl'):
        return 'local'
    return ''


def _build_pool(spec: ExpSpec, args) -> Any:
    if spec.method == 'improve_sft':
        return SamplePool(batch_size=args.sft_batch_size, balanced=True,
                          max_pool=args.pool_max)
    if spec.method == 'sft':
        return SamplePool(batch_size=args.sft_batch_size, balanced=False,
                          max_pool=args.pool_max)
    if spec.method == 'rejection_sft':
        # E18：攒够 --e18-accumulate（128）条才 fire 一次 SFT；_train_batch 内部仍按
        # sft_batch_size 切 micro，所以这里只需保证是 TRAIN_DP 倍数（main.py 校验）。
        return SamplePool(batch_size=args.e18_accumulate, balanced=False,
                          max_pool=args.pool_max)
    return None  # RL / OPSD / bnpo train per-chunk, no accumulation pool


def _load_resume_state(args) -> dict:
    """Resolve --resume-from into {'ckpt_dir', 'updates', 'chunk_idx'}.

    Weights-only resume (lr is constant; Adam moments restart — accepted trade-off).
    Counter source: <ckpt>/train_state.json (written by _save_ckpt); legacy finished runs
    (e.g. E1-final) fall back to <output_dir>/DONE.json {'updates','chunks'}.
    """
    ck = args.resume_from
    if not os.path.isdir(ck):
        ck = os.path.join(args.output_dir, args.resume_from)
    if not os.path.isdir(ck):
        raise FileNotFoundError(f'--resume-from checkpoint dir not found: {args.resume_from}')
    state_path = os.path.join(ck, 'train_state.json')
    done_path = os.path.join(args.output_dir, 'DONE.json')
    if os.path.exists(state_path):
        with open(state_path, encoding='utf-8') as f:
            st = json.load(f)
        # bugfix #17: 旧/手写 train_state.json 缺键时给出可读报错而非裸 KeyError
        missing = [k for k in ('updates', 'chunk_idx') if k not in st]
        if missing:
            raise ValueError(f'resume: {state_path} missing keys {missing}; '
                             f'present keys: {sorted(st)}')
        for k in ('chunk_size', 'min_level', 'n', 'seed'):
            if k in st and getattr(args, k, None) not in (None, '') and st[k] != getattr(args, k):
                sys.stderr.write(f'[ablate] WARNING: resume data config mismatch: {k} '
                                 f'ckpt={st[k]} vs now={getattr(args, k)} — the continued chunk '
                                 f'sequence will NOT align with the original run.\n')
    elif os.path.exists(done_path):
        with open(done_path, encoding='utf-8') as f:
            d = json.load(f)
        st = {'updates': int(d['updates']), 'chunk_idx': int(d['chunks'])}
        sys.stderr.write('[ablate] resume: no train_state.json in ckpt, counters restored '
                         'from DONE.json (legacy run — data-config alignment unverified).\n')
    else:
        raise FileNotFoundError(f'resume: neither {state_path} nor {done_path} exists; '
                                'cannot restore update/chunk counters.')
    return {'ckpt_dir': ck, 'updates': int(st['updates']), 'chunk_idx': int(st['chunk_idx'])}


def run_experiment(args, spec: ExpSpec) -> None:
    # 0) idempotency: DONE.json is written atomically as the very last step of a successful
    # run; if present, this experiment is complete -> skip (unless --force). --resume-from
    # bypasses the guard by design: its whole point is extending a finished run.
    resume = _load_resume_state(args) if getattr(args, 'resume_from', '') else None
    done_path = os.path.join(args.output_dir, 'DONE.json')
    if os.path.exists(done_path) and not getattr(args, 'force', False) and resume is None:
        sys.stderr.write(f'[ablate] {spec.name} already complete ({done_path}); '
                         f'use --force to rerun.\n')
        return
    if resume is not None:
        args.skill_init_model_id = resume['ckpt_dir']  # v2.init_components skill_model bypass
        sys.stderr.write(f'[ablate] resuming {spec.name} from {resume["ckpt_dir"]} '
                         f'(updates={resume["updates"]} chunk={resume["chunk_idx"]}).\n')

    # 1) task / style / align globals must be set BEFORE any prompt is built.
    # set_task 必须在最前：它同时决定 prompt 分派、判分方式与 rubric 判据版本（缓存键）。
    v2.set_task(spec.task, getattr(args, 'test_workers', 24), getattr(args, 'test_timeout', 60))
    v2._ALIGN_MODE = spec.align
    v2._SKILL_STYLE = spec.style
    args.skill_thinking = spec.thinking
    # per-style length budget (#9 statistics: narrative≈1100 / pitfall≈300 chars);
    # freeform 可能产出叙述式长文本，按 narrative 档给 1100 以免误伤；explicit --len-budget on the CLI wins.
    if args.len_budget is None:
        args.len_budget = 1100 if spec.style in ('narrative', 'freeform') else 300
    # explicit --skill-max-tokens on the CLI wins over the per-experiment default.
    if args.skill_max_tokens is None:
        args.skill_max_tokens = spec.skill_max_tokens
    elif args.skill_max_tokens != spec.skill_max_tokens:
        sys.stderr.write(f'[ablate] WARNING: --skill-max-tokens {args.skill_max_tokens} '
                         f'overrides the {spec.name} default {spec.skill_max_tokens}.\n')
    # E14 信噪比消融：训练判分 rollout 数/温度，CLI 显式值优先，否则取 spec（E1-E13=1×T0）。
    if getattr(args, 'reward_rollouts', None) is None:
        args.reward_rollouts = spec.reward_rollouts
    if getattr(args, 'reward_temperature', None) is None:
        args.reward_temperature = spec.reward_temperature
    # OOM guard: think/8192 实验的训练序列长一倍，fp32 主权重下 16/2卡 的 micro backward
    # 会爆显存（E13 实测 Tried to allocate 37.9GiB）；自动把 micro 减半到 8，梯度归一后数学等价，
    # 攒批/采样批（sft_batch_size）冻结口径不变。显式 --train-micro-batch 优先。
    if not args.train_micro_batch and args.skill_max_tokens >= 8192:
        args.train_micro_batch = max(v2.TRAIN_DP, args.sft_batch_size // 2)
        sys.stderr.write(f'[ablate] train_micro_batch auto-set to {args.train_micro_batch} '
                         f'(skill_max_tokens={args.skill_max_tokens} OOM guard).\n')

    if spec.task == 'code':
        records, eval_records = load_code_records(args)
    else:
        records, eval_records = (load_deepmath_records(args) if getattr(args, 'deepmath_dir', '')
                                 else v2._load_records(args))
    if len(records) < args.chunk_size:
        raise ValueError(f'--chunk-size ({args.chunk_size}) exceeds loaded ({len(records)})')

    os.makedirs(args.output_dir, exist_ok=True)
    gen_path = os.path.join(args.output_dir, 'gen_records.jsonl')
    eval_path = os.path.join(args.output_dir, 'eval_records.jsonl')
    train_log_path = os.path.join(args.output_dir, 'train_log.jsonl')

    # 2) rubric checker BEFORE any GPU allocation: view A without a teacher API cannot run
    # (SFT-family would loop forever on an empty pool, OPSD would distill on empty rubrics).
    checker = v2.build_rubric_checker() if spec.needs_rubric else None
    if spec.needs_rubric and checker is None:
        raise RuntimeError(
            f'{spec.name} ({spec.method}) is a view-A experiment and REQUIRES the rubric '
            'teacher API; set LLM_BACKUP_API_KEY / LLM_BACKUP_BASE_URL (or OPENAI_API_KEY).')

    # 3) components (v2 verbatim); override loss to OPSD when needed.
    skill_model, ref_model, skill_sampler, base_sampler, ckpt, skill_dp, base_dp = \
        v2.init_components(args)
    if spec.loss == 'opsd':
        # no beta: OPSDLoss uses only teacher_logps (no ref-KL term, no ref forward at all)
        skill_model.set_loss('OPSDLoss', reverse=True)

    scope = _rubric_scope(spec.method)
    rubric_cache = build_rubric_cache(scope, args.output_dir,
                                      global_dir=args.rubric_global_dir,
                                      enabled=not args.no_cache,
                                      task=spec.task,
                                      executor_thinking=spec.executor_thinking) if scope else None

    # client-side Template clone:
    # - OPSD: skill-model template, used to align response-token positions for teacher logps.
    # - logp_rl / logp_gt: executor template (thinking on), used to slice logP(S | executor prompt).
    encode_template = None
    if spec.method in ('opsd', 'logp_rl', 'logp_gt'):
        encode_template = v2.Template(model_id=v2.MODEL_ID,
                                      enable_thinking=(True if spec.method in ('logp_rl', 'logp_gt')
                                                       else spec.thinking == 'on'),
                                      max_length=args.max_model_len,
                                      truncation_strategy='delete')

    pool = _build_pool(spec, args)
    ctx = MethodContext(skill_model=skill_model, ref_model=ref_model, skill_sampler=skill_sampler,
                        base_sampler=base_sampler, ckpt=ckpt, skill_dp=skill_dp, base_dp=base_dp,
                        args=args, checker=checker, rubric_cache=rubric_cache, pool=pool,
                        encode_template=encode_template)
    method = build_method(spec.method, ctx)

    def _save_ckpt(name: str, updates: int, chunk_idx: int, epoch: int) -> None:
        """Weights-only checkpoint + barrier + resume state.

        skill_model.save dispatches to the train actors; a subsequent cheap blocking call on
        the SAME actors (lr_step — a guaranteed no-op here: no scheduler, constant lr) acts as
        the barrier: Ray actor tasks run serially per actor, so when it returns the save has
        landed on every rank. Without it the driver could exit on a half-written safetensors.
        """
        skill_model.save(name, output_dir=args.output_dir)
        skill_model.lr_step()  # barrier (see docstring)
        state = {'updates': updates, 'chunk_idx': chunk_idx, 'epoch': epoch,
                 'seed': args.seed, 'chunk_size': args.chunk_size, 'n': args.n,
                 'min_level': int(getattr(args, 'min_level', 0) or 0),
                 'lr': args.lr, 'exp': spec.name, 'saved': int(time.time())}
        with open(os.path.join(args.output_dir, name, 'train_state.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(state, f)
        sys.stderr.write(f'[ablate] checkpoint saved: {name} (updates={updates})\n')

    # 4) eval baseline cache (v2 DiskCache) + swanlab.
    # 每次启动强制重算 eval baseline：旧缓存可能来自不同环境/代码版本（torch/vllm/dtype 均影响 T=0 输出），
    # 跨 run 复用会造成 with-skill（现算）vs baseline（陈旧）不可比，lift 虚高/虚低。
    cache_dir = os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    _base_cache_path = os.path.join(cache_dir, 'eval_baseline.jsonl')
    if os.path.exists(_base_cache_path):
        os.remove(_base_cache_path)
        sys.stderr.write('[ablate] stale eval_baseline cache removed (recomputed this run).\n')
    eval_base_cache = v2.DiskCache(_base_cache_path, not args.no_cache)

    use_swan = swanlab is not None and os.environ.get('SWANLAB_MODE') != 'disabled'
    if use_swan:
        # timestamp suffix so FORCE reruns never collide in swanlab (接口方案 #9);
        # --run-tag 用于区分共用同一 ExpSpec 的变体（如 kl_beta 0.001 vs 0.01）。
        _tag = (getattr(args, 'run_tag', '') or '').strip()
        swan_exp = (f'{spec.swanlab_exp}' + (f'_{_tag}' if _tag else '')
                    + f'_{time.strftime("%Y%m%d_%H%M%S")}')
        swanlab.init(project=args.swanlab_project, experiment_name=swan_exp,
                     config={'exp': spec.name, 'view': spec.view, 'method': spec.method,
                             'thinking': spec.thinking, 'style': spec.style, 'align': spec.align,
                             'loss': spec.loss, 'skill_max_tokens': args.skill_max_tokens,
                             'max_updates': args.max_updates, 'lr': args.lr,
                             'n_skills': args.n_skills, 'sft_batch_size': args.sft_batch_size,
                             'len_budget': args.len_budget,
                             'run_tag': _tag,
                             'reward_rollouts': args.reward_rollouts,
                             'reward_temperature': args.reward_temperature,
                             'kl_beta': getattr(args, 'kl_beta', 0.01),
                             'grpo_epsilon': getattr(args, 'grpo_epsilon', 0.2),
                             'adv_clip': getattr(args, 'adv_clip', 0.0),
                             'logp_leak_penalty': getattr(args, 'logp_leak_penalty', 0.0),
                             'reward_leak_gate': getattr(args, 'reward_leak_gate', 0.0),
                             'reward_trunc_penalty': getattr(args, 'reward_trunc_penalty', 0.25),
                             'reward_trunc_lo': getattr(args, 'reward_trunc_lo', 6000),
                             'base_tok_floor': getattr(args, 'base_tok_floor', 5000),
                             'drop_zero_adv': args.drop_zero_adv})

    cfg = {'record_type': 'config', 'exp': spec.name, 'task': spec.task, 'view': spec.view,
           'method': spec.method,
           'thinking': spec.thinking, 'style': spec.style, 'align': spec.align, 'loss': spec.loss,
           'executor_thinking': spec.executor_thinking,
           'skill_max_tokens': args.skill_max_tokens, 'needs_rubric': spec.needs_rubric,
           'rubric_scope': scope, 'rubric_check': bool(checker),
           'n': len(records), 'eval_n': len(eval_records), 'model': v2.MODEL_ID,
           'max_updates': args.max_updates, 'eval_every_updates': args.eval_every_updates,
           'lr': args.lr, 'n_skills': args.n_skills, 'chunk_size': args.chunk_size,
           'sft_batch_size': args.sft_batch_size, 'len_budget': args.len_budget,
           'skill_char_limit': args.skill_char_limit, 'drop_zero_adv': args.drop_zero_adv,
           'improve_skill_temperature': args.improve_skill_temperature,
           'reward_rollouts': args.reward_rollouts,
           'reward_temperature': args.reward_temperature,
           'run_tag': (getattr(args, 'run_tag', '') or ''),
           # loss 侧旋钮同样入账：kl_beta 是唯一对抗漂移的恢复力，此前不落盘导致已跑的臂
           # 无法从 gen_records 反推当时的锚强度（全部是旧默认 0.001；现默认 0.01）。
           'kl_beta': getattr(args, 'kl_beta', 0.01),
           'grpo_epsilon': getattr(args, 'grpo_epsilon', 0.2),
           'adv_clip': getattr(args, 'adv_clip', 0.0),
           # reward 公式的所有旋钮全部入账：之前 leak 惩罚默认开着却不落盘，导致已跑的臂无法
           # 从 gen_records 反推当时用的是哪个 reward。leak 一律不进 reward，此处应恒为 0。
           'logp_leak_penalty': getattr(args, 'logp_leak_penalty', 0.0),
           'reward_leak_gate': getattr(args, 'reward_leak_gate', 0.0),
           'reward_trunc_penalty': getattr(args, 'reward_trunc_penalty', 0.25),
           'reward_trunc_lo': getattr(args, 'reward_trunc_lo', 6000),
           'base_tok_floor': getattr(args, 'base_tok_floor', 5000),
           'reflexion_k': int(getattr(args, 'reflexion_k', 0) or 0),
           'eval_protocol': ('reflexion' if spec.method == 'reflexion' else 'query_only'),
           'eval_rollouts': args.eval_rollouts, 'eval_skill_temperature': args.eval_skill_temperature,
           'seam_parquet_dir': (getattr(args, 'seam_parquet_dir', '') or ''),
           'deepmath_dir': (getattr(args, 'deepmath_dir', '') or ''),
           'min_level': int(getattr(args, 'min_level', 0) or 0),
           'eval_min_level': int(getattr(args, 'eval_min_level', 0) or 0),
           'save_every_updates': int(getattr(args, 'save_every_updates', 0) or 0),
           'resumed_from': (resume['ckpt_dir'] if resume else ''),
           'resumed_updates': (resume['updates'] if resume else 0),
           'started': int(time.time())}

    # resume appends to the record files (the original history stays intact); a fresh run
    # truncates as before.
    _fmode = 'a' if resume is not None else 'w'
    with open(gen_path, _fmode, encoding='utf-8') as gen_f, \
            open(eval_path, _fmode, encoding='utf-8') as eval_f, \
            open(train_log_path, _fmode, encoding='utf-8') as tlog:
        for f in (gen_f, eval_f, tlog):
            v2._write(f, cfg)

        def _do_eval(updates_done: int, swan_step: int) -> None:
            # E17 用 reflexion 协议 eval（只干预裸解做错的题、skill-gen 带 rubric），与训练
            # 同分布；其余臂一律走 v2 的 query-only 全量 eval。两者 summary 键名兼容。
            if spec.method == 'reflexion':
                recs, summary, metrics = run_reflexion_eval(
                    base_sampler, skill_sampler, eval_records, updates_done, updates_done,
                    base_dp, skill_dp, args, eval_base_cache, rubric_cache, checker)
            elif spec.method == 'rejection_sft':
                # E18：eval 用 nothink rollout，且与采集共用同一个 skill_sampler vLLM（用户
                # 2026-07-30 拍板）。set_template 只换客户端编码（sampler/base.py:106），
                # 引擎不重建；训练响应是裸 <skills>（空 think 布局），与 nothink 生成布局
                # 逐 token 一致，所以这才是本臂的同分布读数。finally 必须切回 think，
                # 否则下一个 chunk 的采集会静默变成 nothink。
                skill_sampler.set_template(v2.Template, model_id=v2.MODEL_ID,
                                           enable_thinking=False,
                                           max_length=args.max_model_len)
                try:
                    recs, summary, metrics = v2.run_greedy_eval(
                        base_sampler, skill_sampler, eval_records, updates_done, updates_done,
                        base_dp, skill_dp, args, eval_base_cache)
                finally:
                    skill_sampler.set_template(v2.Template, model_id=v2.MODEL_ID,
                                               enable_thinking=(spec.thinking == 'on'),
                                               max_length=args.max_model_len)
            else:
                recs, summary, metrics = v2.run_greedy_eval(
                    base_sampler, skill_sampler, eval_records, updates_done, updates_done,
                    base_dp, skill_dp, args, eval_base_cache)
            for rec in recs:
                v2._write(eval_f, rec)
            v2._write(eval_f, summary)
            eval_f.flush()
            if use_swan:
                # same chunk-based axis as the train curves; updates recoverable via the
                # logged eval/updates_done scalar.
                swanlab.log({**{f'eval/{k}': v for k, v in metrics.items()},
                             'eval/updates_done': float(updates_done)}, step=swan_step)
            sys.stderr.write(
                f'[eval] u{updates_done}: n={summary["n"]} acc={summary["baseline_acc_mean1"]:.3f}'
                f'->{summary["acc_mean1"]:.3f} lift={summary["lift_mean1"]:+.3f} '
                f'hard_rescue={summary["hard_rescue_rate"]:.3f} fmt={summary["format_mean1"]:.2f}\n')

        if eval_records and resume is None:
            _do_eval(-1, 0)  # baseline before any update (chunk axis position 0)

        pool_pp = v2.ProblemPool(records, args.seed)
        updates = 0
        last_eval_at = 0
        last_eval_updates = -1
        chunk_idx = 0
        last_save_at = 0
        if resume is not None:
            # push the resumed weights into skill_sampler BEFORE the first chunk (otherwise
            # skill-gen would sample from base weights until the first post-update sync).
            ckpt.sync_weights(merge_and_sync=True)
            updates = resume['updates']
            last_eval_at = updates
            last_save_at = updates
            # fast-forward the pool: draws are deterministic (RandomState(seed+epoch)), so
            # replaying chunk_idx draws restores the exact data position — IF chunk_size /
            # data config match the original run (warned in _load_resume_state).
            for _ in range(resume['chunk_idx']):
                pool_pp.draw(args.chunk_size)
            chunk_idx = resume['chunk_idx']
        save_every = int(getattr(args, 'save_every_updates', 0) or 0)
        while updates < args.max_updates:
            chunk = pool_pp.draw(args.chunk_size)
            res = method.step(chunk, chunk_idx)
            n_upd = int(res.get('n_updates', 0))
            updates += n_upd

            for rec in res.get('gen_records') or []:
                v2._write(gen_f, rec)
            gen_f.flush()

            log = {'record_type': 'train_round', 'exp': spec.name, 'chunk': chunk_idx,
                   'epoch': pool_pp.epoch, 'updates': updates, 'n_updates_step': n_upd,
                   'method': spec.method, 'ts': int(time.time()),
                   'metrics': res.get('metrics', {})}
            if 'summary' in res:  # bnpo carries the v2 chunk summary
                log['summary'] = res['summary']
            v2._write(tlog, log)
            tlog.flush()

            sys.stderr.write(f'[gen] e{pool_pp.epoch} c{chunk_idx}: +{n_upd}upd '
                             f'total={updates}/{args.max_updates} '
                             + ' '.join(f'{k}={v:.3g}' for k, v in res.get('metrics', {}).items()
                                        if isinstance(v, (int, float))) + '\n')
            if use_swan:
                m = {f'{k}': float(v) for k, v in res.get('metrics', {}).items()
                     if isinstance(v, (int, float))}
                m['train/updates'] = float(updates)
                m['train/n_updates_step'] = float(n_upd)
                swanlab.log(m, step=chunk_idx + 1)  # +1: step 0 is the eval baseline

            # eval by parameter-update cadence (same chunk-based swan axis)
            if eval_records and updates >= last_eval_at + args.eval_every_updates and n_upd > 0:
                _do_eval(updates, chunk_idx + 1)
                last_eval_at = updates
                last_eval_updates = updates
            # periodic weights-only checkpoint (same cadence semantics as eval)
            if save_every and updates >= last_save_at + save_every and n_upd > 0:
                _save_ckpt(f'{spec.name}-u{updates}', updates, chunk_idx + 1, pool_pp.epoch)
                last_save_at = updates
            # bugfix #12: 方法把候选/rollout 全文挂在共享 record dict 上（ProblemPool 跨 epoch
            # 复用同一批对象），gen_records 已落盘后不清理会让 driver RAM 随触达题数线性增长
            # （5000 题 × 8 候选 × 几十 KB 可达数 GB）。训练/日志都结束后安全清理。
            for r in chunk:
                for k in ('_cands', '_pseudo_rolls', '_pseudo_roll', '_pseudo_solution',
                          '_rubric', '_base_tok', '_base_correct', '_base_stop'):
                    r.pop(k, None)
            chunk_idx += 1

        # final readout — skip if the periodic eval already covered this exact update count
        if eval_records and updates != last_eval_updates:
            _do_eval(updates, chunk_idx + 1)

    eval_base_cache.close()
    if rubric_cache is not None:
        rubric_cache.close()
    # final save goes through _save_ckpt: barrier guarantees the safetensors is fully on disk
    # BEFORE DONE.json can exist, and train_state.json makes the final model resumable too.
    _save_ckpt(f'{spec.name}-final', updates, chunk_idx, pool_pp.epoch)
    # completion sentinel LAST (after the final model lands); temp+rename keeps it atomic so a
    # crash can never leave a truthy half-written marker.
    tmp = done_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'exp': spec.name, 'updates': updates, 'chunks': chunk_idx,
                   'epochs': pool_pp.epoch, 'finished': int(time.time())}, f)
    os.replace(tmp, done_path)
    sys.stderr.write(f'[ablate] {spec.name} done: {updates} updates over {chunk_idx} chunks / '
                     f'{pool_pp.epoch} epochs\n')
