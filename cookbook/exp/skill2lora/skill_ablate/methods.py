# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pluggable training methods (E1-E12) over a shared context.

Each method implements TrainMethod: ``step(chunk, ci) -> dict`` where the dict carries at
least ``n_updates`` (parameter-update count, the "step" unit per skill_quality_analysis.md
#2), plus ``metrics`` (scalars to log) and ``gen_records`` (per-problem rollout audit rows
for gen_records.jsonl). New methods / losses are added by writing one class + one
METHOD_REGISTRY entry — the trainer never changes.

Reuse map (all imported from train_skill_v2 / rollouting, never edited):
- bnpo (view B): v2 ``process_chunk`` + ``_train_step`` verbatim (query-only).
- rl_ab / rl_err (view A): bare greedy solve -> wrong=A(query+rubric) / right=B(query-only),
  8 skills each -> executor greedy reward -> group advantage -> BNPO on the rubric trajectory
  (train-with-rubric); rl_err drops the B line from training. Rubric API calls run on threads
  WHILE the B line rolls out on GPU (API/GPU overlap).
- opsd (view A): error problems, 1 student skill (query-only, T=0.5); teacher forward
  (student prompt + rubric appended to the SYSTEM prompt, same response) -> per-token OPSD
  KL (loss='opsd'). Teacher logps are extracted RESPONSE-ONLY via a client-side template
  encode (teacher/student prompt lengths differ, so the full-sequence form would misalign).
- improve_sft (view A): first-pass 1 skill (query-only, T=0.5); correct -> positive SFT seed
  (no leak, <=4096 chars); parseable-but-wrong -> rubric regen (2-in-8 pick 1) -> negative
  SFT seed; unparseable first pass is SKIPPED (no trajectory to diagnose); balanced 1:1 pool
  (majority side down-sampled per chunk, never backlogged) -> SFT.
- sft (view A): bare wrong -> rubric -> regen (query+rubric) 2-in-8 -> plain pool -> SFT.

Training-batch helper ``_train_batch`` mirrors v2 ``_train_step`` exactly (empty-response
filter, drop_last to TRAIN_DP, micro-batch by sft_batch_size, ppo_mini_batch_size multi-step
with pre-computed ref/old logps, clip+step per mini, ckpt sync, calculate_metric) but takes a
swappable trajectory builder (query-only vs query+rubric) and an optional teacher builder for
OPSD. In the OPSD path no ref forward is done at all: OPSDLoss uses only teacher_logps
(kl_beta / ref_logps play no role there).
"""
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import train_skill_v2 as v2
from train_skill_v2 import (
    TRAIN_DP,
    _answer_leaked,
    _assign_advantages,
    _clean_text,
    _empty_roll,
    _extract_skill,
    _parse_seq,
    _regen_prompt,
    _run_samples,
    _skill_reward,
    _skillgen_prompt,
    _train_step,
    build_direct_prompt,
    build_skill_solve_prompt,
    process_chunk,
)

from .pool import NEG, POS, SamplePool
from .rollouting import (
    opsd_teacher_trajectory,
    query_only_train_trajectory,
    rubric_skillgen_prompt,
    rubric_train_trajectory,
)


@dataclass
class MethodContext:
    """Everything a method needs; assembled once by the trainer."""
    skill_model: Any
    ref_model: Any
    skill_sampler: Any
    base_sampler: Any
    ckpt: Any
    skill_dp: int
    base_dp: int
    args: Any
    checker: Any = None
    rubric_cache: Any = None           # GlobalRubricCache (RL/SFT) or LocalRubricCache (improve/opsd)
    pool: Optional[SamplePool] = None  # SFT-family accumulator (None for RL/OPSD)
    encode_template: Any = None        # client-side Template clone (OPSD teacher alignment only)
    extra: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================================
# shared low-level helpers (reuse v2 primitives; only orchestration is new)
# ===========================================================================================
def _bare_solve(ctx: MethodContext, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bare-problem greedy (T=0) executor solve; returns one roll per record (order-aligned)."""
    out = _run_samples(ctx.base_sampler, [build_direct_prompt(r['problem']) for r in records],
                       1, ctx.args.max_tokens, ctx.base_dp, temperature=0.0)
    return v2._parse_many([(v2._first_seq(seqs), r['reference_answer'])
                           for r, seqs in zip(records, out)])


def _rubric_entry(record: Dict[str, Any], roll: Dict[str, Any]) -> Dict[str, Any]:
    """Build the entry _diagnose_entry expects from a failure trajectory.

    code 任务：fail_segment 不是"输出全文"，而是**提交的代码 + 单测真实报错**（异常类型 /
    断言差异 / 失败用例名）。这是三个数据集横比下 rubric 唯一真正产生增量的原因 —— 数学与
    BFCL 上 judge 手里没有任何客观证据，只能猜，命中率≈随机。
    """
    if v2._TASK == 'code':
        return {'problem': record['problem'], 'reference_answer': record['reference_answer'],
                'data_id': record.get('data_id', ''),
                'fail_segment': v2.code_task.diag_segment(roll),
                'fail_stop_reason': roll.get('stop_reason', 'none')}
    return {'problem': record['problem'], 'reference_answer': record['reference_answer'],
            'data_id': record.get('data_id', ''),
            'fail_segment': roll.get('text', ''),
            'fail_stop_reason': roll.get('stop_reason', 'none')}


def _diagnose_parallel(ctx: MethodContext,
                       jobs: List[Tuple[Dict[str, Any], Optional[str]]]) -> List[str]:
    """Run rubric diagnoses in parallel threads (pure API, no GPU; DiskCache.put is locked).

    ``jobs`` is a list of (entry, skill_or_None); returns diagnoses aligned to jobs
    ('' on cache-off / API error). Parallelism = --rubric-workers (was serial before)."""
    if not jobs or ctx.rubric_cache is None:
        return [''] * len(jobs)
    workers = max(1, min(ctx.args.rubric_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(
            lambda j: ctx.rubric_cache.get_or_diagnose(j[0], ctx.checker, skill=j[1]) or '',
            jobs))


def _skillgen_solve(ctx: MethodContext, items: List[Dict[str, Any]], n_skills: int,
                    temperature: float) -> None:
    """For each item {record, prompt}: sample n skills, greedy-solve each, attach a
    ``_cands`` list (v2 shape) onto the record so ``_assign_advantages`` can be reused."""
    args = ctx.args
    sg_out = _run_samples(ctx.skill_sampler, [it['prompt'] for it in items], n_skills,
                          args.skill_max_tokens, ctx.skill_dp,
                          temperature=temperature, top_p=args.skill_gen_top_p,
                          top_k=args.skill_gen_top_k)
    flat = []
    for it, seqs in zip(items, sg_out):
        it['record']['_cands'] = []
        for s in seqs:
            resp = _clean_text(getattr(s, 'decoded', '') or '')
            block = _extract_skill(resp) or ''
            cand = {'skills': block, 'response': resp, 'parseable': bool(block),
                    'leaked': None, 'with_pass': None, 'reward': None, 'rolls': [],
                    'advantage': 0.0, 'kept': False,
                    'skillgen_stop': getattr(s, 'stop_reason', None),
                    'skillgen_tokens': len(getattr(s, 'tokens', None) or [])}
            it['record']['_cands'].append(cand)
            if block:
                flat.append((it, cand))
    for it, c in flat:
        c['leaked'] = _answer_leaked(c['skills'], it['record']['reference_answer'])
    if flat:
        # V2 fix: pass the raw skill-gen response like v2 process_chunk does — seam align
        # nests the actor's full response_text into the executor prompt (no-op in v2 mode).
        ws = _run_samples(ctx.base_sampler,
                          [build_skill_solve_prompt(it['record']['problem'], c['skills'], c.get('response'))
                           for it, c in flat],
                          1, args.max_tokens, ctx.base_dp, temperature=0.0)
        judged = v2._parse_many([(v2._first_seq(seqs), it['record']['reference_answer'])
                                 for (it, _c), seqs in zip(flat, ws)])
        for (it, c), roll in zip(flat, judged):
            c['rolls'] = [roll]
            c['with_pass'] = 1.0 if roll['correct'] else 0.0
            c['reward'] = _skill_reward(c['parseable'], roll['correct'])
    for it in items:
        for c in it['record']['_cands']:
            if c['reward'] is None:
                c['reward'] = 0.0


def _grpo_records(records: List[Dict[str, Any]], with_rubric: bool) -> List[Dict[str, Any]]:
    """Flatten per-problem _cands into GRPO train records (v2 shape).
    ``with_rubric`` tags each record so the trajectory builder knows which prompt to rebuild."""
    recs = []
    for r in records:
        for c in r.get('_cands', []):
            if c.get('reward') is None:
                continue
            recs.append({'problem': r['problem'], 'reference_answer': r['reference_answer'],
                         'data_id': r.get('data_id', ''), 'response': c['response'],
                         'skills': c['skills'], 'advantage': c['advantage'],
                         'kept': c['kept'], 'reward': c['reward'], 'rubric': r.get('_rubric', ''),
                         'with_rubric': with_rubric, 'sft': False})
    return recs


def _leak_blocks(skill: str, reference) -> bool:
    """Filtering-grade leak gate (bugfix #4): only answers >=2 chars are informative — the
    same gate E14/E16 already apply to their reward penalty. Single-char golds ('2' etc.)
    substring-match ordinary math prose (~84% false positives measured), which starved the
    SFT-family pools by discarding nearly every regen candidate. Monitoring paths keep the
    raw ``_answer_leaked`` so the recorded leak/rate 口径 is unchanged."""
    return len(str(reference).strip()) >= 2 and _answer_leaked(skill, reference)


def _leak_split(pairs: List[Tuple[bool, bool]]) -> Dict[str, float]:
    """#10 monitoring curves: leaked&correct vs leaked&wrong rates over parseable skills.
    "泄露正确答案可接受"不等于"泄露无害"——有害的是错误数值注入，两条曲线拆开监控。"""
    n = len(pairs)
    if not n:
        return {'leak/correct_rate': 0.0, 'leak/wrong_rate': 0.0}
    return {'leak/correct_rate': sum(1 for lk, ok in pairs if lk and ok) / n,
            'leak/wrong_rate': sum(1 for lk, ok in pairs if lk and not ok) / n}


def _cand_leak_pairs(records: List[Dict[str, Any]]) -> List[Tuple[bool, bool]]:
    # bugfix #7: with_pass is a float pass RATE under M>1 rollouts (E16) — bool(0.25) would
    # count a partial pass as "correct"; compare > 0 instead (identical for greedy 0/1 arms).
    return [(bool(c['leaked']), (c['with_pass'] or 0) > 0)
            for r in records for c in r.get('_cands', [])
            if c.get('parseable') and c.get('with_pass') is not None]


def _cand_pass_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean-family train metrics, same family as eval acc_mean1 (ws_acc is pass@8-inflated):
    candidate_pass = P(correct | parseable); clean_pass = P(correct | parseable & terminated).
    Sharper channel split by ANSWER AVAILABILITY (truncated rolls still count correct when a
    balanced \\boxed{} landed before the budget — 13% of E1 truncations did):
    answered_rate = P(pred emitted) and answered_pass = P(correct | answered), the content-only
    channel (E1 vs E5 core comparison curve); plus parse/trunc rates for the format channel."""
    cands = [c for r in records for c in r.get('_cands', [])]
    if not cands:
        return {}
    m = {'skill/parse_rate': sum(1 for c in cands if c.get('parseable')) / len(cands)}
    scored = [c for c in cands
              if c.get('parseable') and c.get('with_pass') is not None and c.get('rolls')]
    if scored:
        m['acc/candidate_pass'] = sum(c['with_pass'] for c in scored) / len(scored)
        clean = [c for c in scored if c['rolls'][0].get('stop_reason') != 'length']
        m['term/withskill_trunc_frac'] = 1.0 - len(clean) / len(scored)
        if clean:
            m['acc/clean_pass'] = sum(c['with_pass'] for c in clean) / len(clean)
        answered = [c for c in scored if c['rolls'][0].get('pred') not in (None, '')]
        m['term/answered_rate'] = len(answered) / len(scored)
        if answered:
            m['acc/answered_pass'] = sum(c['with_pass'] for c in answered) / len(answered)
    return m


def _train_metrics(metric: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """calculate_metric -> swan-ready train/* scalars (same key handling as v2 _swan_metrics)."""
    d = {}
    for k, val in (metric or {}).items():
        if not v2._is_num(val):
            continue
        if k.startswith('learning rate'):
            if 'group 1' in k:
                d['train/lr'] = float(val)
        else:
            d[f'train/{k.replace(" ", "_")}'] = float(val)
    return d


# ===========================================================================================
# OPSD teacher alignment: client-side encode -> response-only teacher logps
# ===========================================================================================
def _align_teacher(ctx: MethodContext, samples, traj_fn, teacher_fn):
    """Encode student & teacher trajectories with a CLIENT-SIDE clone of the remote template
    and keep only samples whose response-token counts match (they always should — same
    assistant text — but max-length 'delete' truncation or tokenizer drift must not silently
    misalign the distillation). Returns (kept_samples, teacher_label_positions, n_dropped).

    teacher_label_positions[i] are the (rolled-)label indices of sample i inside its OWN
    unpadded teacher sequence; right padding keeps these indices valid in the padded batch,
    so they can slice the teacher's full-sequence logps down to the response-only form that
    OPSDLoss requires (full-sequence form would misalign: prompts differ in length).
    """
    tmpl = ctx.encode_template
    assert tmpl is not None, 'OPSD needs ctx.encode_template (built by the trainer)'
    keep, pos_lists, dropped = [], [], 0
    for s in samples:
        st = tmpl.encode(traj_fn(s))
        tt = tmpl.encode(teacher_fn(s))
        if st is None or tt is None:               # deleted by max-length truncation
            dropped += 1
            continue
        spos = np.where(np.asarray(st.get('labels')) != -100)[0]
        tpos = np.where(np.asarray(tt.get('labels')) != -100)[0]
        if len(spos) != len(tpos) or len(tpos) == 0:
            dropped += 1
            continue
        keep.append(s)
        pos_lists.append(tpos)
    return keep, pos_lists, dropped


def _gather_response_logps(full_logps, pos_lists) -> List[List[float]]:
    """Slice full-sequence [B, S] teacher logps down to per-sample response-only lists."""
    rows = []
    for i, pos in enumerate(pos_lists):
        row = full_logps[i]
        row = row.tolist() if hasattr(row, 'tolist') else list(row)
        assert len(pos) == 0 or int(pos[-1]) < len(row), \
            f'teacher logps row {i} shorter than label positions ({len(row)} <= {int(pos[-1])})'
        rows.append([float(row[int(p)]) for p in pos])
    return rows


# ===========================================================================================
# E14+ helpers: executor pseudo-GT + dense logP reward
# ===========================================================================================
def _executor_answer_trajectory(problem: str, skill: str, answer_text: str,
                                raw_response: Optional[str] = None) -> Dict[str, Any]:
    """Teacher-forcing trajectory for executor logP(S | problem + skill)."""
    msgs = [dict(m) for m in build_skill_solve_prompt(problem, skill, raw_response)['messages']]
    return {'messages': msgs + [{'role': 'assistant', 'content': answer_text}],
            'user_data': {'key_rounds': [len(msgs)]}}


def _set_ref_executor_template(ctx: MethodContext, enable_thinking: bool) -> None:
    """Temporarily reuse ref_model as frozen executor scorer, then restore skill/ref layout."""
    ctx.ref_model.set_template(v2.Template, model_id=v2.MODEL_ID,
                               enable_thinking=enable_thinking,
                               max_length=ctx.args.max_model_len,
                               truncation_strategy='delete')


def _mean_logp_rows(full_logps, pos_lists) -> List[float]:
    rows = _gather_response_logps(full_logps, pos_lists)
    return [(sum(x) / len(x)) if x else float('-inf') for x in rows]


def _score_executor_mean_logps(ctx: MethodContext, trajs: List[Dict[str, Any]]) -> List[Optional[float]]:
    """Return mean response-token logP under the frozen executor template; None means truncated."""
    tmpl = ctx.encode_template
    assert tmpl is not None, 'logp_rl needs ctx.encode_template'
    out: List[Optional[float]] = [None] * len(trajs)
    valid_trajs, pos_lists, valid_idx = [], [], []
    for i, tr in enumerate(trajs):
        enc = tmpl.encode(tr)
        if enc is None:
            continue
        pos = np.where(np.asarray(enc.get('labels')) != -100)[0]
        if len(pos) == 0:
            continue
        valid_trajs.append(tr)
        pos_lists.append(pos)
        valid_idx.append(i)
    if not valid_trajs:
        return out
    sft = getattr(ctx.args, 'train_micro_batch', 0) or ctx.args.sft_batch_size
    dp = max(1, v2.REF_DP)
    _set_ref_executor_template(ctx, enable_thinking=True)
    try:
        for st in range(0, len(valid_trajs), sft):
            mb = valid_trajs[st:st + sft]
            mb_pos = pos_lists[st:st + sft]
            n_mb = len(mb)
            # forward_only 按 slice_dp 切分，非 dp 整倍数的尾批用末尾样本补齐，输出只取前 n_mb 行
            if n_mb % dp:
                pad = dp - (n_mb % dp)
                mb = mb + [mb[-1]] * pad
                mb_pos = list(mb_pos) + [mb_pos[-1]] * pad
            vals = _mean_logp_rows(ctx.ref_model.forward_only(inputs=mb).get('logps'), mb_pos)[:n_mb]
            for j, val in enumerate(vals):
                out[valid_idx[st + j]] = float(val)
    finally:
        _set_ref_executor_template(ctx, enable_thinking=(ctx.args.skill_thinking == 'on'))
    return out


def _train_batch(ctx: MethodContext, samples: List[Dict[str, Any]],
                 traj_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                 teacher_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 ) -> Tuple[int, Dict[str, float]]:
    """Parameter update(s) over ``samples`` with a swappable trajectory builder.

    Mirrors v2 ``_train_step`` (empty-response filter, drop_last to TRAIN_DP, micro-batch by
    sft_batch_size, ppo_mini_batch_size multi-step with ALL ref/old/teacher logps pre-computed
    BEFORE the first optimizer step — the teacher is the trainable model itself, so computing
    it after a step would go off-policy). Returns (n_updates, train metrics).
    """
    args = ctx.args
    samples = [s for s in samples if (s.get('response') or '').strip()]
    is_opsd = teacher_fn is not None
    teacher_pos: Optional[List] = None
    n_align_drop = 0
    if is_opsd:
        samples, teacher_pos, n_align_drop = _align_teacher(ctx, samples, traj_fn, teacher_fn)
    if not samples:
        return 0, {}
    n_keep = (len(samples) // TRAIN_DP) * TRAIN_DP
    if n_keep == 0:
        return 0, {}
    samples = samples[:n_keep]
    if teacher_pos is not None:
        teacher_pos = teacher_pos[:n_keep]
    trajs = [traj_fn(s) for s in samples]
    advs = None if is_opsd else [float(s['advantage']) for s in samples]
    # micro 尺寸与“攒批/采样批（sft_batch_size=16，冻结口径）”解耦：think/8192 实验序列长一倍，
    # fp32 主权重后 8 条/卡的 backward 会 OOM，用 --train-micro-batch 切细（梯度按 micro 数归一，
    # 数学等价）；默认 0 = 跟随 sft_batch_size，nothink 实验行为不变。
    n = len(trajs)
    sft = getattr(args, 'train_micro_batch', 0) or args.sft_batch_size
    mini = args.ppo_mini_batch_size if args.ppo_mini_batch_size > 0 else n
    mini = max(sft, (mini // sft) * sft)
    multi_step = mini < n
    # pre-compute every micro's ref/old/teacher logps BEFORE any update (v2 pattern)
    micro_ref, micro_old, micro_teacher = [], [], []
    for i in range(0, n, sft):
        mb = trajs[i:i + sft]
        if is_opsd:
            # no ref forward at all: OPSDLoss uses only teacher_logps (kl_beta plays no role)
            t_mb = [teacher_fn(s) for s in samples[i:i + sft]]
            t_full = ctx.skill_model.forward_only(inputs=t_mb).get('logps')
            micro_teacher.append(_gather_response_logps(t_full, teacher_pos[i:i + sft]))
            micro_ref.append(None)
            micro_old.append(None)
        else:
            micro_ref.append(ctx.ref_model.forward_only(inputs=mb).get('logps'))
            micro_old.append(ctx.skill_model.forward_only(inputs=mb).get('logps') if multi_step else None)
            micro_teacher.append(None)
    n_steps = 0
    for ms in range(0, n, mini):
        for i in range(ms, min(ms + mini, n), sft):
            k = i // sft
            if is_opsd:
                ctx.skill_model.forward_backward(inputs=trajs[i:i + sft],
                                                 teacher_logps=micro_teacher[k])
            else:
                ctx.skill_model.forward_backward(inputs=trajs[i:i + sft], advantages=advs[i:i + sft],
                                                 old_logps=micro_old[k], ref_logps=micro_ref[k])
        ctx.skill_model.clip_grad_and_step()
        n_steps += 1
    ctx.ckpt.sync_weights(merge_and_sync=True)
    metrics = _train_metrics(ctx.skill_model.calculate_metric(is_training=True))
    metrics['train/n_samples'] = float(n)
    if is_opsd and n_align_drop:
        metrics['train/n_align_dropped'] = float(n_align_drop)
    return n_steps, metrics


# ===========================================================================================
# method plugins
# ===========================================================================================
class TrainMethod:
    needs_rubric: bool = False

    def __init__(self, ctx: MethodContext):
        self.ctx = ctx

    def step(self, chunk: List[Dict[str, Any]], ci: int) -> Dict[str, Any]:
        raise NotImplementedError


class BnpoMethod(TrainMethod):
    """view B, query-only GRPO/BNPO — v2 process_chunk + _train_step verbatim."""
    needs_rubric = False

    def step(self, chunk, ci):
        ctx = self.ctx
        full, summary, grpo, _buf_a = process_chunk(
            ctx.base_sampler, ctx.skill_sampler, chunk, ci, ctx.base_dp, ctx.skill_dp, ctx.args)
        if grpo and getattr(ctx.args, 'drop_zero_adv', False):
            grpo = [g for g in grpo if abs(g['advantage']) > 1e-9]
        n_upd, tmetrics = 0, {}
        if grpo:
            log = _train_step(ctx.skill_model, ctx.ref_model, ctx.ckpt, grpo, ctx.args)
            # step = ACTUAL parameter updates: empty-response filter / drop_last may yield 0
            n_upd = int(log.get('n_steps', 0))
            tmetrics = _train_metrics(log.get('metric'))
            tmetrics['train/n_samples'] = float(log.get('n_grpo', 0) + log.get('n_sft', 0))
        metrics = {'signal/zero_grad_frac': summary['zero_grad_frac'],
                   'signal/group_reward_std_mean': summary['group_reward_std_mean'],
                   'signal/n_train_samples': float(summary['n_train_samples']),
                   'signal/n_groups': float(summary['n_groups']),
                   # 旧名（题级 pass@K，历史面板兼容）
                   'acc/withskill_pass': summary['avg_withskill_pass'],
                   # ⭐ 与 SEAM ray_trainer.py:1569-1583 同名同口径，用于 swanlab 直接叠图对齐：
                   #   train/with_skill_accuracy = 全部候选的 mean(correct)（SEAM withskill_pass）
                   #   acc/reward_mean = mean(correct∧format)（SEAM reward_mean）
                   #   skill/format_rate = SEAM format_mean
                   'train/with_skill_accuracy': summary['withskill_pass_all_cands'],
                   'acc/reward_mean': summary['reward_mean'],
                   'skill/format_rate': summary['parse_rate'],
                   'leak/rate': summary['leak_rate'], **tmetrics,
                   **_cand_pass_metrics(chunk),
                   **_leak_split(_cand_leak_pairs(chunk))}
        return {'n_updates': n_upd, 'summary': summary, 'metrics': metrics,
                'gen_records': full}


class _RLViewA(TrainMethod):
    """Shared view-A RL: bare solve -> A(query+rubric)/B(query-only) skill-gen -> reward ->
    group advantage -> BNPO. ``train_b`` toggles whether the right-answer B line is trained.

    API/GPU overlap: rubric diagnoses for the wrong (A) problems run on background threads
    WHILE the right (B) problems' query-only skill-gen + greedy validation run on GPU; the
    A-line rollout starts as soon as the diagnoses land."""
    needs_rubric = True
    train_b = True

    def step(self, chunk, ci):
        ctx = self.ctx
        args = ctx.args
        rolls = _bare_solve(ctx, chunk)
        wrong = [(r, roll) for r, roll in zip(chunk, rolls) if not roll['correct']]
        right = [r for r, roll in zip(chunk, rolls) if roll['correct']]
        # kick off rubric API calls in the background, then run the B line on GPU meanwhile
        diag_pool = ThreadPoolExecutor(max_workers=1)
        diag_fut = diag_pool.submit(
            _diagnose_parallel, ctx, [(_rubric_entry(r, roll), None) for r, roll in wrong])
        try:
            b_items = [{'record': r, 'prompt': _skillgen_prompt(r['problem'])} for r in right]
            for r in right:
                r['_rubric'] = ''
            if b_items:
                _skillgen_solve(ctx, b_items, args.n_skills, temperature=args.skill_gen_temperature)
            diags = diag_fut.result()
        finally:
            diag_pool.shutdown(wait=False)
        a_items = []
        degraded = []  # bugfix #2: rubric 缺失（API 失败/坏缓存）→ 降级 query-only B 线，绝不训练空 rubric prompt
        for (r, _roll), diag in zip(wrong, diags):
            r['_rubric'] = diag
            if diag:
                a_items.append({'record': r, 'prompt': rubric_skillgen_prompt(r['problem'], diag)})
            else:
                degraded.append({'record': r, 'prompt': _skillgen_prompt(r['problem'])})
        if degraded:
            _skillgen_solve(ctx, degraded, args.n_skills, temperature=args.skill_gen_temperature)
        if a_items:
            _skillgen_solve(ctx, a_items, args.n_skills, temperature=args.skill_gen_temperature)
        _assign_advantages(chunk, args)
        a_recs = _grpo_records([it['record'] for it in a_items], with_rubric=True)
        b_recs = _grpo_records(right + [it['record'] for it in degraded], with_rubric=False)
        train_recs = a_recs + (b_recs if self.train_b else [])
        has_signal = any(abs(s['advantage']) > 1e-9 for s in train_recs)
        if has_signal and getattr(args, 'drop_zero_adv', False):
            train_recs = [s for s in train_recs if abs(s['advantage']) > 1e-9]
        n_upd, tmetrics = 0, {}
        if has_signal:
            n_upd, tmetrics = _train_batch(
                ctx, train_recs,
                traj_fn=lambda s: (rubric_train_trajectory(s) if s['with_rubric']
                                   else query_only_train_trajectory(s)))
        return {'n_updates': n_upd,
                'metrics': {'signal/n_wrong_A': float(len(a_items)),
                            'signal/n_right_B': float(len(right)),
                            'signal/n_rubric_missing': float(len(degraded)), **tmetrics,
                            **_cand_pass_metrics(chunk),
                            **_leak_split(_cand_leak_pairs(chunk))},
                'gen_records': v2._full_records(chunk, ci)}


class RlAbMethod(_RLViewA):
    train_b = True


class RlErrMethod(_RLViewA):
    train_b = False


class OpsdMethod(TrainMethod):
    """view A OPSD (skill_quality_analysis.md 改进skill+OPSD): first-pass ONE skill
    (query-only, T=improve), executor solve WITH skill; for WRONG problems, diagnose the
    with-skill failure (local cache, key=data_id+skill), then distill the skill-gen response
    from the query-only (student) toward the rubric-in-system-prompt (teacher) distribution
    per token. Rubric API calls run threaded right after the failures are known."""
    needs_rubric = True

    def step(self, chunk, ci):
        ctx = self.ctx
        args = ctx.args
        # first-pass ONE skill per problem (query-only, improve temperature)
        sg = _run_samples(ctx.skill_sampler, [_skillgen_prompt(r['problem']) for r in chunk],
                          1, args.skill_max_tokens, ctx.skill_dp,
                          temperature=args.improve_skill_temperature)
        first = []
        for r, seqs in zip(chunk, sg):
            resp = _clean_text(getattr(seqs[0], 'decoded', '') or '') if seqs else ''
            first.append((r, resp, _extract_skill(resp) or ''))
        # executor solve WITH skill (only parseable skills)
        flat = [(r, resp, sk) for r, resp, sk in first if sk]
        roll_by = {}
        if flat:
            solve = _run_samples(ctx.base_sampler,
                                 [build_skill_solve_prompt(r['problem'], sk) for r, _, sk in flat],
                                 1, args.max_tokens, ctx.base_dp, temperature=0.0)
            for (r, _, sk), seqs in zip(flat, solve):
                roll_by[id(r)] = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
        wrong = [(r, resp, sk, roll_by[id(r)]) for r, resp, sk in first
                 if sk and id(r) in roll_by and not roll_by[id(r)]['correct']]
        leak_pairs = [(bool(_answer_leaked(sk, r['reference_answer'])), roll_by[id(r)]['correct'])
                      for r, _resp, sk in flat if id(r) in roll_by]
        # rubric diagnoses (threaded; nothing left to overlap on GPU this chunk)
        diags = _diagnose_parallel(
            ctx, [(_rubric_entry(r, roll), sk) for r, _resp, sk, roll in wrong])
        samples = [{'problem': r['problem'], 'reference_answer': r['reference_answer'],
                    'data_id': r.get('data_id', ''), 'response': resp, 'rubric': diag}
                   for (r, resp, sk, _roll), diag in zip(wrong, diags)]
        n_upd, tmetrics = 0, {}
        if samples:
            n_upd, tmetrics = _train_batch(ctx, samples,
                                           traj_fn=query_only_train_trajectory,  # student: query-only
                                           teacher_fn=opsd_teacher_trajectory)   # teacher: +rubric in system
        return {'n_updates': n_upd,
                'metrics': {'signal/n_wrong': float(len(wrong)), **tmetrics,
                            **_leak_split(leak_pairs)},
                'gen_records': [
                    {'record_type': 'problem', 'chunk': ci, 'data_id': r.get('data_id', ''),
                     'problem': r['problem'], 'reference_answer': r['reference_answer'],
                     'skill': sk, 'parseable': bool(sk),
                     'withskill_correct': roll_by[id(r)]['correct'] if id(r) in roll_by else None}
                    for r, _resp, sk in first]}


class LogpRlMethod(TrainMethod):
    """E14+: rubric-audited pseudo-GT + executor logP dense reward.

    Per problem, sample K executor attempts without skill at T>0, keep the first locally-correct
    and non-truncated response S, audit it through the rubric API/cache, then score each generated
    skill by Δ mean logP_executor(S | problem + skill). Problems with no S are skipped this round.
    """
    needs_rubric = True

    def step(self, chunk, ci):
        usable, sg = self._prepare(chunk, ci)
        return self._score_and_train(chunk, ci, usable, sg)

    def _skillgen(self, usable):
        ctx = self.ctx
        args = ctx.args
        return _run_samples(ctx.skill_sampler, [_skillgen_prompt(r['problem']) for r in usable],
                            args.n_skills, args.skill_max_tokens, ctx.skill_dp,
                            temperature=args.skill_gen_temperature, top_p=args.skill_gen_top_p,
                            top_k=args.skill_gen_top_k)

    def _prepare(self, chunk, ci):
        """executor T>0 采 K 条 -> 选本地判分正确且非截断的伪 GT S -> rubric API 后台审计
        （与 skill-gen 的 GPU rollout 重叠）。返回 (usable, 每题 skill-gen 序列)。"""
        ctx = self.ctx
        args = ctx.args
        for r in chunk:
            r['_cands'] = []
        K = max(1, int(getattr(args, 'reward_rollouts', 1) or 1))
        temp = float(getattr(args, 'reward_temperature', 0.0) or 0.0)
        out = _run_samples(ctx.base_sampler, [build_direct_prompt(r['problem']) for r in chunk],
                           K, args.max_tokens, ctx.base_dp, temperature=temp)
        usable, audit_jobs = [], []
        for r, seqs in zip(chunk, out):
            rolls = [_parse_seq(s, r['reference_answer']) for s in (seqs or [])]
            ok = next((x for x in rolls if x['correct'] and x.get('stop_reason') != 'length'), None)
            r['_pseudo_rolls'] = rolls
            if ok is None:
                continue
            r['_pseudo_roll'] = ok
            r['_pseudo_solution'] = ok.get('text', '')
            usable.append(r)
            audit_jobs.append((_rubric_entry(r, ok), ok.get('text', '')))
        # rubric 审计是纯 API：后台线程跑，与 skill-gen 的 GPU rollout 重叠（API/GPU overlap）
        diag_pool = ThreadPoolExecutor(max_workers=1)
        diag_fut = diag_pool.submit(_diagnose_parallel, ctx, audit_jobs)
        try:
            sg = self._skillgen(usable)
            diags = diag_fut.result()
        finally:
            diag_pool.shutdown(wait=False)
        for r, diag in zip(usable, diags):
            r['_rubric'] = diag
        return usable, sg

    def _score_and_train(self, chunk, ci, usable, sg):
        ctx = self.ctx
        args = ctx.args
        flat = []
        for r, seqs in zip(usable, sg):
            for si, s in enumerate(seqs or []):
                resp = _clean_text(getattr(s, 'decoded', '') or '')
                block = _extract_skill(resp) or ''
                pseudo = dict(r['_pseudo_roll'])
                if si > 0:
                    pseudo['text'] = ''  # 磁盘保护：伪 GT 全文每题只在首个候选保留一份
                cand = {'skills': block, 'response': resp, 'parseable': bool(block),
                        'leaked': None, 'with_pass': None, 'reward': None, 'rolls': [pseudo],
                        'advantage': 0.0, 'kept': False,
                        'skillgen_stop': getattr(s, 'stop_reason', None),
                        'skillgen_tokens': len(getattr(s, 'tokens', None) or []),
                        'logp_base': None, 'logp_skill': None, 'logp_delta': None}
                r['_cands'].append(cand)
                if block:
                    cand['leaked'] = _answer_leaked(block, r['reference_answer'])
                    flat.append((r, cand))
            for c in r['_cands']:
                if c['leaked'] is None:
                    c['leaked'] = False
        if flat:
            base_trajs = [_executor_answer_trajectory(r['problem'], '', r['_pseudo_solution'])
                          for r in usable]
            base_logps = _score_executor_mean_logps(ctx, base_trajs)
            base_by_id = {id(r): lp for r, lp in zip(usable, base_logps)}
            cand_trajs = [_executor_answer_trajectory(r['problem'], c['skills'], r['_pseudo_solution'],
                                                       c.get('response')) for r, c in flat]
            cand_logps = _score_executor_mean_logps(ctx, cand_trajs)
            # bugfix #14: logP 目标超长被 truncation='delete' 删掉时 lp=None → reward 地板，
            # 整组塔到 -1.0 会零梯度空转；这里显式监控 encode 失败占比（E15 的 R1 参考解尤其长）。
            _n_enc = len(base_logps) + len(cand_logps)
            enc_fail_frac = ((sum(1 for v in base_logps if v is None)
                              + sum(1 for v in cand_logps if v is None)) / _n_enc) if _n_enc else 0.0
            # leak 一律不进 reward（项目既定要求）：--logp-leak-penalty 默认 0，leaked 只做监控。
            # 该指标假阳性极高（单字符 gold 误报 ~84%，c0 实测 raw leak/rate=0.889），-1.0 量级是
            # delta 信号的 50~100 倍，会用噪声主导组内 advantage 并抬高组 std 压小其余候选。
            # format 地板（unparseable / logP 编码失败）固定 -1.0，与 leak 完全解耦、不受影响。
            floor = 1.0
            leak_pen = abs(float(getattr(args, 'logp_leak_penalty', 0.0)))
            for (r, c), lp in zip(flat, cand_logps):
                base_lp = base_by_id.get(id(r))
                c['logp_base'], c['logp_skill'] = base_lp, lp
                if base_lp is None or lp is None:
                    c['reward'] = -floor
                    continue
                delta = float(lp) - float(base_lp)
                c['logp_delta'] = delta
                # leak_pen 默认 0（leak 只做监控口径，不进 reward）。若显式开启，仍只对 >=2 字符的
                # 答案生效：usable 子集答案多为 0/1/2/4 等单字符，_answer_leaked 子串匹配在正常数学
                # 叙述里误报率 ~84%。leaked 字段全量记录，监控口径不变。
                informative = len(str(r['reference_answer']).strip()) >= 2
                c['reward'] = delta - (leak_pen if (c.get('leaked') and informative) else 0.0)
        for r in chunk:
            for c in r.get('_cands', []):
                if c['reward'] is None:
                    c['reward'] = -1.0
        _assign_advantages(chunk, args)
        grpo = _grpo_records(usable, with_rubric=False)
        has_signal = any(abs(s['advantage']) > 1e-9 for s in grpo)
        if has_signal and getattr(args, 'drop_zero_adv', False):
            grpo = [s for s in grpo if abs(s['advantage']) > 1e-9]
        n_upd, tmetrics = 0, {}
        if has_signal:
            n_upd, tmetrics = _train_batch(ctx, grpo, traj_fn=query_only_train_trajectory)
        summary = v2._chunk_summary(chunk, ci)
        # logp_rl 的 reward 不是 executor rollout 通过率；覆盖旧 BNPO summary 中与 pass 绑定的字段，
        # 避免 train_log 把“reward 非零”误读成 with-skill pass。
        summary['avg_withskill_pass'] = 0.0
        summary['candidate_withskill_pass'] = 0.0
        summary['withskill_trunc_frac'] = 0.0
        summary['termination_rate_withskill'] = 0.0
        rewards = [c['reward'] for r in usable for c in r.get('_cands', [])]
        deltas = [c['logp_delta'] for r in usable for c in r.get('_cands', [])
                  if c.get('logp_delta') is not None]
        metrics = {'signal/pseudo_gt_rate': float(len(usable)) / max(1, len(chunk)),
                   'signal/n_pseudo_gt': float(len(usable)),
                   'signal/zero_grad_frac': summary['zero_grad_frac'],
                   'logp/encode_fail_frac': (enc_fail_frac if flat else 0.0),
                   'logp/reward_mean': v2._mean(rewards),
                   'logp/reward_std': v2._std(rewards),
                   'logp/delta_mean': v2._mean(deltas),
                   'logp/delta_std': v2._std(deltas),
                   'leak/rate': summary['leak_rate'], **tmetrics,
                   **_leak_split(_cand_leak_pairs(chunk))}
        return {'n_updates': n_upd, 'summary': summary, 'metrics': metrics,
                'gen_records': v2._full_records(chunk, ci)}


class LogpGtMethod(LogpRlMethod):
    """E15: identical dense executor-logP reward, but the target S is DeepMath's external R1
    reference solution (record 'solution'), NOT an executor-sampled pseudo-GT. No executor
    rollout and no rubric audit -> much faster; every problem carrying a solution is usable."""
    needs_rubric = False

    def _prepare(self, chunk, ci):
        for r in chunk:
            r['_cands'] = []
        usable = []
        for r in chunk:
            sol = (r.get('solution') or '').strip()
            r['_pseudo_rolls'] = []
            if not sol:
                continue
            # 合成 roll：logP 目标是外部 R1 参考解，无 executor 采样；stop_reason='gt' 仅作标记，
            # 字段与 _parse_seq 输出对齐（_roll 序列化需 pred/correct/terminated/stop_reason/gen_tokens/text）。
            r['_pseudo_roll'] = {'pred': str(r['reference_answer']), 'correct': True,
                                 'terminated': True, 'stop_reason': 'gt',
                                 'gen_tokens': 0, 'text': sol}
            r['_pseudo_solution'] = sol
            r['_rubric'] = ''
            usable.append(r)
        return usable, self._skillgen(usable)


def _hinge_trunc(rolls: List[Dict[str, Any]], lo: int, budget: int = 8192) -> float:
    """Mean hinge over rollouts: 0 below ``lo`` tokens, ramps to 1.0 at the ``budget`` line.
    Probe (skill_quality_analysis.md 2026-07-29): the length->correctness link exists ONLY near
    the truncation budget (zero-truncation groups show +0.03), so penalize the danger zone only,
    smoothly, giving gradient BEFORE the hard 'length' cutoff fires."""
    if not rolls:
        return 0.0
    span = max(1, budget - lo)
    return sum(max(0.0, (int(x.get('gen_tokens') or 0) - lo) / span) for x in rolls) / len(rolls)


# 自我推翻标记词。词表由 .tmp_analysis/reward_shape_calib.py 在 E4 11155 条 rollout 上判别力筛出
# （错误密度/正确密度比值）：confusing 3.06、contradiction 2.49、mistake|error|wrong 1.84、
# alternatively 1.59。刻意排除的词："let me check" 比值 0.75（**反向**——检查一次是好行为，罚它
# 有害）、"hmm" 0.86、"recompute|again" 1.00（零信号）、"wait" 仅 1.51 且覆盖率 0.993（几乎人人都写，
# 区分度低）。
_LOOP_MARKER_RE = re.compile(
    r'\b(?:confusing|confused|contradiction|contradicts|contradictory|mistake|error|wrong'
    r'|alternatively)\b', re.I)


def _loop_density(roll: Dict[str, Any]) -> float:
    """Self-revision marker density, occurrences per 1000 generated tokens."""
    tok = int(roll.get('gen_tokens') or 0)
    if tok <= 0:
        return 0.0
    return len(_LOOP_MARKER_RE.findall(roll.get('text') or '')) / tok * 1000.0


# --- skill 套话度监控 -------------------------------------------------------------------
# ★ 定位：这两条是退化监控器（看趋势），不是质量预测器（看绝对值）。
# 2026-07-30 在 E17 的1246 个真实候选上试过 7 种定义（.tmp_analysis/generic_index_*.py），
# 题内配对对 pass 的预测力全部在 -0.033 到 +0.012 之间，都弱。根因已查清：narrative
# 文体里“元指令”和“具体动作”是同一句话里交织的（“Avoid rechecking the same rounding
# or error estimates”既是元指令又指向具体对象），句子级二分类根本不成立，调词表无法解决。
# 保留词频法的依据是方向正确：指数低组 leak=0.610 / 高组 leak=0.474，即套话越多越不给
# 具体内容。但绝对值偏高（`avoid` 在本数据上命中 1223 次，大部分在具体建议里），
# ★ 因此只能比较同一根曲线的前后变化，不能拿绝对值评判 skill 好坏。
_GENERIC_ADVICE_RE = re.compile(
    r'\b(carefully|careful|make sure|makes sure|ensure|ensuring|be sure|avoid|avoiding|'
    r'remember|keep in mind|bear in mind|double[- ]check|double[- ]checking|verify|verifying|'
    r'validate|validating|consider|considering|systematically|systematic|efficiently|efficient|'
    r'properly|correctly|accurately|appropriately|appropriate|rigorously|rigorous|'
    r'manage|managing|track|tracking|monitor|monitoring|streamline|streamlining|'
    r'focus on|stay focused|trust|commit to|committing|self[- ]correct\w*|'
    r'step by step|methodical\w*|thorough\w*|concise\w*|precise\w*|'
    r'token budget|length budget|within the budget|redundan\w*|unnecessar\w*)\b', re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENT_RE = re.compile(r'(?<=[.!?;:])\s+|\n+')
# “句中有没有数学内容”的锚点：数字 | LaTeX/算式 | 句中大写专名（定理名）。
# 不维护数学名词/动词词表：V5 试过，加了两张表反而把具体洞察误判成空话、预测力降到 0.000。
_MATH_ANCHOR_RE = re.compile(r'\d|\\[A-Za-z]+|\$|\^|_\{|=|≤|≥|≠|∈|∑|∏|√|(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}')


def _generic_advice_fraction(text: str) -> float:
    """泛泛建议词占 skill 总词数的比例；空文本返回 0。只看趋势，见上方注释。"""
    words = _WORD_RE.findall(text or '')
    if not words:
        return 0.0
    return len(_GENERIC_ADVICE_RE.findall(text)) / len(words)


def _no_math_sentence_fraction(text: str) -> float:
    """不含任何数学内容（数字/算式/定理名）的句子占比。

    名字只陈述它实际测的东西，不声称它能判“空话”—— 实测题内配对 pass 只有 -0.016，
    但它不经由 leak 中介（词频法那一条经由），两条一起看才能分开“套话变多”与“不给答案”。
    """
    sents = [s.strip() for s in _SENT_RE.split(text or '') if len(s.strip()) >= 15]
    if not sents:
        return 0.0
    return sum(1 for s in sents if not _MATH_ANCHOR_RE.search(s)) / len(sents)


def _digit_fraction(text: str) -> float:
    """数字字符占 skill 总字符数的比例（递答案/递具体中间量的代理指标）。"""
    t = text or ''
    if not t:
        return 0.0
    return sum(1 for ch in t if ch.isdigit()) / len(t)


def _skill_text_metrics(skills: List[str]) -> Dict[str, float]:
    """skill 文本面板（按任务分派）。

    math: 数字占比 / 泛泛建议词占比 / 无数学内容句占比。
    code: 数字占比与"句中有没有数学锚点"在代码域没有语义（API 名、参数、类型天然带大写与
          符号），换成 skill_contains_code_fraction —— skill 本该是方法论，写出代码围栏或成段
          def/return 就是退化成抄实现，这是代码域最该盯的那条退化曲线。泛泛建议词那条保留：
          它的词表是任务无关的（carefully / make sure / step by step ...）。
    """
    generic = v2._mean([_generic_advice_fraction(s) for s in skills])
    if v2._TASK == 'code':
        return {'train/skill_generic_advice_fraction': generic,
                'train/skill_contains_code_fraction': v2._mean(
                    [1.0 if v2.code_task.skill_has_code(s) else 0.0 for s in skills])}
    return {'train/skill_digit_fraction': v2._mean([_digit_fraction(s) for s in skills]),
            'train/skill_generic_advice_fraction': generic,
            'train/skill_no_math_sentence_fraction': v2._mean(
                [_no_math_sentence_fraction(s) for s in skills])}


# --- E18 拒绝采样第三道筛：skill 与 rubric 诊断的词频余弦 --------------------------------
# 定位：在"executor 已做对"的候选里挑与诊断内容对得上的那条，压掉两类假赢家——
# 与诊断无关的碰巧做对（含泄露式速通的残余）和与谁都不像的空泛套话。
# 刻意用去停用词的词频余弦而不是 tfidf/语义模型：可迁移性判别器一节实测"仅 tfidf"
# in-sample 0.983 / OOS 0.541 是纯过拟合；词频余弦纯 stdlib、确定性、可离线复算。
_SIM_STOPWORDS = frozenset(
    'the a an and or of to in is are be for with that this it on as by from at not no was '
    'were will would can could should may might do does did have has had you your we they '
    'he she its if then than so but into over under out up down when where which what how '
    'why all any each more most other some such only own same very'.split())
_SIM_WORD_RE = re.compile(r"[a-z][a-z'\-]{2,}")


def _rubric_similarity(skill: str, rubric: str) -> float:
    """内容词词频余弦 ∈ [0,1]；任一侧无内容词返回 0。"""
    ca = Counter(w for w in _SIM_WORD_RE.findall((skill or '').lower())
                 if w not in _SIM_STOPWORDS)
    cb = Counter(w for w in _SIM_WORD_RE.findall((rubric or '').lower())
                 if w not in _SIM_STOPWORDS)
    if not ca or not cb:
        return 0.0
    dot = float(sum(v * cb[k] for k, v in ca.items() if k in cb))
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _efficiency_terms(rolls: List[Dict[str, Any]], *, budget: int, len_lo: int, len_pow: float,
                      alpha_len: float, beta_loop: float, loop_lo: float, loop_hi: float
                      ) -> Tuple[float, float, Dict[str, float]]:
    """Per-rollout efficiency factor, aggregated. Returns (mean_score, mean_inefficiency, diag).

    Calibration (.tmp_analysis/reward_shape_calib.py, E4 11155 rollouts + DeepMath r1_solution_1):
      * Length is NOT harmful per se on this dataset. P(correct | tokens) is flat at 0.96-0.98 up
        to 5500 tokens, dips to 0.932/0.903 at 5500-7500, then collapses to 0.226 past 7500 — and
        the same shape holds inside every difficulty stratum. Non-truncated long rollouts still
        pass at ~0.93, so the damage comes from hitting the wall, not from being long.
      * The GT reference solutions (r1_solution_1) are LONGER than the model's correct answers
        (p50 4377 vs 3444 tokens; GT p90 9816 already exceeds the 8192 budget). "Shorter is
        better" is empirically false here, so a monotone-from-zero length penalty would tax the
        median GOOD answer — hence the dead zone below ``len_lo`` (人工拍板 A, 2026-07-29).
      * Convex ramp (``len_pow`` > 1) concentrates the penalty in the last ~1300 tokens before the
        budget, matching the flat-then-cliff damage curve while still making the marginal penalty
        grow with length.
      * Marker density has a real but small INDEPENDENT effect: raw dose-response is pass
        0.908 -> 0.412 across density 0-2 -> 6-9, but inside a fixed token band it shrinks to
        0.976 -> 0.931 and 0.971 -> 0.851, i.e. ~85% of the raw effect is just "longer outputs
        mechanically contain more markers". Hence ``beta_loop`` is deliberately small.

    Composition is multiplicative per rollout (人工拍板): eff = (1-a*len_pen)*(1-b*loop_pen),
    score = correct * eff. Per-rollout (not per-candidate) so a short correct rollout is never
    punished for a sibling rollout that burned the budget.
    """
    if not rolls:
        return 0.0, 0.0, {'len_pen': 0.0, 'loop_pen': 0.0, 'eff': 1.0, 'loop_density': 0.0}
    span = max(1, budget - len_lo)
    d_span = max(1e-9, loop_hi - loop_lo)
    s_sum = ineff_sum = lp_sum = mp_sum = eff_sum = dens_sum = 0.0
    for x in rolls:
        tok = int(x.get('gen_tokens') or 0)
        len_pen = min(1.0, (max(0, tok - len_lo) / span) ** len_pow)
        dens = _loop_density(x)
        loop_pen = min(1.0, max(0.0, (dens - loop_lo) / d_span))
        eff = (1.0 - alpha_len * len_pen) * (1.0 - beta_loop * loop_pen)
        s_sum += eff if x.get('correct') else 0.0
        ineff_sum += 1.0 - eff
        lp_sum += len_pen
        mp_sum += loop_pen
        eff_sum += eff
        dens_sum += dens
    n = float(len(rolls))
    diag = {'len_pen': lp_sum / n, 'loop_pen': mp_sum / n,
            'eff': eff_sum / n, 'loop_density': dens_sum / n}
    return s_sum / n, ineff_sum / n, diag


class PassrateHingeMethod(TrainMethod):
    """E16 (view B, query-only): the data-driven closure of the reward probe.

    Per chunk: (1) baseline greedy solve (T=0) to measure each problem's no-skill executor
    output length, keep only the danger band ``base_tok > --base-tok-floor`` (probe: base_tok
    vs skill lift +0.62 — the strongest problem filter; a soft floor keeps >=TRAIN_DP problems
    so a chunk never fully empties). (2) query-only skill-gen, N candidates. (3) score each
    parseable skill over M=``reward_rollouts`` executor rollouts at T=``reward_temperature``
    with a per-rollout multiplicative efficiency factor (see ``_efficiency_terms``):

        eff_i  = (1 - alpha_len * len_pen_i) * (1 - beta_loop * loop_pen_i)
        reward = mean_i(correct_i * eff_i) - kappa * mean_i(1 - eff_i)
        unparseable -> -1.0 floor

    The ``- kappa * mean(1 - eff)`` tail is what keeps FAILING candidates separable: a pure
    product would send every wrong candidate back to 0, which is exactly E4's pathology (format
    failure / burned budget / wrong method all collapsing onto one reward value).

    Coefficient sizing: the total deduction ``(1 + kappa) * (1 - eff)`` is kept under ONE
    pass_rate quantum (1/M), and ``reward = max(reward, pass_rate - 1/M)`` enforces that as a
    hard guard. Rationale: pass_rate is what eval measures, so the efficiency signal may break
    ties but must never rank "solved it once" below "never solved it". The absolute coefficient
    size barely matters — in an all-fail group every reward is -kappa*(1-eff) and A=(R-mean)/std
    rescales that spread back to unit magnitude, so the CURVE SHAPE, not the scale, is the signal.

    leak 不参与 reward（项目既定要求；``--reward-leak-gate`` 默认 0），只走监控口径。

    Group-relative advantage + BNPO on the query-only trajectory (identical to bnpo/logp).
    """
    needs_rubric = False

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        # bugfix #13: 长度死区 len_lo 不随 --max-tokens 联动；lo >= budget 时惩罚恒 0，静默失效。
        # 按标定比例（5500/8192）自动缩放并告警。
        lo = int(getattr(ctx.args, 'reward_trunc_lo', 5500) or 5500)
        if lo >= ctx.args.max_tokens:
            new_lo = max(1, int(ctx.args.max_tokens * 5500 / 8192))
            sys.stderr.write(f'[ablate] WARNING: --reward-trunc-lo {lo} >= --max-tokens '
                             f'{ctx.args.max_tokens} disables the length penalty; '
                             f'rescaled to {new_lo}.\n')
            ctx.args.reward_trunc_lo = new_lo

    def _score_candidates(self, kept, prompts):
        """skill-gen -> M-rollout with-skill solve -> 效率加权 reward（不做 advantage）。

        从 step() 里抽出只为了让 E17(ReflexionMethod) 在不复制 reward 代码的前提下换掉 prompt
        （query-only -> query+rubric）。行为与抽出前逐字一致；prompts 与 kept 同序同长。
        """
        ctx = self.ctx
        args = ctx.args
        M = max(1, int(getattr(args, 'reward_rollouts', 8) or 8))
        # `or 0.5` 会把显式的 0.0 当成“未设”静默提到 0.5（SEAM 口径 m=1/T=0 因此根本
        # 设不进来）。改成 None 判定：未设才用默认。已跑完的 E16 显式传 0.5，行为不变。
        _t = getattr(args, 'reward_temperature', None)
        temp = 0.5 if _t is None else float(_t)
        alpha_len = abs(float(getattr(args, 'reward_trunc_penalty', 0.12)))
        len_lo = int(getattr(args, 'reward_trunc_lo', 5500) or 5500)
        len_pow = max(1.0, float(getattr(args, 'reward_len_pow', 2.0) or 2.0))
        beta_loop = abs(float(getattr(args, 'reward_loop_penalty', 0.04)))
        loop_lo = float(getattr(args, 'reward_loop_lo', 2.0))
        loop_hi = float(getattr(args, 'reward_loop_hi', 9.0))
        kappa = abs(float(getattr(args, 'reward_ineff_kappa', 0.10)))
        # 不可反转护栏：总扣分封顶在一个 pass_rate 量子（1/M）以内，保证"做对过一次"永远排在
        # "一次没做对"之前。量级校验：max 扣分 = (1 + kappa) * (1 - eff_min)。
        pen_cap = 1.0 / M - 1e-6
        leak_gate = abs(float(getattr(args, 'reward_leak_gate', 0.0)))  # 默认 0：leak 只做监控
        sg = _run_samples(ctx.skill_sampler, list(prompts),
                          args.n_skills, args.skill_max_tokens, ctx.skill_dp,
                          temperature=args.skill_gen_temperature, top_p=args.skill_gen_top_p,
                          top_k=args.skill_gen_top_k)
        flat = []
        for r, seqs in zip(kept, sg):
            for s in seqs or []:
                resp = _clean_text(getattr(s, 'decoded', '') or '')
                block = _extract_skill(resp) or ''
                cand = {'skills': block, 'response': resp, 'parseable': bool(block),
                        'leaked': None, 'with_pass': None, 'reward': None, 'rolls': [],
                        'advantage': 0.0, 'kept': False,
                        'skillgen_stop': getattr(s, 'stop_reason', None),
                        'skillgen_tokens': len(getattr(s, 'tokens', None) or []),
                        'trunc_pen': None, 'pass_rate': None,
                        'loop_pen': None, 'eff': None, 'loop_density': None}
                r['_cands'].append(cand)
                if block:
                    cand['leaked'] = _answer_leaked(block, r['reference_answer'])
                    flat.append((r, cand))
        # M-rollout with-skill solve (T>0) -> per-rollout efficiency-weighted reward
        if flat:
            solve = _run_samples(
                ctx.base_sampler,
                # V2 fix: pass the raw response like v2 process_chunk (seam nesting; v2-mode no-op)
                [build_skill_solve_prompt(r['problem'], c['skills'], c.get('response')) for r, c in flat],
                M, args.max_tokens, ctx.base_dp, temperature=temp)
            # 判分批量化（code 任务：跑单测的子进程必须并行，见 v2._parse_many）
            pairs, spans = [], []
            for (r, _c), seqs in zip(flat, solve):
                start = len(pairs)
                pairs.extend((s, r['reference_answer']) for s in (seqs or []))
                spans.append((start, len(pairs)))
            judged = v2._parse_many(pairs)
            for (r, c), (a, b) in zip(flat, spans):
                rolls = judged[a:b] or [_empty_roll()]
                c['rolls'] = rolls
                pr = sum(1.0 for x in rolls if x['correct']) / len(rolls)
                score, ineff, diag = _efficiency_terms(
                    rolls, budget=args.max_tokens, len_lo=len_lo, len_pow=len_pow,
                    alpha_len=alpha_len, beta_loop=beta_loop, loop_lo=loop_lo, loop_hi=loop_hi)
                c['pass_rate'], c['with_pass'] = pr, pr
                c['trunc_pen'] = diag['len_pen']   # 名字保留，语义为长度惩罚（swanlab 面板连续）
                c['loop_pen'], c['eff'] = diag['loop_pen'], diag['eff']
                c['loop_density'] = diag['loop_density']
                reward = score - kappa * ineff
                # 护栏：总扣分不得超过一个 pass 量子，否则会反转 pass_rate 排序
                reward = max(reward, pr - pen_cap)
                informative = len(str(r['reference_answer']).strip()) >= 2
                if c['leaked'] and informative and leak_gate > 0:
                    reward -= leak_gate
                c['reward'] = reward
        for r in kept:
            for c in r['_cands']:
                if c['reward'] is None:  # unparseable / no rolls -> format floor
                    c['reward'] = -1.0

    def _reward_panel(self, kept) -> Dict[str, float]:
        """reward / 效率面板（E16 与 E17 共用，面板曲线口径保持一致）。"""
        def g(k):
            return [c[k] for r in kept for c in r['_cands'] if c.get(k) is not None]
        rewards = g('reward')
        return {'acc/pass_rate_mean': v2._mean(g('pass_rate')),
                'term/trunc_pen_mean': v2._mean(g('trunc_pen')),   # = 长度惩罚 len_pen
                'term/loop_pen_mean': v2._mean(g('loop_pen')),
                'term/eff_mean': v2._mean(g('eff')),
                'term/loop_density_mean': v2._mean(g('loop_density')),
                'reward/mean': v2._mean(rewards), 'reward/std': v2._std(rewards)}

    def _gen_records(self, kept, ci) -> List[Dict[str, Any]]:
        """v2._full_records 加上 E16/E17 特有的字段（pass_rate / 各惩罚项 / base_* / rubric）。"""
        gen_records = v2._full_records(kept, ci)
        kept_by_id = {r.get('data_id', ''): r for r in kept}
        for gr in gen_records:
            r = kept_by_id.get(gr.get('data_id', ''))
            if r is not None:
                gr['base_tok'] = r['_base_tok']
                gr['base_correct'] = r['_base_correct']
                if r.get('_base_stop') is not None:
                    gr['base_stop'] = r['_base_stop']
                if r.get('_rubric') is not None:
                    gr['rubric'] = r['_rubric']
            for gc, c in zip(gr.get('candidates', []), (r['_cands'] if r else [])):
                gc['pass_rate'] = c.get('pass_rate')
                gc['trunc_pen'] = c.get('trunc_pen')
                gc['loop_pen'] = c.get('loop_pen')
                gc['eff'] = c.get('eff')
                gc['loop_density'] = c.get('loop_density')
        return gen_records

    def step(self, chunk, ci):
        ctx = self.ctx
        args = ctx.args
        # 1) baseline (no-skill) greedy solve -> base_tok danger-band filter
        base_rolls = _bare_solve(ctx, chunk)
        for r, br in zip(chunk, base_rolls):
            r['_cands'] = []
            r['_base_tok'] = int(br.get('gen_tokens') or 0)
            r['_base_correct'] = bool(br['correct'])
        floor_tok = int(getattr(args, 'base_tok_floor', 5000) or 0)
        if floor_tok > 0:
            kept = [r for r in chunk if r['_base_tok'] > floor_tok]
            min_keep = max(TRAIN_DP, 4)
            if len(kept) < min_keep:  # soft floor: never waste a whole chunk on a thin draw
                kept = sorted(chunk, key=lambda r: r['_base_tok'], reverse=True)[:min_keep]
        else:
            kept = list(chunk)
        # 2+3) query-only skill-gen -> M-rollout with-skill solve -> 效率加权 reward
        self._score_candidates(kept, [_skillgen_prompt(r['problem']) for r in kept])
        _assign_advantages(kept, args)
        grpo = _grpo_records(kept, with_rubric=False)
        has_signal = any(abs(s['advantage']) > 1e-9 for s in grpo)
        if has_signal and getattr(args, 'drop_zero_adv', False):
            grpo = [s for s in grpo if abs(s['advantage']) > 1e-9]
        n_upd, tmetrics = 0, {}
        if has_signal:
            n_upd, tmetrics = _train_batch(ctx, grpo, traj_fn=query_only_train_trajectory)
        summary = v2._chunk_summary(kept, ci)
        base_toks = [r['_base_tok'] for r in chunk]
        metrics = {'signal/zero_grad_frac': summary['zero_grad_frac'],
                   'signal/n_kept': float(len(kept)),
                   'signal/kept_frac': float(len(kept)) / max(1, len(chunk)),
                   'signal/base_tok_mean': v2._mean(base_toks),
                   'signal/base_correct_frac': v2._mean([1.0 if r['_base_correct'] else 0.0 for r in chunk]),
                   **self._reward_panel(kept),
                   'leak/rate': summary['leak_rate'], **tmetrics,
                   **_leak_split(_cand_leak_pairs(kept))}
        gen_records = self._gen_records(kept, ci)
        # bugfix #15: 被 base_tok 门槛筛掉的题也落盘一行精简记录，筛选器本身可审计
        kept_ids = {id(r) for r in kept}
        for r in chunk:
            if id(r) not in kept_ids:
                gen_records.append({'record_type': 'problem_dropped', 'chunk': ci,
                                    'data_id': r.get('data_id', ''),
                                    'base_tok': r['_base_tok'], 'base_correct': r['_base_correct']})
        return {'n_updates': n_upd, 'summary': summary, 'metrics': metrics,
                'gen_records': gen_records}


class _SFTFamily(TrainMethod):
    """Shared SFT accumulation + fire. Subclasses fill ``collect`` to add pool samples."""
    needs_rubric = True

    def _sft_record(self, problem, ref, data_id, skill):
        # 实测（2026-07-29，.tmp_analysis/verify_v1v2_think.py）：thinking-on 模板对这种裸
        # <skills> content 会自动注入空 think 块（<think>\n\n</think>\n\n，且计入 labels），
        # 编码后是合法的 nothink 布局，与带实质 think 的 GRPO 样本混训 token 布局兼容；
        # 副作用是把模型推向短/空 think，属设计权衡而非 bug（review #5 定案）。
        return {'problem': problem, 'reference_answer': ref, 'data_id': data_id,
                'response': f'<skills>\n{skill}\n</skills>', 'skills': skill,
                'advantage': float(self.ctx.args.sft_weight), 'sft': True}

    def collect(self, chunk) -> Tuple[List[Tuple[bool, bool]], List[Dict[str, Any]]]:
        """Roll out + fill the pool; returns (leak_pairs, gen_records)."""
        raise NotImplementedError

    def step(self, chunk, ci):
        ctx = self.ctx
        leak_pairs, gen_records = self.collect(chunk)
        ctx.pool.rebalance()  # 1:1 by the minority side, surplus DISCARDED (#18b 不积压)
        n_upd = 0
        batch_metrics: List[Dict[str, float]] = []
        for batch in ctx.pool.draw_all_ready():
            n, m = _train_batch(ctx, batch, traj_fn=query_only_train_trajectory)  # SFT: query-only (#6)
            n_upd += n
            if m:
                batch_metrics.append(m)
        # bugfix #16: 多 batch 时按键均值聚合，不再相互覆盖只留最后一批
        tmetrics = {}
        if batch_metrics:
            keys = set().union(*batch_metrics)
            tmetrics = {k: sum(bm[k] for bm in batch_metrics if k in bm)
                        / sum(1 for bm in batch_metrics if k in bm) for k in keys}
        return {'n_updates': n_upd,
                'metrics': {**{f'pool/{k}': float(x) for k, x in ctx.pool.sizes().items()},
                            **tmetrics, **_leak_split(leak_pairs)},
                'gen_records': gen_records}

    # -- shared: regenerate skills under rubric, greedy-validate, pick a 2-in-8 passer --
    def _regen_pick(self, record, diag: str, use_orig_skill: bool, orig_skill: str = ''):
        ctx = self.ctx
        args = ctx.args
        if not diag:
            return None
        prompt = (_regen_prompt(record['problem'], orig_skill, diag) if use_orig_skill
                  else rubric_skillgen_prompt(record['problem'], diag))
        sg = _run_samples(ctx.skill_sampler, [prompt], args.passatk_k, args.skill_max_tokens,
                          ctx.skill_dp, temperature=args.passatk_skill_temp,
                          top_p=args.passatk_skill_top_p)
        seqs = sg[0] if sg else []
        cands = []
        for s in seqs:
            resp = _clean_text(getattr(s, 'decoded', '') or '')
            skill = _extract_skill(resp) or ''
            if not skill or len(skill) > args.skill_char_limit:
                continue
            if _leak_blocks(skill, record['reference_answer']):  # bugfix #4: informative gate
                continue
            cands.append(skill)
        if not cands:
            return None
        solve = _run_samples(ctx.base_sampler,
                             [build_skill_solve_prompt(record['problem'], sk) for sk in cands],
                             1, args.max_tokens, ctx.base_dp, temperature=0.0)
        passers = []
        for sk, seqs2 in zip(cands, solve):
            roll2 = _parse_seq(seqs2[0], record['reference_answer']) if seqs2 else _empty_roll()
            if roll2['correct'] and roll2['terminated']:
                passers.append(sk)
        if len(passers) < args.passatk_m:
            return None
        # pick the passer closest to the length budget (short-but-not-empty floor, as in v2)
        return min(passers, key=lambda sk: abs(len(sk) - args.len_budget))


class SftMethod(_SFTFamily):
    """Plain SFT: bare wrong -> rubric regen (query+rubric, no orig skill) 2-in-8 -> pool."""
    def collect(self, chunk):
        ctx = self.ctx
        rolls = _bare_solve(ctx, chunk)
        wrong = [(r, roll) for r, roll in zip(chunk, rolls) if not roll['correct']]
        diags = _diagnose_parallel(ctx, [(_rubric_entry(r, roll), None) for r, roll in wrong])
        gen_records = []
        for (r, _roll), diag in zip(wrong, diags):
            skill = self._regen_pick(r, diag, use_orig_skill=False)
            if skill:
                ctx.pool.add(self._sft_record(r['problem'], r['reference_answer'],
                                              r.get('data_id', ''), skill), NEG)
            gen_records.append({'record_type': 'problem', 'data_id': r.get('data_id', ''),
                                'problem': r['problem'], 'regen_accepted': bool(skill)})
        return [], gen_records  # bare solve has no skill, so no leak pairs here


class ImproveSftMethod(_SFTFamily):
    """Improve-skill + SFT: first-pass 1 skill (query-only, T=0.5); correct -> positive pool
    (no leak, <=char_limit); parseable-but-wrong -> rubric regen (with orig skill) 2-in-8 ->
    negative pool; unparseable first pass is skipped (empty trajectory would only feed the
    teacher garbage). Balanced 1:1 pool, majority side discarded per chunk (#15b/#18b).

    API/GPU overlap: the wrong problems' rubric diagnoses run on background threads WHILE
    the regen sampling for previously-diagnosed problems occupies the GPU (diagnoses land
    before the first regen finishes, so the loop below never blocks on the API)."""
    def collect(self, chunk):
        ctx = self.ctx
        args = ctx.args
        # first-pass ONE skill per problem (query-only, improve temperature), greedy-solve
        sg = _run_samples(ctx.skill_sampler, [_skillgen_prompt(r['problem']) for r in chunk],
                          1, args.skill_max_tokens, ctx.skill_dp,
                          temperature=args.improve_skill_temperature)
        first = []
        for r, seqs in zip(chunk, sg):
            resp = _clean_text(getattr(seqs[0], 'decoded', '') or '') if seqs else ''
            first.append((r, _extract_skill(resp) or ''))
        flat = [(r, sk) for r, sk in first if sk]
        rolls_by = {}
        if flat:
            solve = _run_samples(ctx.base_sampler,
                                 [build_skill_solve_prompt(r['problem'], sk) for r, sk in flat],
                                 1, args.max_tokens, ctx.base_dp, temperature=0.0)
            for (r, sk), seqs in zip(flat, solve):
                rolls_by[id(r)] = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
        leak_pairs = [(bool(_answer_leaked(sk, r['reference_answer'])), rolls_by[id(r)]['correct'])
                      for r, sk in flat if id(r) in rolls_by]
        wrong = [(r, sk, rolls_by[id(r)]) for r, sk in flat
                 if id(r) in rolls_by and not rolls_by[id(r)]['correct']]
        # launch ALL diagnoses on threads first, then regen serially on GPU as they land
        diag_pool = ThreadPoolExecutor(max_workers=max(1, min(args.rubric_workers, len(wrong) or 1)))
        futs = [diag_pool.submit(
            lambda e=_rubric_entry(r, roll), s=sk:
            (ctx.rubric_cache.get_or_diagnose(e, ctx.checker, skill=s) or '')
            if ctx.rubric_cache else '') for r, sk, roll in wrong]
        gen_records = []
        try:
            for r, sk in flat:
                roll = rolls_by.get(id(r))
                if roll is None:
                    continue
                if roll['correct']:
                    # positive seed: first-pass skill that worked, no leak, within char limit
                    if len(sk) <= args.skill_char_limit \
                            and not _leak_blocks(sk, r['reference_answer']):
                        ctx.pool.add(self._sft_record(r['problem'], r['reference_answer'],
                                                      r.get('data_id', ''), sk), POS)
                    gen_records.append({'record_type': 'problem', 'data_id': r.get('data_id', ''),
                                        'problem': r['problem'], 'first_correct': True, 'skill': sk})
            for (r, sk, roll), fut in zip(wrong, futs):
                # negative seed: rubric regen conditioned on the FAILED first-pass skill
                skill = self._regen_pick(r, fut.result(), use_orig_skill=True, orig_skill=sk)
                if skill:
                    ctx.pool.add(self._sft_record(r['problem'], r['reference_answer'],
                                                  r.get('data_id', ''), skill), NEG)
                gen_records.append({'record_type': 'problem', 'data_id': r.get('data_id', ''),
                                    'problem': r['problem'], 'first_correct': False,
                                    'skill': sk, 'regen_accepted': bool(skill)})
        finally:
            diag_pool.shutdown(wait=False)
        return leak_pairs, gen_records


def _answered(roll: Dict[str, Any]) -> float:
    """"这条 rollout 到底交没交出一个可判的答案"。

    math 看 ``<answer>`` / ``\\boxed``；code 域这两个标记恒不出现，必须换成"抽出了可解析的代码
    块"（judge_many 把 no_code 记在 kind 里）。不换的话该特征在 code run 上组内方差恒为 0，
    observe 会整组跳过 —— 面板上信号最强的那条曲线（E17 实测 cum_sigma 23.70）会静默消失。
    """
    if v2._TASK == 'code':
        return 1.0 if (roll.get('code') and roll.get('kind') != 'no_code') else 0.0
    text = roll.get('text') or ''
    return 1.0 if ('<answer>' in text or '\\boxed' in text) else 0.0


class SnrProbe:
    """在线可学信号探针：把组内 advantage 投影到 skill 文本特征上，逐 chunk 上报累积证据。

    为什么需要它（E16 事后分析的直接产物，见 .tmp_analysis/batch_size_math.py）：
    E16 的 reward 在【结果层】极其确定——"哪个候选让 executor 收住了"每组 SNR 2.25、97% 的组
    方向一致；但同一个信号投影到任何【skill 文本特征】上，SNR 塌到 0.046、同向组占比 0.481
    （= 抛硬币）。信息是在"从结果归因到文本"这一步丢的，损失约 50 倍。后果是：整个 50 步 run
    在唯一有内容的方向上只累积到 sqrt(1600) x 0.047 ~ 1.9 sigma，连显著性门槛都没到，而策略却
    以恒定步长（GRPO 组内标准化让 mean|a| 恒为 0.74，信号退化成噪声时步长不会变小）持续扩散
    离开初始分布——而"冻结 executor 能在预算内收住"恰恰是初始分布自带的脆弱属性。

    所以判断一个新臂值不值得跑满 50 步，看的不是 reward/mean（它被 parse 地板的构成变化掩盖，
    E16 实测总均值 +0.026 = parse 构成 +0.160 + 层内 -0.134），而是这里的 cum_sigma：
        cum_sigma = sqrt(N_groups) * |mean(g)| / std(g)
    g 是每组 advantage 与组内标准化目标量的协方差。跑 10-15 个 chunk 就能看出 rubric 条件下的
    文本层信号是否比 0.046 高一个量级；不高就该停，省 80% 卡时。
    """

    # (键, 取值函数, 期望方向)；dir=-1 表示"我们希望 reward 压低该量"，上报时已翻正号，
    # 所以 mu>0 一律读作"reward 在往我们想要的方向推"。skill_digits 也是 -1：E16 的主导机制
    # 是答案中继（obey|断言对 = 0.99），所以"skill 里的数字变多"是需要报警的方向、不是
    # 鼓励的方向；先前写 +1 会让面板上的正值被误读成好消息。
    # 注：决策变量 cum_sigma 用 |mean|，与方向约定无关，只有 mu / agree 的可读性依赖它。
    TARGETS = (
        ('skill_chars', lambda c: float(len(c.get('skills') or '')), -1),
        ('skill_digits', lambda c: float(sum(ch.isdigit() for ch in (c.get('skills') or ''))), -1),
        ('think_tokens', lambda c: float(c.get('skillgen_tokens') or 0), -1),
        ('exec_trunc', lambda c: _roll_mean(c, lambda x: 1.0 if x.get('stop_reason') == 'length'
                                           else 0.0), -1),
        ('exec_tok', lambda c: _roll_mean(c, lambda x: float(int(x.get('gen_tokens') or 0))), -1),
        # ---- 2026-07-30 新增 4 条。先用 .tmp_analysis/snr_feature_scan.py 在 E17 的 168 组/
        # 1344 候选上扫过 13 个备选（同一口径），只留下 cum_sigma 过 5 且不与现有项重复的。
        # 被剔掉的（全部 < 3）：exec_tok_spread 2.87、skill_mean_sentence_len 2.67、
        # skill_hedge_frac 2.23、skill_n_sentences 2.04、skill_action_verb_frac 1.72、
        # skill_proper_noun_frac 1.54、skill_step_marker_frac 0.35。
        # ★ exec_answered 实测 23.70、agree 0.966，比旧冠军 exec_trunc(22.21) 还高：把“reward
        #   到底在推什么”说得比截断率更直——不是“别写太长”而是“把答案写出来”。dir=+1。
        #   （判据按 _TASK 分派，见 _answered）
        ('exec_answered', lambda c: _roll_mean(c, _answered), +1),
        # loop_pen 就在 reward 公式里（beta=0.04）却一直没进 SNR 面板，那一项到底有没起作用
        # 之前看不到。实测 10.70 / agree 0.789。复用 _loop_density（与 reward 同一个函数）。
        ('exec_loop_density', lambda c: _roll_mean(c, lambda x: _loop_density(x)), -1),
        # 唯一直接量化“reward 在多大程度上奖励泄露”的 SNR。实测 8.80 / agree 0.819 / mu=-0.424。
        # 只在组内 leak 有差异的组（实测 83/168）取到值，但那正是要看的那些组。
        # ★ 这是监控形态，不得进 reward（项目既定要求）。dir=-1。
        ('skill_leaked', lambda c: (1.0 if c.get('leaked') else 0.0), -1),
        # 比现有的绝对数字数 skill_digits(10.11) 剔掉了 skill 长度混杂；两条都留，差值能
        # 分开“数字变多”与“skill 变长”。实测 6.55 / agree 0.721。
        # 空 skill 返回 0.0（不是 None）：与 skill_chars 口径一致，否则 parse 率随训练上升
        # 会让本特征的取样面系统漂动（正是 observe 里那段注释警告的假趋势源）。
        ('skill_digit_fraction', lambda c: _digit_fraction(c.get('skills') or ''), -1),
    )

    def __init__(self):
        # 每个 target 一个累积器：(n, sum(g), sum(g^2), 正号计数)
        self._acc: Dict[str, List[float]] = {k: [0.0, 0.0, 0.0, 0.0] for k, _, _ in self.TARGETS}

    def observe(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """records = 本次更新真正参与训练的 per-problem 记录（带 '_cands'）。返回 swan 标量。"""
        out: Dict[str, float] = {}
        for key, fn, direction in self.TARGETS:
            gs = []
            for r in records:
                cs = [c for c in r.get('_cands', []) if c.get('reward') is not None
                      and c.get('advantage') is not None]
                # 只丢"该特征取不到值"的候选，不丢整组。不可解析候选没有 rollout，exec_* 取值
                # 为 None；而 parse 率会随训练上升（E16 实测：全可解析组占比 c0-9 0.757 ->
                # c40-49 0.979），所以"任一 None 丢整组"会让取样面随时间系统性扩大 24pp
                # —— 那本身就是一个假趋势源，而 cum_sigma 正是要用来判断趋势的。
                # （g = sum(a_i z_i)/n 对 a 加常数不变，z 在存活子集上重新中心化后仍无偏。）
                pairs = [(c, fn(c)) for c in cs]
                pairs = [(c, v) for c, v in pairs if v is not None]
                if len(pairs) < 2:
                    continue
                # advantage 全 0 的组（组内 reward 无差异，_assign_advantages 已置 0）不携带方向
                # 信息；计入只会把均值往 0 拉、同时虚增 n，使 cum_sigma 系统偏低。
                if all(abs(float(c['advantage'])) < 1e-12 for c, _ in pairs):
                    continue
                vals = [v for _, v in pairs]
                mu = sum(vals) / len(vals)
                var = sum((v - mu) ** 2 for v in vals) / len(vals)
                if var < 1e-12:      # 组内该特征无差异 -> 这一组对该方向不提供信息
                    continue
                sd = var ** 0.5
                g = sum(float(c['advantage']) * (v - mu) / sd for c, v in pairs) / len(pairs)
                gs.append(direction * g)
            if not gs:
                continue
            a = self._acc[key]
            a[0] += len(gs)
            a[1] += sum(gs)
            a[2] += sum(x * x for x in gs)
            a[3] += sum(1.0 for x in gs if x > 0)    # 只记正号数，与漂动的运行均值解耦
            n_cum = a[0]
            mean_cum = a[1] / n_cum
            # 累积 std（总体口径；n 已达数百，与样本口径无实质差别）
            var_cum = max(a[2] / n_cum - mean_cum ** 2, 0.0)
            sd_cum = var_cum ** 0.5
            out[f'snr/{key}_mu'] = sum(gs) / len(gs)             # 本 chunk 的每组均值
            out[f'snr/{key}_mu_cum'] = mean_cum
            out[f'snr/{key}_snr_cum'] = (abs(mean_cum) / sd_cum) if sd_cum > 1e-12 else 0.0
            # 同向组占比：取两个符号桶的多数侧，不依赖当时的运行均值（旧实现拿
            # mean_cum 做参系，早期均值不稳时会把同一批组判到不同侧）。
            out[f'snr/{key}_agree_cum'] = max(a[3], n_cum - a[3]) / n_cum
            # ★ 决策变量：整个 run 至今在该方向上累积的证据（sigma）。E16 全程只到 1.9。
            out[f'snr/{key}_cum_sigma'] = ((n_cum ** 0.5) * abs(mean_cum) / sd_cum
                                           if sd_cum > 1e-12 else 0.0)
            out[f'snr/{key}_n_groups'] = n_cum
        return out


def _roll_mean(cand: Dict[str, Any], fn: Callable[[Dict[str, Any]], float]) -> Optional[float]:
    rolls = cand.get('rolls') or []
    if not rolls:
        return None
    return sum(fn(x) for x in rolls) / len(rolls)


class ReflexionMethod(PassrateHingeMethod):
    """E17 —— Reflexion 条件化臂：只在【裸 executor 做错】的题上，用 rubric 生成 skill 并训练。

    与父类 PassrateHingeMethod（E16）完全共享 reward 形状（M-rollout pass_rate x 效率加权、
    死区 5500 起的二次凸长度惩罚、循环惩罚、不可反转护栏、leak 只监控），三处结构差异：

    1) 选题：父类按 base_tok 危险带筛（floor=5000），本类按【裸解错误】筛，并把批量对齐到
       恰好 --reflexion-k 道题。对齐的理由：每次更新的组数必须恒定，否则步长与噪声逐 chunk
       变化，SnrProbe 的累积证据和任何趋势读数都会被批量抖动污染。
       实现上不动态多抽（MethodContext 拿不到 ProblemPool，且 trainer 的断点恢复靠"重放 N 次
       等长 draw"，变长抽取会破坏恢复），而是把 chunk_size 放大到 ~5K、裸解后逐批取错题
       并对 rubric 缺失做回填，直到凑满 K；真凑不够时用现有的全部并上报 k_short。
       标定（.tmp_analysis/e17_param_calib.py，E16 落盘）：全量题池裸错率 0.329（527/1600），
       E16 每次更新实际 23.46 组 / 全程 1173 组，所以 K=24 才能追平 E16 的证据量（累积
       证据 ∝sqrt(N)，E16 全程在文本层只累到 1.9 sigma，再砍组数就没判别力了）；
       chunk=128 时 P(错题<24) = 0.012%。⭐ 不要用 0.446，那是 base_tok>5000 筛选后
       子集的错误率（偏难），而本臂 floor=0；用它会把 chunk 低估到 96（P=3.6%）。
    2) skill-gen 走 view A：prompt = rubric_skillgen_prompt(problem, diag)，训练轨迹相应换成
       rubric_train_trajectory（否则会在 query-only 轨迹上训一个 query+rubric 分布下采出的
       response，与 E6 的已知缺陷同型）。rubric API 缺失（失败/坏缓存）的题一律丢弃而不降级
       成 query-only —— 本臂的唯一自变量就是 rubric，降级样本会把它稀释掉。
    3) base_tok_floor 强制视为 0。⚠️ 但要知道这几乎不起作用：实测全量题池的 527 道错题里
       96.96% 是没写完（base_tok>=8192），floor=5000 筛选后也只是 97.71% —— 截断是题目
       （level>=6 配 8192 预算）造成的，不是筛选造成的。所以 rubric 在 ~97% 的题上只能说
       "你超预算了"，signal/wrong_trunc_frac 会直接开在 0.97。

    API/GPU 重叠：rubric 诊断是纯 API，父类的 GPU 路径在它之后才开始，所以这里先起线程池发
    诊断、同时不做别的 GPU 工作（本臂没有 B 线可以并行），诊断落地后再进 skill-gen。
    """

    needs_rubric = True

    def __init__(self, ctx: MethodContext):
        super().__init__(ctx)
        # 题集由【裸解错】定义，base_tok 危险带筛选强制关闭（用户 2026-07-29 拍板）：错题集
        # 必须保留"推理错 + 没写完"的混合，否则就只剩长尾截断题、测不到方法修正。
        # 在这里而不是在 shell 里置 0，是为了让 config 指纹（trainer 在 build_method 之后才
        # 落盘）记录真实生效值，而不是一个未被读取的默认 5000。
        if int(getattr(ctx.args, 'base_tok_floor', 0) or 0):
            sys.stderr.write('[ablate] reflexion: --base-tok-floor forced to 0 (problem set is '
                             'defined by bare-solve failure, not by output length).\n')
        ctx.args.base_tok_floor = 0
        self.snr = SnrProbe()

    def step(self, chunk, ci):
        ctx = self.ctx
        args = ctx.args
        K = max(1, int(getattr(args, 'reflexion_k', 16) or 16))
        # 1) 裸解全 chunk，挑错题并对齐到恰好 K 道
        base_rolls = _bare_solve(ctx, chunk)
        for r, br in zip(chunk, base_rolls):
            r['_cands'] = []
            r['_base_tok'] = int(br.get('gen_tokens') or 0)
            r['_base_correct'] = bool(br['correct'])
            r['_base_stop'] = br.get('stop_reason')
        wrong = [(r, br) for r, br in zip(chunk, base_rolls) if not br['correct']]
        # 2) rubric 诊断（纯 API，线程并行；缓存键 = data_id + 裸解轨迹，跳臂全局缓存）。
        # 逐批回填到恰好 K 道：直接 wrong[:K] 会让 rubric 缺失把本 chunk 的组数打到 K 以下，
        # 而组数恒定是本臂的硬要求（否则步长、噪声底与 SnrProbe 的累积证据全跟着抖）。
        # 回填只花网络时间不占 GPU；E6/E7 实测缺失率 0，正常路径上循环只进一轮。
        picked, dropped_no_rubric, cursor = [], 0, 0
        while len(picked) < K and cursor < len(wrong):
            take = wrong[cursor:cursor + (K - len(picked))]
            cursor += len(take)
            diags = _diagnose_parallel(ctx, [(_rubric_entry(r, br), None) for r, br in take])
            for (r, br), diag in zip(take, diags):
                r['_rubric'] = diag or ''
                if diag:
                    picked.append((r, br, diag))
                else:
                    dropped_no_rubric += 1     # 不降级成 query-only：rubric 是唯一自变量
        k_short = max(0, K - len(picked))       # 错题不够 + rubric 全打水两种原因合计
        # 组数恒定是本臂的硬要求（步长、噪声底与 SnrProbe 的累积证据都跟着它抖）。原来靠
        # signal/k_short + signal/n_rubric_missing 两条面板指标暴露，面板精简后改走 stderr，
        # 否则这个不变量会变成静默失败。
        if k_short or dropped_no_rubric:
            sys.stderr.write(f'[E17] c{ci}: WARNING 组数 {len(picked)}/{K}'
                             f'（缺 {k_short}；其中 rubric 拉不到 {dropped_no_rubric} 道）\n')
        kept = [r for r, _br, _d in picked]
        # 3) 复用父类的 skill-gen -> M-rollout -> 效率加权 reward -> advantage 流水线
        if kept:
            self._score_candidates(kept, [rubric_skillgen_prompt(r['problem'], d)
                                          for r, _br, d in picked])
            _assign_advantages(kept, args)
        grpo = _grpo_records(kept, with_rubric=True)
        has_signal = any(abs(s['advantage']) > 1e-9 for s in grpo)
        if has_signal and getattr(args, 'drop_zero_adv', False):
            grpo = [s for s in grpo if abs(s['advantage']) > 1e-9]
        n_upd, tmetrics = 0, {}
        if has_signal:
            n_upd, tmetrics = _train_batch(ctx, grpo, traj_fn=rubric_train_trajectory)
        # 4) 指标（2026-07-30 精简）：只保留用户指定的这几条，命名不缩写。
        #    去掉的 leak/* term/* signal/* 仍全量落在 gen_records 里，离线随时可算。
        summary = v2._chunk_summary(kept, ci) if kept else {'zero_grad_frac': 1.0, 'leak_rate': 0.0}
        # ★ 两个口径必须分开（否则退化会被隐掉）：
        #   all   = 所有候选，包括解析失败的（skill 为空、reward 拍 -1.0 地板、从未跑 executor）
        #   scored= 只有真跑了 executor 的（pass_rate 不为 None）
        # 只用 scored 算 skill 文本指标会把“退化成空 skill”这个最重要的信号完全遮住
        # （v6 那轮实测空 skill 率 7.4%，全部是 skill 自己写到 8192 撞顶）；
        # reward 也必须含地板，否则面板上的 reward 不等于优化器真正看到的那个。
        all_cands = [c for r in kept for c in r['_cands']]
        scored = [c for c in all_cands if c.get('pass_rate') is not None]
        # roll 里只有 gen_tokens（见 v2._parse_seq），没有 tokens 字段。
        exec_tokens = [float(int(x.get('gen_tokens') or 0))
                       for c in scored for x in (c.get('rolls') or [])]
        skills = [c.get('skills') or '' for c in all_cands]
        # 训练题一律是裸解做错的题，所以 baseline 恒为 0；显式上报是为了让 lift 曲线自解释。
        baseline_accuracy = 0.0
        with_skill_accuracy = v2._mean([c['pass_rate'] for c in scored])
        rewards = [c['reward'] for c in all_cands if c.get('reward') is not None]
        metrics = {
            'train/baseline_accuracy': baseline_accuracy,
            'train/with_skill_accuracy': with_skill_accuracy,
            'train/lift': with_skill_accuracy - baseline_accuracy,
            'train/reward_mean': v2._mean(rewards),
            'train/reward_std': v2._std(rewards),
            'train/zero_gradient_fraction': summary['zero_grad_frac'],
            # 解析失败率：with_skill_accuracy 只在 scored 上算，这条负责把分母的变化讲出来。
            'train/skill_parse_failure_rate': (1.0 - len(scored) / len(all_cands)) if all_cands else 0.0,
            'train/skill_length_characters': v2._mean([float(len(s)) for s in skills]),
            **_skill_text_metrics(skills),
            'train/executor_length_tokens': v2._mean(exec_tokens),
            'train/executor_loop_rate': v2._mean([c['loop_pen'] for c in scored
                                                 if c.get('loop_pen') is not None]),
            **tmetrics,   # train/loss, train/grad_norm, train/lr, train/iters, train/n_samples
            **self.snr.observe(kept),   # snr/*：eval 在 R=1 下 MDE 很大，方向判断只能靠这个
        }
        gen_records = self._gen_records(kept, ci)
        kept_ids = {id(r) for r in kept}
        for r in chunk:
            if id(r) not in kept_ids:
                # drop_reason 让对齐可审计：否则 dump 里分不出"裸解对"、"超过 K 没用上"、
                # "rubric 拉不到"三种丢弃，而只有第三种是需要报警的。
                reason = ('base_correct' if r['_base_correct']
                          else 'no_rubric' if r.get('_rubric') == '' else 'beyond_k')
                gen_records.append({'record_type': 'problem_dropped', 'chunk': ci,
                                    'data_id': r.get('data_id', ''), 'drop_reason': reason,
                                    'base_tok': r['_base_tok'], 'base_correct': r['_base_correct'],
                                    'base_stop': r['_base_stop']})
        return {'n_updates': n_upd, 'summary': summary, 'metrics': metrics,
                'gen_records': gen_records}


class RejectionSftMethod(TrainMethod):
    """E18 —— 拒绝采样 SFT（2026-07-30 用户拍板的 9 步方案）：

    每 chunk：① 全部 query 裸解一次（greedy T=0）判对错 → ② 错题过 rubric 诊断（缺失即丢，
    不降级）→ ③ 按 E17 的 rollout 方式：rubric 条件化 skill-gen（think 模式、T=1.0 × n_skills），
    每个 skill 让 executor 推理一次（greedy T=0）→ ④ 三道筛选一条：
        a. 只留做对的；
        b. leak 过滤：含最终答案的丢掉（用 _leak_blocks 的 >=2 字符门：裸 _answer_leaked 对
           单字符 gold 误报 ~84%，会把池饿死——SFT 家族 bugfix #4 的同一个门；超
           skill_char_limit 的一并丢）；
        c. 长度预筛：取离 len_budget 最近的前一半 → 其中与原始 rubric 词频余弦相似度最高的。
    ⑤ 胜者写进本地数据集文件 e18_sft_dataset.jsonl（append-only，含 rubric/相似度/pass 全审计字段）
    并入池 → ⑥ 池满 --e18-accumulate（16，2026-07-30 从 128 改小）条就 SFT 一次（advantage=--sft-weight=1，轨迹用
    query-only + 裸 <skills> 响应 = nothink 布局：thinking-on 模板会自动注入空 think 块，
    与 Qwen3 enable_thinking=False 的生成布局逐 token 一致，review #5 定案）。
    ⑦ _train_batch 内部 ckpt.sync_weights 把新权重推到 vLLM → ⑧ eval 在 trainer 侧：同一个
    skill_sampler vLLM 临时切 nothink 模板跑 query-only greedy eval（只换客户端编码，引擎不动）。

    与 SftMethod(E12) 的本质区别：E12 靠 rubric 重生成 2-in-8 验证入池（无拒绝排序）；
    E18 在做对的候选里再按 leak/长度/rubric 对齐度三道筛取唯一胜者，且留下可复现的
    本地数据集文件。train-with-rubric/train-query-only 的选择沿用 SFT 家族定案 #6：
    用 query-only 轨迹训，避免采集分布（query+rubric）与部署分布（query-only）错配。
    """
    needs_rubric = True

    def step(self, chunk, ci):
        ctx = self.ctx
        args = ctx.args
        # ① 裸解全 chunk（greedy T=0 单次），判对错
        base_rolls = _bare_solve(ctx, chunk)
        for r, br in zip(chunk, base_rolls):
            r['_cands'] = []
            r['_rubric'] = ''
            r['_base_correct'] = bool(br['correct'])
        wrong = [(r, br) for r, br in zip(chunk, base_rolls) if not br['correct']]
        # ② rubric 诊断（纯 API 线程并行；缺失即丢，不降级 query-only：没有诊断就没有
        # 相似度筛的参照系，与 E17「rubric 是唯一自变量」的丢弃规则同型）
        diags = _diagnose_parallel(ctx, [(_rubric_entry(r, br), None) for r, br in wrong])
        todo = []
        for (r, br), diag in zip(wrong, diags):
            r['_rubric'] = diag or ''
            if diag:
                todo.append((r, diag))
        n_rubric_missing = len(wrong) - len(todo)
        # ③ rubric 条件化 skill-gen（sampler 模板是 think-on，即用户要求的 think 模式采集）；
        # _skillgen_solve 内部对每个可解析 skill 跑 executor greedy T=0 单次，正是本臂口径。
        items = [{'record': r, 'prompt': rubric_skillgen_prompt(r['problem'], d)}
                 for r, d in todo]
        if items:
            _skillgen_solve(ctx, items, args.n_skills, temperature=args.skill_gen_temperature)
        # ④ 逐题三道筛：做对 -> 不 leak/不超长 -> 长度预筛前半 + rubric 相似度最高
        accepted = []
        sims_pool = []
        n_pass_cands = n_leak_dropped = 0
        for r, d in todo:
            passers = [c for c in r['_cands']
                       if c.get('parseable') and (c.get('with_pass') or 0) > 0]
            n_pass_cands += len(passers)
            survivors = [c for c in passers
                         if len(c['skills']) <= args.skill_char_limit
                         and not _leak_blocks(c['skills'], r['reference_answer'])]
            n_leak_dropped += len(passers) - len(survivors)
            if not survivors:
                continue
            # 长度选择：取离 len_budget 最近的前一半（至少 1 条），再在其中比相似度。
            # 两阶而非加权求和：两个量纲不同（字符距 vs 余弦），权重没法标定。
            by_len = sorted(survivors, key=lambda c: abs(len(c['skills']) - args.len_budget))
            shortlist = by_len[:max(1, (len(by_len) + 1) // 2)]
            for c in shortlist:
                c['rubric_similarity'] = _rubric_similarity(c['skills'], d)
            best = max(shortlist, key=lambda c: c['rubric_similarity'])
            best['kept'] = True
            sims_pool.append(best['rubric_similarity'])
            accepted.append({'problem': r['problem'],
                             'reference_answer': r['reference_answer'],
                             'data_id': r.get('data_id', ''),
                             'response': f"<skills>\n{best['skills']}\n</skills>",
                             'skills': best['skills'],
                             'advantage': float(args.sft_weight), 'sft': True,
                             # 审计字段（只进数据集文件，不进训练轨迹）
                             'rubric': d, 'chunk': ci,
                             'rubric_similarity': best['rubric_similarity'],
                             'skill_chars': len(best['skills']),
                             'n_candidates_passed': len(passers)})
        # ⑤ 胜者落盘本地数据集（append-only，逐 chunk 开关避免长持句柄）+ 入池
        if accepted:
            with open(os.path.join(args.output_dir, 'e18_sft_dataset.jsonl'), 'a',
                      encoding='utf-8') as f:
                for s in accepted:
                    f.write(json.dumps(s, ensure_ascii=False) + '\n')
            for s in accepted:
                # 训练样本只留 _train_trajectory 需要的键（rubric 不进 query-only 轨迹）
                ctx.pool.add({k: s[k] for k in ('problem', 'reference_answer', 'data_id',
                                                'response', 'skills', 'advantage', 'sft')}, NEG)
        # ⑥+⑦ 池满 --e18-accumulate 条即 SFT；_train_batch 内部已含 ckpt.sync_weights
        n_upd = 0
        batch_metrics: List[Dict[str, float]] = []
        for batch in ctx.pool.draw_all_ready():
            n, m = _train_batch(ctx, batch, traj_fn=query_only_train_trajectory)
            n_upd += n
            if m:
                batch_metrics.append(m)
        tmetrics = {}
        if batch_metrics:
            keys = set().union(*batch_metrics)
            tmetrics = {k: sum(bm[k] for bm in batch_metrics if k in bm)
                        / sum(1 for bm in batch_metrics if k in bm) for k in keys}
        # 指标：命名不缩写（沿用 E17 面板约定）
        n_wrong = len(wrong)
        metrics = {
            'train/pool_size': float(ctx.pool.sizes().get('pool', 0)),
            'train/accept_rate': (len(accepted) / len(todo)) if todo else 0.0,
            'train/candidate_pass_rate': (n_pass_cands / (len(todo) * args.n_skills))
                                          if todo else 0.0,
            'train/leak_or_overlength_dropped_fraction': (n_leak_dropped / n_pass_cands)
                                                          if n_pass_cands else 0.0,
            'train/selected_rubric_similarity': v2._mean(sims_pool),
            'train/selected_skill_length_characters': v2._mean(
                [float(s['skill_chars']) for s in accepted]),
            'signal/n_wrong': float(n_wrong),
            'signal/n_rubric_missing': float(n_rubric_missing),
            **tmetrics,
        }
        # gen_records：v2._full_records 不落 rubric/相似度，补上审计字段（筛选器本身可审计）
        gen_records = v2._full_records(chunk, ci)
        by_id = {r.get('data_id', ''): r for r in chunk}
        for gr in gen_records:
            r = by_id.get(gr.get('data_id', ''))
            if r is None:
                continue
            gr['base_correct'] = r.get('_base_correct')
            if r.get('_rubric'):
                gr['rubric'] = r['_rubric']
            for gc, c in zip(gr.get('candidates', []), r.get('_cands', [])):
                if c.get('rubric_similarity') is not None:
                    gc['rubric_similarity'] = c['rubric_similarity']
        return {'n_updates': n_upd, 'metrics': metrics,
                'gen_records': gen_records}


METHOD_REGISTRY: Dict[str, Callable[[MethodContext], TrainMethod]] = {
    'bnpo': BnpoMethod,
    'rl_ab': RlAbMethod,
    'rl_err': RlErrMethod,
    'opsd': OpsdMethod,
    'sft': SftMethod,
    'improve_sft': ImproveSftMethod,
    'logp_rl': LogpRlMethod,
    'logp_gt': LogpGtMethod,
    'passrate_hinge': PassrateHingeMethod,
    'reflexion': ReflexionMethod,
    'rejection_sft': RejectionSftMethod,
}


def build_method(method: str, ctx: MethodContext) -> TrainMethod:
    if method not in METHOD_REGISTRY:
        raise KeyError(f'unknown method {method!r}; valid: {sorted(METHOD_REGISTRY)}')
    return METHOD_REGISTRY[method](ctx)
