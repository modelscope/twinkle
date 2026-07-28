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
    return [(_parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll())
            for r, seqs in zip(records, out)]


def _rubric_entry(record: Dict[str, Any], roll: Dict[str, Any]) -> Dict[str, Any]:
    """Build the entry _diagnose_entry expects from a failure trajectory."""
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
        ws = _run_samples(ctx.base_sampler,
                          [build_skill_solve_prompt(it['record']['problem'], c['skills']) for it, c in flat],
                          1, args.max_tokens, ctx.base_dp, temperature=0.0)
        for (it, c), seqs in zip(flat, ws):
            roll = _parse_seq(seqs[0], it['record']['reference_answer']) if seqs else _empty_roll()
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


def _leak_split(pairs: List[Tuple[bool, bool]]) -> Dict[str, float]:
    """#10 monitoring curves: leaked&correct vs leaked&wrong rates over parseable skills.
    "泄露正确答案可接受"不等于"泄露无害"——有害的是错误数值注入，两条曲线拆开监控。"""
    n = len(pairs)
    if not n:
        return {'leak/correct_rate': 0.0, 'leak/wrong_rate': 0.0}
    return {'leak/correct_rate': sum(1 for lk, ok in pairs if lk and ok) / n,
            'leak/wrong_rate': sum(1 for lk, ok in pairs if lk and not ok) / n}


def _cand_leak_pairs(records: List[Dict[str, Any]]) -> List[Tuple[bool, bool]]:
    return [(bool(c['leaked']), bool(c['with_pass']))
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
                   'acc/withskill_pass': summary['avg_withskill_pass'],
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
        for (r, _roll), diag in zip(wrong, diags):
            r['_rubric'] = diag
            a_items.append({'record': r, 'prompt': rubric_skillgen_prompt(r['problem'], diag)})
        if a_items:
            _skillgen_solve(ctx, a_items, args.n_skills, temperature=args.skill_gen_temperature)
        _assign_advantages(chunk, args)
        a_recs = _grpo_records([r for r, _ in wrong], with_rubric=True)
        b_recs = _grpo_records(right, with_rubric=False)
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
                'metrics': {'signal/n_wrong_A': float(len(wrong)),
                            'signal/n_right_B': float(len(right)), **tmetrics,
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


class _SFTFamily(TrainMethod):
    """Shared SFT accumulation + fire. Subclasses fill ``collect`` to add pool samples."""
    needs_rubric = True

    def _sft_record(self, problem, ref, data_id, skill):
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
        n_upd, tmetrics = 0, {}
        for batch in ctx.pool.draw_all_ready():
            n, m = _train_batch(ctx, batch, traj_fn=query_only_train_trajectory)  # SFT: query-only (#6)
            n_upd += n
            tmetrics.update(m)
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
            if _answer_leaked(skill, record['reference_answer']):
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
                            and not _answer_leaked(sk, r['reference_answer']):
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


METHOD_REGISTRY: Dict[str, Callable[[MethodContext], TrainMethod]] = {
    'bnpo': BnpoMethod,
    'rl_ab': RlAbMethod,
    'rl_err': RlErrMethod,
    'opsd': OpsdMethod,
    'sft': SftMethod,
    'improve_sft': ImproveSftMethod,
}


def build_method(method: str, ctx: MethodContext) -> TrainMethod:
    if method not in METHOD_REGISTRY:
        raise KeyError(f'unknown method {method!r}; valid: {sorted(METHOD_REGISTRY)}')
    return METHOD_REGISTRY[method](ctx)
