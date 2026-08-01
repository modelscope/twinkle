# Copyright (c) ModelScope Contributors. All rights reserved.
"""E17 专用 eval：reflexion 协议 —— 只在裸 executor 做错的题上做 rubric 条件化 skill 干预。

与 v2 ``run_greedy_eval`` 的区别只有"作用域"和"skill-gen 的输入"两点：

1. baseline 正确的题**完全不动**（不生成 skill、不重跑 executor），按定义计 1.0。理由：本臂的
   命题是"reflexion 能不能救回不会的题"，正确题上的 with-skill rollout 既不提供信息、又在
   E16 里贡献了全部 -0.060 的破坏项，把它留在指标里只会用一个已知的、与命题无关的效应稀释
   趋势。因此当 rubric 无缺失时 acc 恰好等于 ``base + (1-base) * hard_rescue``，两者只差一个
   线性缩放，**主指标是 hard_rescue_rate**。
2. hard 子集的 skill-gen 输入 = query + rubric（与训练同分布），rubric 由裸解轨迹诊断得来。
   这修掉了 E6 的已知缺陷（训练在 query+rubric 分布、eval 却是 query-only）。

成本：rubric 条目的缓存键 = data_id + 裸解轨迹，而 eval baseline 是冻结+缓存的，所以同一道
eval 题在整个 run 里只会调一次 rubric API；skill-gen / executor 也只跑 hard 子集（≈40%）。

⚠️ 口径不与 E1-E16 横比（用户 2026-07-29 拍板：只看本实验自身趋势）。
"""
import sys
from typing import Any, Dict, List, Tuple

import train_skill_v2 as v2
from train_skill_v2 import (_clean_text, _extract_skill, _run_samples, build_direct_prompt,
                            build_skill_solve_prompt)

from .methods import _rubric_entry
from .rollouting import rubric_skillgen_prompt


def _baseline_rolls(base_sampler, eval_records, base_dp, args, base_cache) -> List[Dict[str, Any]]:
    """裸 greedy(T=0) 判分，走 v2 DiskCache（与 run_greedy_eval 同一缓存文件、同一键）。"""
    todo = [r for r in eval_records if v2.DiskCache.key_for(r['problem']) not in base_cache]
    if todo:
        out = _run_samples(base_sampler, [build_direct_prompt(r['problem']) for r in todo],
                           1, args.max_tokens, base_dp, temperature=0.0)
        # 批量判分：code 任务下每条判分是一个跑单测的子进程，200 道题串行要 ~7 分钟。
        rolls = v2._parse_many([(v2._first_seq(seqs), r['reference_answer'])
                                for r, seqs in zip(todo, out)])
        for r, roll in zip(todo, rolls):
            base_cache.put(v2.DiskCache.key_for(r['problem']), roll)
    return [base_cache.get(v2.DiskCache.key_for(r['problem'])) for r in eval_records]


def _diagnose(rubric_cache, checker, jobs: List[Dict[str, Any]], workers: int) -> List[str]:
    from concurrent.futures import ThreadPoolExecutor
    if not jobs or rubric_cache is None:
        return [''] * len(jobs)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        return list(ex.map(lambda e: rubric_cache.get_or_diagnose(e, checker) or '', jobs))


def _gen_skills(skill_sampler, prompts, R, skill_dp, args) -> List[List[Tuple[str, str]]]:
    """每题采 R 个 skill；返回 [(skill_block, raw_response)] * R（缺位补空串，与 v2 同）。"""
    sg_out = _run_samples(skill_sampler, prompts, R, args.skill_max_tokens, skill_dp,
                          temperature=args.eval_skill_temperature)
    per = []
    for seqs in sg_out:
        seqs = list(seqs or [])
        row = []
        for j in range(R):
            s = seqs[j] if j < len(seqs) else None
            if s is None:
                row.append(('', ''))
            else:
                sresp = _clean_text(getattr(s, 'decoded', '') or '')
                row.append((_extract_skill(sresp) or '', sresp))
        per.append(row)
    return per


def run_reflexion_eval(base_sampler, skill_sampler, eval_records, ci, rounds,
                       base_dp, skill_dp, args, base_cache, rubric_cache, checker):
    """返回 (recs, summary, metrics)，键名与 v2.run_greedy_eval 兼容（trainer 打印共用）。"""
    R = max(1, args.eval_rollouts)
    base_rolls = _baseline_rolls(base_sampler, eval_records, base_dp, args, base_cache)

    # ---- 1) 切分：baseline 正确的题按协议原样通过，不做任何 GPU 工作 ----
    hard: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    recs: List[Dict[str, Any]] = []
    head = {'record_type': 'eval_problem', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
            'protocol': 'reflexion', 'n_rollouts': R,
            'eval_skill_temperature': args.eval_skill_temperature}
    for r, br in zip(eval_records, base_rolls):
        if br['correct']:
            recs.append({**head, 'data_id': r.get('data_id', ''), 'problem': r['problem'],
                         'reference_answer': r['reference_answer'], 'baseline_pass': 1.0,
                         'intervened': False, 'rubric_ok': None,
                         # 协议：不干预 -> 保持 baseline 结果
                         'withskill_acc_mean': 1.0, 'withskill_acc_strict_mean': 1.0,
                         'withskill_pass_any': 1.0, 'skill_parseable_mean': 1.0,
                         'withskill_terminated_mean': 1.0 if br['terminated'] else 0.0})
        else:
            hard.append((r, br))

    # ---- 2) hard 子集：rubric 诊断（纯 API，缓存命中后零成本） ----
    # entry 直接复用训练侧的 _rubric_entry：判据表与 fail_segment 的构成（code 任务下含单测
    # 真实报错）必须与训练逐字同源，否则 eval 的干预分布与训练分布不同。
    entries = [_rubric_entry(r, br) for r, br in hard]
    diags = _diagnose(rubric_cache, checker, entries, args.rubric_workers)
    todo = [(r, br, d) for (r, br), d in zip(hard, diags) if d]
    n_rubric_missing = len(hard) - len(todo)

    # ---- 3) rubric 条件化 skill-gen -> with-skill greedy 重跑 ----
    per_skills = _gen_skills(skill_sampler, [rubric_skillgen_prompt(r['problem'], d)
                                             for r, _br, d in todo], R, skill_dp, args) \
        if todo else []
    flat_prompts, flat_idx = [], []
    for pi, ((r, _br, _d), row) in enumerate(zip(todo, per_skills)):
        for j, (sk, sresp) in enumerate(row):
            flat_prompts.append(build_skill_solve_prompt(r['problem'], sk, sresp))
            flat_idx.append((pi, j))
    ws_out = _run_samples(base_sampler, flat_prompts, 1, args.max_tokens, base_dp,
                          temperature=0.0) if flat_prompts else []
    ws_rolls = v2._parse_many([(v2._first_seq(seqs), todo[pi][0]['reference_answer'])
                               for (pi, _j), seqs in zip(flat_idx, ws_out)])
    roll_by = {idx: roll for idx, roll in zip(flat_idx, ws_rolls)}

    hard_recs: List[Dict[str, Any]] = []
    for pi, ((r, br, d), row) in enumerate(zip(todo, per_skills)):
        rolls = [roll_by[(pi, j)] for j in range(len(row))]
        corr = [1.0 if x['correct'] else 0.0 for x in rolls]
        parses = [1.0 if sk else 0.0 for sk, _ in row]
        terms = [1.0 if x['terminated'] else 0.0 for x in rolls]
        rec = {**head, 'data_id': r.get('data_id', ''), 'problem': r['problem'],
               'reference_answer': r['reference_answer'], 'baseline_pass': 0.0,
               'intervened': True, 'rubric_ok': True, 'rubric': d,
               'base_stop_reason': br.get('stop_reason', 'none'),
               # code 任务：裸失败的种类（assertion / exception / no_code / timeout / ...）。
               # 这是本臂唯一能区分"方法错"与"格式崩塌"的字段，math 下恒为 None。
               'base_kind': br.get('kind'),
               'withskill_acc_mean': sum(corr) / len(corr) if corr else 0.0,
               # strict：unparseable 计 0（回退成 direct 时会被 baseline 掩护，此处 baseline=0
               # 所以两条曲线分叉纯粹反映格式崩塌）
               'withskill_acc_strict_mean': (sum(c * p for c, p in zip(corr, parses)) / len(corr)
                                             if corr else 0.0),
               'withskill_pass_any': 1.0 if any(corr) else 0.0,
               'skill_parseable_mean': sum(parses) / len(parses) if parses else 0.0,
               'withskill_terminated_mean': sum(terms) / len(terms) if terms else 0.0,
               'skill': row[0][0], 'skill_parseable': bool(row[0][0]), 'skill_chars': len(row[0][0]),
               'withskill_pred': rolls[0]['pred'], 'withskill_correct': rolls[0]['correct'],
               'withskill_terminated': rolls[0]['terminated'],
               'withskill_stop_reason': rolls[0]['stop_reason'], 'withskill_text': rolls[0]['text']}
        hard_recs.append(rec)
    # rubric 缺失的 hard 题：不进 rescue 分母（与训练侧"缺 rubric 一律丢弃"一致），但必须以
    # 显式零进 acc：它们确实没被干预、baseline 也确实错了。分母漂移靠 hard_rubric_missing
    # 可审计（rubric 成功后会永久进缓存，所以只会单调收敛到 0）。
    for (r, br), d in zip(hard, diags):
        if not d:
            recs.append({**head, 'data_id': r.get('data_id', ''), 'problem': r['problem'],
                         'reference_answer': r['reference_answer'], 'baseline_pass': 0.0,
                         'intervened': False, 'rubric_ok': False,
                         'base_stop_reason': br.get('stop_reason', 'none'),
                         'withskill_acc_mean': 0.0, 'withskill_acc_strict_mean': 0.0,
                         'withskill_pass_any': 0.0, 'skill_parseable_mean': 0.0,
                         'withskill_terminated_mean': 0.0})
    recs.extend(hard_recs)

    # ---- 4) 汇总 ----
    # acc 统一取"全部 eval 行的 withskill_acc_mean 均值"，与任何下游按行求均的脚本逐字一致。
    # 不能写成 base + (1-base)*rescue：那个式子隐含"缺 rubric 的题也按 rescue 率被救"，
    # 在缺失不为零时会系统高估 acc（缺失=0 时两者相等）。
    n_all = len(eval_records)
    n_hard = len(hard_recs)
    base = (sum(1.0 for br in base_rolls if br['correct']) / n_all) if n_all else 0.0
    acc = (sum(x['withskill_acc_mean'] for x in recs) / n_all) if n_all else 0.0
    acc_strict = (sum(x['withskill_acc_strict_mean'] for x in recs) / n_all) if n_all else 0.0
    rescue = (sum(x['withskill_acc_mean'] for x in hard_recs) / n_hard) if n_hard else 0.0
    rescue_strict = (sum(x['withskill_acc_strict_mean'] for x in hard_recs) / n_hard) if n_hard else 0.0
    rescue_any = (sum(x['withskill_pass_any'] for x in hard_recs) / n_hard) if n_hard else 0.0
    fmt = (sum(x['skill_parseable_mean'] for x in hard_recs) / n_hard) if n_hard else 0.0
    term = (sum(x['withskill_terminated_mean'] for x in hard_recs) / n_hard) if n_hard else 0.0
    # 错题结构：'length' = 没写完（E16 实测占裸失败的 97.6%），其余 = 写完但答错。
    # 本臂能不能测到"方法修正"完全取决于后者不为零，所以逐次 eval 都上报。
    trunc = (sum(1.0 for x in hard_recs if x.get('base_stop_reason') == 'length') / n_hard
             if n_hard else 0.0)
    # code 任务：裸失败的种类分布。math 上截断率就够（97.6% 是没写完），代码域必须分开看
    # —— 只有 assertion/exception 这类才是 rubric 有客观证据可诊断的失败，no_code/timeout
    # 是格式或环境问题。键名带 kind_ 前缀，随 summary 落盘（不进 swanlab 三条主指标）。
    kind_fracs = {}
    if v2._TASK == 'code' and n_hard:
        for kind in ('assertion', 'exception', 'import_or_syntax', 'no_code', 'no_entry',
                     'timeout'):
            kind_fracs[f'hard_base_kind_{kind}'] = (
                sum(1.0 for x in hard_recs if x.get('base_kind') == kind) / n_hard)

    summary = {'record_type': 'eval_summary', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
               'protocol': 'reflexion', 'n': n_all, 'n_rollouts': R,
               'eval_skill_temperature': args.eval_skill_temperature,
               'baseline_acc_mean1': base, 'acc_mean1': acc, 'lift_mean1': acc - base,
               'acc_strict_mean1': acc_strict, 'lift_strict_mean1': acc_strict - base,
               'format_mean1': fmt, 'term_mean1': term,
               # ★ 主指标
               'hard_n': n_hard, 'hard_rescue_rate': rescue,
               'hard_rescue_strict_rate': rescue_strict, 'hard_rescue_pass_any': rescue_any,
               'hard_rescued': sum(x['withskill_acc_mean'] for x in hard_recs),
               'hard_rubric_missing': n_rubric_missing,
               'hard_base_trunc_frac': trunc, **kind_fracs}
    # swanlab 只上报三条（2026-07-30 精简，命名不缩写）。口径：只用错题子集（baseline 做对
    # 的题不干预、不计入），所以 baseline_accuracy 恒为 0、with_skill_accuracy 就是救活率。
    # 其余读数（strict / pass_any / format / term / 混合 acc）仍全量在 summary 里落盘。
    # 不带 'eval/' 前缀：trainer.py 会统一加（f'eval/{k}'），写了会变成 eval/eval/xxx。
    metrics = {'baseline_accuracy': 0.0,
               'with_skill_accuracy': rescue,
               'lift': rescue}
    if n_hard and n_hard < 60:
        sys.stderr.write(f'[eval] WARNING: reflexion protocol has only {n_hard} hard problems; '
                         f'SE(rescue) ~ {(0.25 / n_hard) ** 0.5:.3f} — raise --eval-size.\n')
    return recs, summary, metrics
