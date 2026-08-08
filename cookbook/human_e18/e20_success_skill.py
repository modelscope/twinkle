# -*- coding: utf-8 -*-
"""E20：**成功** trajectory -> narrative skill（无 rubric），什么情况下该保留？

与 E18 的差别只有一处：题源从「裸解失败」换成「裸解**第一次就成功**」，
于是 skillmodel 拿到的是一条**成功的** trajectory，而且**没有 rubric**
（rubric 是失败诊断，成功的题没有可诊断的失败）。

⭐ 本实验的核心难点：这些题裸解已经通过，**pass_rate 没有提升空间**。
所以 E18 那套「+0.25 增益」门槛在这里恒不成立，直接套用会得出「一条都不该留」
的空洞结论。必须换保留判据。本实验同时量四条候选判据：

  J1 **不倒退 (do-no-harm)**：加了 skill 后 pass_rate 不下降。
      —— 这是**必要条件**，不是充分条件（什么都不说的废话 skill 也满足）。
  J2 **稳健性提升**：裸解 M 次里**并非全对**（0<base<1）的题，加 skill 后升到 1.0。
      —— 「第一次成功」不等于「稳定成功」，这类题才有真实增量。
  J3 **token 效率**：pass 不变但生成长度显著变短（少走弯路）。
  J4 **迁移性**：skill 不含本题专属标识（函数名/变量名），才可能对别的题有用。

判据的取舍理由写在 decide_keep() 里。

流程（8 卡：4 skill + 4 executor）：
  1. 裸解 BARE_ROLLOUTS 次，挑**第 1 次就通过**的题
  2. 把该次成功 trajectory（代码 + 通过信息）喂给 skillmodel，narrative、**无 rubric**
  3. 生成 N_SKILLS 个候选
  4. 每候选 executor 重解 EXEC_ROLLOUTS 次
  5. 按 J1-J4 分类，报「各判据下可保留的比例」
"""
import json
import os
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List

import torch
import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, '..', 'human'))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e18_kodcode import clean_text, judge_seqs, load_records  # noqa: E402
from e18_prompts import direct_prompt, skill_solve_prompt  # noqa: E402

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3-4B')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(_HERE, 'output.e20.success'))
SEED = int(os.environ.get('SEED', 42))
SKILL_GPUS = int(os.environ.get('SKILL_GPUS', 4))
EXEC_GPUS = int(os.environ.get('EXEC_GPUS', 4))
NUM_GPUS = SKILL_GPUS + EXEC_GPUS
GPU_MEM = float(os.environ.get('GPU_MEM', 0.85))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 24000))

N_TASKS = int(os.environ.get('N_TASKS', 64))          # 需要凑够的「首次成功」题数
POOL_MULT = int(os.environ.get('POOL_MULT', 4))       # 题池放大倍数（首次成功率约 40%）
N_SKILLS = int(os.environ.get('N_SKILLS', 4))
BARE_ROLLOUTS = int(os.environ.get('BARE_ROLLOUTS', 4))
EXEC_ROLLOUTS = int(os.environ.get('EXEC_ROLLOUTS', 8))
SKILL_MAX_TOKENS = int(os.environ.get('SKILL_MAX_TOKENS', 8192))
EXEC_MAX_TOKENS = int(os.environ.get('EXEC_MAX_TOKENS', 15000))
EXEC_TEMPERATURE = float(os.environ.get('EXEC_TEMPERATURE', 0.6))
EXEC_TOP_P = float(os.environ.get('EXEC_TOP_P', 0.95))
SKILL_TEMPERATURE = float(os.environ.get('SKILL_TEMPERATURE', 1.0))
SKILL_CHAR_LIMIT = int(os.environ.get('SKILL_CHAR_LIMIT', 1500))
RUN_ID = time.strftime('%m%d-%H%M%S')


# ===========================================================================
# prompt：成功 trajectory -> narrative skill（无 rubric）
# ===========================================================================
# ⭐ 与 E18 的 SKILLGEN_SYSTEM 保持同一 narrative 家族（散文体、第一人称、不提外部上下文、
# 不写代码块），只把「诊断失败」换成「复盘一次成功」。刻意**不引入** rubric 位 ——
# 本实验的自变量就是「没有 rubric」。
SKILLGEN_SYSTEM_SUCCESS = (
    'You are a Python programmer writing a note to your future self.\n\n'
    'You just solved a coding task on the first attempt. Write down the one '
    'insight that made it work, so that next time you meet a task of this shape '
    'you get it right immediately again.\n\n'
    'Requirements:\n'
    '- Write one flowing narrative in the first person, not bullet points or headings.\n'
    '- Name the concrete API, argument, keyword, data shape, or edge case that mattered.\n'
    '- Write it as guidance that transfers to other tasks of the same shape, not a '
    'description of this one task. Do not mention the specific function name you wrote.\n'
    '- Do NOT include any code block, and do NOT restate the solution.\n'
    '- If nothing non-obvious was involved, say so briefly instead of inventing a lesson.\n'
    '- Keep it under 150 words.\n'
    'Wrap the note in <skills> and </skills> tags.')


def skillgen_success_prompt(problem: str, code: str) -> Dict[str, Any]:
    """成功复盘 prompt。**无 rubric、无 GT** —— 只有题面和自己刚写对的代码。

    ⭐ 「If nothing non-obvious was involved, say so briefly」这一句是刻意加的逃生口：
    首次成功的题很多是**平凡题**，逼模型硬编一条"经验"只会得到套话。给它说"没什么特别"
    的许可，才能让 J1（不倒退）这个判据真正区分出「有料」和「没料」。
    """
    user = (f'The task:\n{problem}\n\n'
            f'The solution you wrote, which passed on the first attempt:\n'
            f'```python\n{code}\n```\n\n'
            'Write the note to your future self.')
    return {'messages': [{'role': 'system', 'content': SKILLGEN_SYSTEM_SUCCESS},
                         {'role': 'user', 'content': user}]}


# ===========================================================================
# 工具（与 e18_collect_kod 同源）
# ===========================================================================
def run_samples(sampler, prompts, num_samples, max_tokens, gen_dp,
                temperature=None, top_p=None):
    if not prompts:
        return []
    import copy
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.6 if temperature is None else temperature,
        top_p=0.95 if top_p is None else top_p,
        num_samples=num_samples)
    padded = prompts
    if gen_dp > 1 and 0 < len(prompts) < gen_dp:
        padded = prompts + [copy.deepcopy(prompts[-1]) for _ in range(gen_dp - len(prompts))]
    resp = sampler.sample(padded, params)[:len(prompts)]
    return [list(r.sequences) if (r and r.sequences) else [] for r in resp]


def seq_text(seq) -> str:
    return clean_text(getattr(seq, 'decoded', '') or '') if seq is not None else ''


def _mean(xs) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _pass_rate(rolls) -> float:
    return (sum(1.0 for x in rolls if x['correct']) / len(rolls)) if rolls else 0.0


def extract_skill(text: str) -> str:
    if '<skills>' not in text:
        return ''
    body = text.split('<skills>', 1)[1]
    return body.split('</skills>', 1)[0].strip() if '</skills>' in body else ''


# ===========================================================================
# 保留判据
# ===========================================================================
IDENT = re.compile(r'\bdef\s+(\w+)')
# 「无信息」自述：模型用了 prompt 给的逃生口，说明它自己认为这题没什么可学的
NOINFO = ('nothing non-obvious', 'nothing particularly', 'nothing special',
          'straightforward', 'no special', 'nothing unusual', 'not much to')


def decide_keep(base_rate: float, with_rate: float, skill: str,
                gt_code: str, base_tokens: float, with_tokens: float) -> Dict[str, Any]:
    """四条判据各自独立判定，不合成单一分数。

    ⭐ 为何不合成一个总分：这四条问的是**不同的问题**，权重取决于 skill 池的用途
    （做 SFT 目标 vs 做检索库），此处只如实报出各判据的通过情况，把权衡留给决策。

    J1 do-no-harm：with >= base。必要不充分 —— 一句废话也满足，所以**不能单独用它保留**。
    J2 稳健性：仅对 0<base<1（首次成功但不稳定）的题有意义，升到 1.0 才算。
       base 已经=1.0 的题在这条判据下恒为 False（没有可升空间），这是**设计如此**不是 bug。
    J3 token 效率：pass 不降且生成长度显著变短（>=10%），说明少走弯路。
       10% 门槛是任意的，但比"变短一点"要求高，避免采样噪声。
    J4 迁移性：skill 不含 GT 里的函数名 + 没使用「没什么特别」的自述。
       含函数名 -> 只对本题有效；自述无信息 -> 模型自己承认没料。
    """
    harmless = with_rate >= base_rate - 1e-9
    robust = (base_rate < 1.0) and (with_rate >= 1.0 - 1e-9)
    shorter = harmless and with_tokens > 0 and base_tokens > 0 and \
        (with_tokens <= 0.90 * base_tokens)
    names = set(IDENT.findall(gt_code or ''))
    low = (skill or '').lower()
    has_name = any(n and n.lower() in low for n in names)
    self_noinfo = any(p in low for p in NOINFO)
    transfer = (not has_name) and (not self_noinfo)
    return {'J1_harmless': harmless, 'J2_robust': robust, 'J3_shorter': shorter,
            'J4_transfer': transfer, 'has_own_fn_name': has_name,
            'self_says_noinfo': self_noinfo}


# ===========================================================================
# 主流程
# ===========================================================================
def build_runtime():
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, lazy_collect=False, groups=[
        DeviceGroup(name='skill', ranks=list(range(SKILL_GPUS)), device_type='GPU'),
        DeviceGroup(name='exec', ranks=list(range(SKILL_GPUS, NUM_GPUS)),
                    device_type='GPU')])

    def mk(group, world, thinking):
        s = vLLMSampler(model_id=MODEL_ID, remote_group=group,
                        device_mesh=DeviceMesh.from_sizes(world_size=world, dp_size=world),
                        engine_args={'gpu_memory_utilization': GPU_MEM,
                                     'max_model_len': MAX_MODEL_LEN,
                                     'tensor_parallel_size': 1})
        s.set_template(Template, model_id=MODEL_ID, enable_thinking=thinking,
                       max_length=MAX_MODEL_LEN)
        return s

    # skill 侧开 thinking（与 E18 采集口径一致，靠 thinking 拿多样性）；executor 关。
    return mk('skill', SKILL_GPUS, True), mk('exec', EXEC_GPUS, False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    ds, _ = load_records(SEED, 0, OUTPUT_DIR)
    pool = [r for i, r in enumerate(ds.dataset) if i < N_TASKS * POOL_MULT]
    logger.info(f'E20 start: 题池 {len(pool)}（目标首次成功 {N_TASKS} 题）'
                f' n_skills={N_SKILLS} bare={BARE_ROLLOUTS} exec={EXEC_ROLLOUTS}')
    skill_sampler, exec_sampler = build_runtime()

    # ---- 1. 裸解，挑「第一次就通过」的题 ----
    bare = run_samples(exec_sampler, [direct_prompt(r['problem']) for r in pool],
                       BARE_ROLLOUTS, EXEC_MAX_TOKENS, EXEC_GPUS,
                       temperature=EXEC_TEMPERATURE, top_p=EXEC_TOP_P)
    pairs, spans = [], []
    for r, seqs in zip(pool, bare):
        seqs = list(seqs or [])
        spans.append(len(seqs))
        pairs.extend((s, r['reference_answer']) for s in seqs)
    judged = judge_seqs(pairs) if pairs else []
    picked, i = [], 0
    for r, n in zip(pool, spans):
        rolls = judged[i:i + n]
        i += n
        if not rolls or not rolls[0]['correct']:
            continue          # ⭐ 只要**第 1 次**就通过的题（按用户要求）
        picked.append({'rec': r, 'base_rate': _pass_rate(rolls),
                       'code': rolls[0].get('code') or '',
                       'base_tokens': _mean([x.get('gen_tokens') for x in rolls])})
        if len(picked) >= N_TASKS:
            break
    logger.info(f'[bare] 首次成功 {len(picked)}/{len(pool)} 题'
                f'（其中 base<1 的不稳定题 {sum(1 for p in picked if p["base_rate"] < 1.0)}）')
    if not picked:
        raise RuntimeError('没有首次成功的题')

    # ---- 2. 成功 traj -> narrative skill（无 rubric）----
    sg = run_samples(skill_sampler,
                     [skillgen_success_prompt(p['rec']['problem'], p['code'])
                      for p in picked],
                     N_SKILLS, SKILL_MAX_TOKENS, SKILL_GPUS,
                     temperature=SKILL_TEMPERATURE)
    flat = []
    for p, seqs in zip(picked, sg):
        for ci in range(N_SKILLS):
            seq = seqs[ci] if seqs and ci < len(seqs) else None
            sk = extract_skill(seq_text(seq))
            flat.append({'p': p, 'cand_idx': ci, 'skill': sk,
                         'stop': getattr(seq, 'stop_reason', None) if seq else None})
    n_ok = sum(1 for f in flat if f['skill'])
    logger.info(f'[skillgen] {len(flat)} 候选，可解析 {n_ok} ({100*n_ok/len(flat):.0f}%)')

    # ---- 3. 带 skill 重解 ----
    todo = [f for f in flat if f['skill']]
    ws = run_samples(exec_sampler,
                     [skill_solve_prompt(f['p']['rec']['problem'], f['skill']) for f in todo],
                     EXEC_ROLLOUTS, EXEC_MAX_TOKENS, EXEC_GPUS,
                     temperature=EXEC_TEMPERATURE, top_p=EXEC_TOP_P)
    pairs, spans = [], []
    for f, seqs in zip(todo, ws):
        seqs = list(seqs or [])
        spans.append(len(seqs))
        pairs.extend((s, f['p']['rec']['reference_answer']) for s in seqs)
    judged = judge_seqs(pairs) if pairs else []
    i = 0
    for f, n in zip(todo, spans):
        rolls = judged[i:i + n]
        i += n
        f['with_rate'] = _pass_rate(rolls)
        f['with_tokens'] = _mean([x.get('gen_tokens') for x in rolls])

    # ---- 4. 判据 ----
    out = os.path.join(OUTPUT_DIR, 'e20_candidates.jsonl')
    with open(out, 'w', encoding='utf-8') as fh:
        for f in flat:
            p = f['p']
            if not f['skill']:
                row = {'data_id': p['rec']['data_id'], 'cand_idx': f['cand_idx'],
                       'parseable': False, 'base_rate': p['base_rate'],
                       'with_rate': None, 'skill': '', 'skill_chars': 0}
            else:
                d = decide_keep(p['base_rate'], f['with_rate'], f['skill'],
                                p['rec']['reference_answer'].get('canonical_solution', ''),
                                p['base_tokens'], f['with_tokens'])
                row = {'data_id': p['rec']['data_id'], 'cand_idx': f['cand_idx'],
                       'parseable': True, 'base_rate': p['base_rate'],
                       'with_rate': f['with_rate'],
                       'delta': round(f['with_rate'] - p['base_rate'], 6),
                       'base_tokens': round(p['base_tokens'], 1),
                       'with_tokens': round(f['with_tokens'], 1),
                       'skill_chars': len(f['skill']), 'stop': f['stop'],
                       'skill': f['skill'], **d}
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    logger.info(f'[done] 落盘 {out}，用时 {(time.time()-t0)/60:.1f} 分钟')


if __name__ == '__main__':
    main()
