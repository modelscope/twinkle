#!/usr/bin/env python3
"""reflexion_probe.py — reflexion 链路探针（纯推理，不训练）。

链路：executor 失败轨迹（复用 v2 eval 的 baseline 缓存原文）
  → LLM rubric 诊断（qwen-plus，7 条 _MATH_RUBRIC，[PASS]/[FAIL]+fix 文本，防泄漏硬规则）
  → skillmodel(Qwen3-4B, T=0.5) 条件于 (题目 + 失败轨迹 + 诊断) 生成 <skills>
  → executor(Qwen3-4B, T=0, v2 eval 逐字口径) 带 skill 重试
目的：在"有一次真实失败 + rubric 诊断"的条件下，比较 skillmodel 的 prompt 写法 ×
thinking on/off 哪种救活率最高，为 buffer B 如何用 rubric 经验提供依据。

题目：eval 200 题中 baseline=0 的失败题抽 N 道（seed 固定），全部是"裸解必错"题，
因此 executor 重试的 acc 即救活率。诊断按 data_id 缓存（rubric_diag_cache.jsonl），
on/off 两次运行复用，不重复调 API。

prompt 变体（信息量递增，用于分离各级情报的增量价值）：
  R4_blind      题目（无失败、无诊断；= 上轮 P5_pitfall 原文，跨轮对照锚点）
  R0_trace_only 题目 + 失败轨迹尾部（无诊断；消融 rubric 的增量）
  R1_needle     题目 + 失败 + 诊断 → 纠错针（WARNING/INSTEAD + 纪律后缀）
  R2_narrative  题目 + 失败 + 诊断 → 训练现版叙述式（对照 v2 regen 路径）
  R3_toy_fix    题目 + 失败 + 诊断 → 针对诊断错误点的玩具题示范

用法（~/.env 需含 LLM_BACKUP_BASE_URL / LLM_BACKUP_API_KEY，脚本自动加载）：
  EXEC_GPUS=2 SKILL_GPUS=2 python3 reflexion_probe.py --skill-thinking off
  EXEC_GPUS=2 SKILL_GPUS=2 python3 reflexion_probe.py --skill-thinking on
输出：reflexion_{tag}.jsonl（含 skill-gen 全文与 executor 全文）+ stdout 汇总。
"""
import argparse
import copy
import hashlib
import json
import os
import random
import re
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_home_env():
    """加载 ~/.env（KEY=VALUE 简单格式；不覆盖已存在的环境变量）。"""
    p = os.path.expanduser('~/.env')
    if not os.path.exists(p):
        return
    for line in open(p):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_home_env()

import twinkle  # noqa: E402
from twinkle import DeviceGroup, DeviceMesh  # noqa: E402
from twinkle.data_format import SamplingParams  # noqa: E402
from twinkle.sampler import vLLMSampler  # noqa: E402
from twinkle.template import Template  # noqa: E402

MODEL_ID = os.environ.get('GEN_MODEL_ID', 'Qwen/Qwen3-4B')
GPU_MEM = float(os.environ.get('GEN_GPU_MEM', 0.8))
EXEC_GPUS = int(os.environ.get('EXEC_GPUS', 2))
SKILL_GPUS = int(os.environ.get('SKILL_GPUS', 2))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 16384))
DIAG_MODEL = os.environ.get('LLM_BACKUP_MODEL', 'qwen-plus')
EVAL_RECORDS = os.path.join(SCRIPT_DIR, '..', 'skill_v2', 'eval_records.jsonl')
BASE_CACHE = os.path.join(SCRIPT_DIR, '..', 'skill_v2', 'cache', 'eval_baseline.jsonl')
DIAG_CACHE = os.path.join(SCRIPT_DIR, 'rubric_diag_cache.jsonl')

# ===========================================================================
# executor 侧（逐字复刻 v2 eval，与 skill_config_probe.py 相同）
# ===========================================================================
_ANSWER_FORMAT_V2 = ('Present your reasoning, then put ONLY the final numeric result inside '
                     '\\boxed{}. For example: \\boxed{42}.')


def build_direct_prompt(problem):
    content = f'The problem you need to solve:\n{problem}\n\n' + _ANSWER_FORMAT_V2
    return {'messages': [{'role': 'user', 'content': content}]}


def build_skill_solve_prompt(problem, skill):
    skill = (skill or '').strip()
    if not skill:
        return build_direct_prompt(problem)
    content = (f'The problem you need to solve:\n{problem}\n\n'
               'Skill hint:\nFor this problem, a skill-generation model has analyzed it and '
               'provided some advisory skills:\n'
               f'{skill}\n'
               'Prefer using its techniques when they fit, but if you have a more efficient or '
               'clearer correct method, you may use it. If you diverge from this advice, briefly '
               'explain why. Be concise and accurate.\n'
               + _ANSWER_FORMAT_V2)
    return {'messages': [{'role': 'user', 'content': content}]}


_BOXED_RE = re.compile(r'\\boxed\s*\{')


def extract_boxed(text):
    if not text:
        return None
    last = None
    for m in _BOXED_RE.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            depth += (text[i] == '{') - (text[i] == '}')
            i += 1
        if depth == 0:
            last = text[m.end():i - 1].strip()
    return last


_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>')


def _clean_text(d):
    return _SPECIAL_TOKEN_RE.sub('', d or '').rstrip()


_SEAM_TAG_RE = re.compile(r'<\s*answer\s*>(.*?)<\s*/\s*answer\s*>', re.I | re.S)
_SEAM_BOX_RE = re.compile(r'(?:\\{1,2}\(|)\\{1,2}boxed\s*\{\s*([^}]*)\s*}(?:\)|)', re.S)
_SEAM_INLINE_RE = re.compile(r'\$([^$]+)\$|\\\(([^)]+)\\\)', re.S)
_SEAM_FRAC_RE = re.compile(r'(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)')
_SEAM_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _seam_norm(num):
    try:
        f = float(num)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return num.strip()


def _seam_sanitize(txt):
    txt = (txt or '').strip()
    if (m := _SEAM_TAG_RE.search(txt)):
        txt = m.group(1).strip()
    elif (m := _SEAM_BOX_RE.search(txt)):
        txt = m.group(1).strip()
    elif (m := _SEAM_INLINE_RE.search(txt)):
        txt = (m.group(1) or m.group(2)).strip()
    txt = re.sub(r'\\frac\s*\{\s*([^}]+?)\s*}\s*\{\s*([^}]+?)\s*}', r'\1/\2', txt)
    if (m := _SEAM_FRAC_RE.search(txt)):
        p, q = map(float, m.groups())
        if q:
            return _seam_norm(str(p / q))
    if (m := _SEAM_NUM_RE.search(txt)):
        return _seam_norm(m.group())
    return txt


def grade(seq, gold):
    text = _clean_text(getattr(seq, 'decoded', '') or '')
    raw = extract_boxed(text)
    pred = _seam_sanitize(raw) if raw else None
    return {'pred': pred, 'correct': bool(pred) and (pred == _seam_sanitize(str(gold))),
            'stop_reason': getattr(seq, 'stop_reason', None),
            'gen_tokens': len(getattr(seq, 'tokens', None) or []), 'text': text}


def extract_skill(text):
    low = text.lower()
    end_think = low.rfind('</think>')
    answer = text[end_think + len('</think>'):] if end_think >= 0 else text
    s = answer.lower().rfind('<skills>')
    if s < 0:
        return None
    inner = s + len('<skills>')
    e = answer.lower().find('</skills>', inner)
    if e < 0:
        return None
    block = answer[inner:e].strip()
    return re.sub(r'</?(?:skills|skill|think)>', '', block, flags=re.I).strip() or None


def answer_leaked(text, reference):
    """双口径泄漏检测：A=裸数字任意位置；B 口径由分析端按 |gts|>=10 复算。"""
    if not text:
        return False
    g = _seam_sanitize(str(reference))
    if not g or not re.fullmatch(r'-?\d+(\.\d+)?', g):
        return False
    return bool(re.search(r'(?<![\d.])' + re.escape(g) + r'(?!\d)(?!\.\d)', text))


# ===========================================================================
# rubric 诊断（对齐 v2 的 _MATH_RUBRIC；qwen-plus；防泄漏硬规则；文本格式 [PASS]/[FAIL]+fix）
# ===========================================================================
MATH_RUBRIC = [
    'The attempt chooses a method suitable for the problem structure',
    'The attempt identifies the key constraint, invariant, or quantity before computing',
    'Algebraic and logical transformations preserve validity at each step',
    'The attempt checks required constraints, domains, boundary cases, or validity conditions',
    'The attempt avoids redundant casework, looping, or re-deriving known facts',
    'The attempt reaches a final answer within the length budget',
    'The approach stays focused on the actual question asked',
]

# 中文注释：诊断 system prompt——硬规则禁止给出最终答案/最终数值，输出 [PASS]/[FAIL]+reason+fix
# 的紧凑文本（与 v2 _format_diagnosis 的落盘形态一致），供 skillmodel 直接消费。
DIAG_SYSTEM = """\
You are a rigorous math-competition grader. You will see a problem and a FAILED solution attempt (possibly truncated). Evaluate the attempt against each rubric criterion.

Output format: one line per criterion, exactly:
- [PASS] <criterion text>
or
- [FAIL] <criterion text>: <one-sentence reason> (fix: <one-sentence concrete correction direction>)
Then a final line: Summary: <2-3 sentences naming the single most damaging error and the correct turn to take>.

HARD RULES: never state, compute, or hint at the problem's final numeric answer or any final-stage numeric result; describe errors and directions only. Keep the whole output under 250 words."""

DIAG_USER = """## Problem
{problem}

## Rubric
{rubric}

## Failed attempt (may be truncated)
{segment}

Now output the diagnostic lines."""


def diagnose(client, problem, fail_text, gold):
    rubric = '\n'.join(f'{i+1}. {t}' for i, t in enumerate(MATH_RUBRIC))
    seg = fail_text[-4000:]
    msg = [{'role': 'system', 'content': DIAG_SYSTEM},
           {'role': 'user', 'content': DIAG_USER.format(problem=problem, rubric=rubric, segment=seg)}]
    r = client.chat.completions.create(model=DIAG_MODEL, messages=msg, max_tokens=600,
                                       temperature=0.2, timeout=120)
    text = (r.choices[0].message.content or '').strip()
    # 防泄漏兜底：诊断若带出 gts 数值，重试一次更严的指令；仍泄漏则截去含数值的行
    if answer_leaked(text, gold):
        msg.append({'role': 'assistant', 'content': text})
        msg.append({'role': 'user', 'content': 'Your output contained a forbidden final numeric value. '
                    'Rewrite the SAME diagnosis with every final-stage number removed.'})
        r = client.chat.completions.create(model=DIAG_MODEL, messages=msg, max_tokens=600,
                                           temperature=0.2, timeout=120)
        text = (r.choices[0].message.content or '').strip()
        if answer_leaked(text, gold):
            g = _seam_sanitize(str(gold))
            text = '\n'.join(l for l in text.splitlines()
                             if not re.search(r'(?<![\d.])' + re.escape(g) + r'(?!\d)(?!\.\d)', l))
    return text


# ===========================================================================
# skillmodel prompt 变体（英文 prompt + 中文注释；统一 <skills> 输出）
# ===========================================================================
# R4：无情报锚点 = 上轮 P5_pitfall 原文（跨实验可比）。
R4_BLIND_SYS = """\
You are a skill-generation model. A separate executor model will solve the problem; it only sees your <skills> block.

First think privately: solve the problem in your head AND identify the single most likely way a solver goes wrong on this type (a tempting but wrong turn, an off-by-one, a wasteful brute-force, a wrong branch). Then, inside <skills></skills>, write under 90 words:
- WARNING: name that most likely mistake concretely and say why it is wrong.
- INSTEAD: one or two sentences pointing to the correct turn (technique name + where to apply it), without solving the problem or revealing any numeric result.
- End with: "Avoid re-checking loops; box a bare number as soon as it is computed."
"""

# R0：只有失败轨迹（无诊断）——消融 rubric 的增量价值。
R0_TRACE_SYS = """\
You are a skill-generation model. A separate executor model previously FAILED this problem; you will see the tail of its failed attempt. The executor will retry seeing ONLY your <skills> block.

First think privately: read the failed attempt, find where it went wrong, and decide the correct turn. Then, inside <skills></skills>, write under 90 words:
- WARNING: the concrete mistake the previous attempt made (quote its wrong move briefly).
- INSTEAD: one or two sentences pointing to the correct turn, without solving the problem or revealing any numeric result.
- End with: "Avoid re-checking loops; box a bare number as soon as it is computed."
"""

# R1：query + 诊断（无轨迹）→ 纠错针（把 rubric 的 FAIL/fix 转译成对 executor 的直接行为指令）。
R1_NEEDLE_SYS = """\
You are a skill-generation model. A separate executor model previously FAILED this problem. You will see an expert rubric diagnosis of that failure (you will NOT see the failed attempt itself). The executor will retry seeing ONLY your <skills> block.

First think privately: from the diagnosis, pinpoint the decisive error. Then, inside <skills></skills>, write under 90 words:
- WARNING: the decisive mistake (grounded in the diagnosis, stated concretely for THIS problem).
- INSTEAD: the corrective route distilled from the diagnosis's fix directions - technique name + where to apply it. Do not solve the problem; never state any numeric result.
- End with: "Avoid re-checking loops; box a bare number as soon as it is computed."
"""

# R2：query + 诊断（无轨迹）→ 训练现版叙述式（对照 v2 _regen_prompt 的文体路径）。
R2_NARR_SYS = """\
You are a skill-generation model. A separate executor model previously FAILED this problem. You will see an expert rubric diagnosis of that failure (you will NOT see the failed attempt itself). The executor will retry seeing ONLY your <skills> block.

First, think privately: work the problem out and understand why the attempt failed. Then write the <skills> block as ONE coherent analysis narrative (not a bullet list): name what the problem is essentially asking, walk through the recommended approach, and weave in - informed by the diagnosis - the specific pitfall that sank the previous attempt and how to avoid it. Do NOT solve the problem, do NOT reveal or compute the final answer, and do NOT substitute the problem's specific numbers into the steps. Keep it to roughly one focused paragraph.

Put ONLY the methodology inside <skills></skills>.
"""

# R3：query + 诊断（无轨迹）→ 玩具题示范（针对诊断指出的错误技巧点造 toy，异数字防泄漏）。
R3_TOYFIX_SYS = """\
You are a skill-generation model. A separate executor model previously FAILED this problem. You will see an expert rubric diagnosis of that failure (you will NOT see the failed attempt itself). The executor will retry seeing ONLY your <skills> block.

First think privately: from the diagnosis, identify the ONE technique the executor got wrong. Then, inside <skills></skills>, do exactly this (under 110 words):
1. Invent a MINIATURE problem exercising that same technique with DIFFERENT, much smaller numbers, and solve the miniature completely in at most 5 short lines, making the correct move (the one the failed attempt missed) explicit.
2. One transfer sentence: "Your problem needs the same move where the previous attempt went wrong - apply it, then box a bare number."
Hard rules: never use any number from the original problem; never state its answer.
"""

PROMPTS = {
    'R4_blind': ('none', R4_BLIND_SYS),
    'D1_needle': ('diag', R1_NEEDLE_SYS),
    'D2_narr': ('diag', R2_NARR_SYS),
    'D3_toyfix': ('diag', R3_TOYFIX_SYS),
}


def skillgen_prompt(name, problem, fail_tail, diag):
    mode, sys_p = PROMPTS[name]
    user = f'Problem:\n{problem}'
    if mode in ('trace', 'trace+diag'):
        user += f'\n\nFailed attempt (tail):\n{fail_tail}'
    if mode in ('diag', 'trace+diag'):
        user += f'\n\nExpert rubric diagnosis of the failure:\n{diag}'
    return {'messages': [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': user}]}


# ===========================================================================
# 数据与主流程
# ===========================================================================
def md5_key(problem):
    return hashlib.md5('\x1f'.join([problem]).encode('utf-8')).hexdigest()


def load_fail_problems(n, seed):
    probs = {}
    for line in open(EVAL_RECORDS):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('record_type') != 'eval_problem' or r.get('chunk') != -1:
            continue
        if r['data_id'] not in probs and float(r.get('baseline_pass', 1)) == 0:
            probs[r['data_id']] = {'data_id': r['data_id'], 'problem': r['problem'],
                                   'reference_answer': r['reference_answer']}
    cache = {}
    for line in open(BASE_CACHE):
        try:
            c = json.loads(line)
            cache[c['key']] = c['value']
        except Exception:
            continue
    items = []
    for p in sorted(probs.values(), key=lambda x: x['data_id']):
        v = cache.get(md5_key(p['problem']))
        if v and not v.get('correct') and (v.get('text') or '').strip():
            p['fail_text'] = v['text']
            items.append(p)
    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))
    sample.sort(key=lambda x: x['data_id'])
    print(f'[抽样] baseline 失败且有轨迹全文的题 {len(items)} -> 抽 {len(sample)}（seed={seed}）')
    return sample


def run_batch(sampler, prompts, max_tokens, temperature, top_p, dp, num_samples=1):
    if not prompts:
        return []
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, num_samples=num_samples)
    padded = prompts
    if dp > 1 and 0 < len(prompts) < dp:
        padded = prompts + [copy.deepcopy(prompts[-1]) for _ in range(dp - len(prompts))]
    responses = sampler.sample(padded, params)[:len(prompts)]
    return [list(r.sequences) if (r and r.sequences) else [] for r in responses]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skill-thinking', choices=('on', 'off'), required=True)
    ap.add_argument('--prompts', default=','.join(PROMPTS.keys()))
    ap.add_argument('--n-problems', type=int, default=130)
    ap.add_argument('--n-rollouts', type=int, default=8, help='每题每思路的 skill 采样数（T>0）；executor 对每条 skill 各跑一次 greedy')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skill-temperature', type=float, default=0.5)
    ap.add_argument('--skill-max-tokens', type=int, default=8192)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--fail-tail-chars', type=int, default=1800)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    names = [x for x in args.prompts.split(',') if x]
    problems = load_fail_problems(args.n_problems, args.seed)
    out_path = args.out or os.path.join(SCRIPT_DIR, f'reflexion_{args.skill_thinking}.jsonl')

    # ---- 阶段1：rubric 诊断（带磁盘缓存，8 线程并发）----
    diag_cache = {}
    if os.path.exists(DIAG_CACHE):
        for line in open(DIAG_CACHE):
            try:
                c = json.loads(line)
                diag_cache[c['data_id']] = c['diag']
            except Exception:
                continue
    todo = [p for p in problems if p['data_id'] not in diag_cache]
    if todo:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ['LLM_BACKUP_API_KEY'],
                        base_url=os.environ['LLM_BACKUP_BASE_URL'])
        print(f'[诊断] 需调 API {len(todo)} 题（model={DIAG_MODEL}），其余 {len(problems)-len(todo)} 题走缓存')

        def _one(p):
            try:
                return p['data_id'], diagnose(client, p['problem'], p['fail_text'], p['reference_answer'])
            except Exception as e:
                return p['data_id'], f'[DIAG_ERROR] {e}'
        with ThreadPoolExecutor(max_workers=8) as ex:
            with open(DIAG_CACHE, 'a') as f:
                for did, diag in ex.map(_one, todo):
                    diag_cache[did] = diag
                    f.write(json.dumps({'data_id': did, 'diag': diag}, ensure_ascii=False) + '\n')
    n_err = sum(1 for p in problems if str(diag_cache.get(p['data_id'], '')).startswith('[DIAG_ERROR]'))
    print(f'[诊断] 完成，失败 {n_err} 题')

    # ---- 阶段2：skill-gen + executor ----
    think = args.skill_thinking == 'on'
    twinkle.initialize(mode='ray', nproc_per_node=SKILL_GPUS + EXEC_GPUS, lazy_collect=False, groups=[
        DeviceGroup(name='skill', ranks=list(range(SKILL_GPUS)), device_type='GPU'),
        DeviceGroup(name='exec', ranks=list(range(SKILL_GPUS, SKILL_GPUS + EXEC_GPUS)), device_type='GPU')])

    def make_sampler(group, world, enable_thinking):
        s = vLLMSampler(model_id=MODEL_ID,
                        engine_args={'gpu_memory_utilization': GPU_MEM,
                                     'max_model_len': MAX_MODEL_LEN, 'tensor_parallel_size': 1},
                        device_mesh=DeviceMesh.from_sizes(world_size=world, dp_size=world),
                        remote_group=group)
        s.set_template(Template, model_id=MODEL_ID, enable_thinking=enable_thinking, max_length=MAX_MODEL_LEN)
        return s

    skill_sampler = make_sampler('skill', SKILL_GPUS, enable_thinking=think)
    exec_sampler = make_sampler('exec', EXEC_GPUS, enable_thinking=True)

    sg_prompts, meta = [], []
    for name in names:
        for p in problems:
            tail = p['fail_text'][-args.fail_tail_chars:]
            sg_prompts.append(skillgen_prompt(name, p['problem'], tail, diag_cache.get(p['data_id'], '')))
            meta.append((name, p))
    print(f'[skill-gen] {len(sg_prompts)} 条 x {args.n_rollouts} rollouts, thinking={args.skill_thinking}, T={args.skill_temperature}')
    sg_out = run_batch(skill_sampler, sg_prompts, args.skill_max_tokens,
                       args.skill_temperature, 0.95, SKILL_GPUS, num_samples=args.n_rollouts)

    trials = []
    for (name, p), seqs in zip(meta, sg_out):
        seqs = list(seqs or [])
        for si in range(args.n_rollouts):
            seq = seqs[si] if si < len(seqs) else None
            full = _clean_text(getattr(seq, 'decoded', '') or '') if seq is not None else ''
            sk = extract_skill(full) or ''
            trials.append({'config': name, 'data_id': p['data_id'], 'sample_idx': si,
                           'problem': p['problem'],
                           'reference_answer': p['reference_answer'],
                           'diag': diag_cache.get(p['data_id'], ''),
                           'diag_leaked': answer_leaked(diag_cache.get(p['data_id'], ''), p['reference_answer']),
                           'skill': sk, 'parseable': bool(sk), 'skill_chars': len(sk),
                           'leaked': answer_leaked(sk, p['reference_answer']),
                           'skillgen_full': full,
                           'skillgen_stop': getattr(seq, 'stop_reason', None) if seq is not None else 'empty',
                           'skillgen_tokens': len(getattr(seq, 'tokens', None) or []) if seq is not None else 0})

    ex_prompts = [build_skill_solve_prompt(t['problem'], t['skill']) for t in trials]
    print(f'[executor] {len(ex_prompts)} 条, T=0')
    ex_out = run_batch(exec_sampler, ex_prompts, args.max_tokens, 0.0, 1.0, EXEC_GPUS)

    with open(out_path, 'w') as f:
        for t, seqs in zip(trials, ex_out):
            roll = grade(seqs[0], t['reference_answer']) if seqs else {
                'pred': None, 'correct': False, 'stop_reason': 'empty', 'gen_tokens': 0, 'text': ''}
            t.update({'withskill_pred': roll['pred'], 'withskill_correct': roll['correct'],
                      'withskill_stop': roll['stop_reason'], 'withskill_tokens': roll['gen_tokens'],
                      'withskill_text': roll['text'], 'skill_thinking': args.skill_thinking})
            f.write(json.dumps(t, ensure_ascii=False) + '\n')

    # ---- 汇总（全是 baseline=0 的题，acc 即救活率；另报题级 pass@k）----
    print('\n' + '=' * 112)
    print('%-11s %-6s %-6s %-6s %-6s %-8s %-8s %-8s %-8s' % (
        'config', 'n', 'parse', 'leakA', 'trunc', 'skill字符', '救活@1', '救活@k', '救活@1(gts>=10无泄漏)'))
    print('-' * 112)
    for name in names:
        sub = [t for t in trials if t['config'] == name]
        n = len(sub)
        parse = sum(t['parseable'] for t in sub) / n
        leak = sum(t['leaked'] for t in sub) / n
        trunc = sum(1 for t in sub if t['withskill_stop'] == 'length') / n
        chars = int(st.median([t['skill_chars'] for t in sub if t['parseable']] or [0]))
        acc = sum(t['withskill_correct'] for t in sub) / n
        byq = {}
        for t in sub:
            byq.setdefault(t['data_id'], []).append(t)
        p_at_k = sum(1 for v in byq.values() if any(x['withskill_correct'] for x in v)) / len(byq)
        clean = [t for t in sub if not t['leaked']]
        cacc = sum(t['withskill_correct'] for t in clean) / max(1, len(clean))
        print('%-11s %-6d %-6.2f %-6.2f %-6.2f %-8d %-8.3f %-8.3f %-.3f(n=%d)' % (
            name, n, parse, leak, trunc, chars, acc, p_at_k, cacc, len(clean)))
    print('-' * 112)
    print(f'明细（含 skill-gen/executor 全文）已写入 {out_path}')


if __name__ == '__main__':
    main()
