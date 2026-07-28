#!/usr/bin/env python3
"""skill_config_probe.py — Qwen3-4B skill 模型 × executor 的纯推理配置探针。

目的：不训练，只推理。用 Qwen3-4B 当 skill 生成模型（T=0.5），产出 <skills> 喂给
executor（T=0，与 train_skill_v2.py 的 v2 eval 逐字同口径），比较不同配置的效果：
  - skill 模型 enable_thinking：on / off（由 --skill-thinking 指定，跑两次对比）
  - skill 模型 system prompt：7 种变体（含训练现版对照、六思路混合模板、toy 类比等）

题目：从 eval_records.jsonl(chunk=-1) 的 200 题按 baseline_pass 分层抽 50 题，
通过/失败比例与整集一致（难度配比同实际 eval），seed 固定保证跨配置可比。
baseline 直接复用缓存的 baseline_pass（同模型同 greedy 口径，无需重跑）。

用法：
  EXEC_GPUS=2 SKILL_GPUS=2 python3 skill_config_probe.py --skill-thinking off
  EXEC_GPUS=2 SKILL_GPUS=2 python3 skill_config_probe.py --skill-thinking on
  python3 skill_config_probe.py --skill-thinking off --prompts P2_combo,P3_toy   # 只跑子集
输出：skillcfg_{tag}.jsonl（逐 trial）+ stdout 汇总表。
"""
import argparse
import copy
import json
import os
import random
import re
import statistics as st
import sys
from typing import Dict, List, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.environ.get('GEN_MODEL_ID', 'Qwen/Qwen3-4B')
GPU_MEM = float(os.environ.get('GEN_GPU_MEM', 0.8))
EXEC_GPUS = int(os.environ.get('EXEC_GPUS', 2))
SKILL_GPUS = int(os.environ.get('SKILL_GPUS', 2))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 16384))
DEFAULT_EVAL_RECORDS = os.path.join(SCRIPT_DIR, '..', 'skill_v2', 'eval_records.jsonl')

# ===========================================================================
# executor 侧 —— 逐字复刻 train_skill_v2.py v2 分支（与 eval 口径一致）
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


def extract_boxed(text: str) -> Optional[str]:
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


def _clean_text(decoded: Optional[str]) -> str:
    return _SPECIAL_TOKEN_RE.sub('', decoded or '').rstrip()


_SEAM_TAG_RE = re.compile(r'<\s*answer\s*>(.*?)<\s*/\s*answer\s*>', re.I | re.S)
_SEAM_BOX_RE = re.compile(r'(?:\\{1,2}\(|)\\{1,2}boxed\s*\{\s*([^}]*)\s*}(?:\)|)', re.S)
_SEAM_INLINE_RE = re.compile(r'\$([^$]+)\$|\\\(([^)]+)\\\)', re.S)
_SEAM_FRAC_RE = re.compile(r'(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)')
_SEAM_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _seam_norm(num: str) -> str:
    try:
        f = float(num)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return num.strip()


def _seam_sanitize(txt: str) -> str:
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


def grade(seq, gold) -> Dict:
    text = _clean_text(getattr(seq, 'decoded', '') or '')
    raw = extract_boxed(text)
    pred = _seam_sanitize(raw) if raw else None
    correct = bool(pred) and (pred == _seam_sanitize(str(gold)))
    return {'pred': pred, 'correct': correct,
            'stop_reason': getattr(seq, 'stop_reason', None),
            'gen_tokens': len(getattr(seq, 'tokens', None) or []), 'text': text}


# ---- skill 抽取（v2 泛化版：砍 think 后取最后一个 <skills> 块）----
def extract_skill(text: str) -> Optional[str]:
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
    block = re.sub(r'</?(?:skills|skill|think)>', '', block, flags=re.IGNORECASE).strip()
    return block or None


def answer_leaked(skill: str, reference) -> bool:
    """诊断用：skill 文本中是否出现 gts 数值（digit-boundary，简版）。"""
    if not skill:
        return False
    g = _seam_sanitize(str(reference))
    if not g or not re.fullmatch(r'-?\d+(\.\d+)?', g):
        return False
    return bool(re.search(r'(?<![\d.])' + re.escape(g) + r'(?!\d)(?!\.\d)', skill))


# ===========================================================================
# skill 模型 system prompt 变体（英文 prompt + 中文注释；输出统一 <skills> 块便于解析）
# ===========================================================================
# P1：训练现版（方案1）SKILL_GEN_SYSTEM 逐字对照组。
P1_NARRATIVE = """\
You are a skill-generation model. Your <skills> block will be fed to a SEPARATE downstream executor model that must solve the problem on its own. The executor will NOT see your private reasoning — it only sees what is inside <skills>...</skills>.

First, think privately: actually work the problem out in your head to make sure you understand it, then step back and abstract WHAT MAKES THIS TYPE OF PROBLEM SOLVABLE into transferable methodology.

Then write the <skills> block following these rules:
- Give general, transferable solving techniques for this TYPE of problem: the key concepts/theorems it relies on, the recommended strategy and steps, and the common pitfalls to avoid — plus a brief reason for each piece of advice so the executor understands why.
- Write it as one coherent analysis narrative (not a bullet list): first name what the problem is essentially asking, then walk through how to approach it, blending concepts, steps, pitfalls and reasons into a single connected story.
- CRITICAL: Do NOT solve the problem for the executor. Do NOT reveal or compute the final answer, and do NOT substitute the problem's specific given numbers into the steps or state any intermediate numeric results. Leave ALL concrete numbers for the executor to compute on its own. If you catch yourself writing a specific number from the problem, replace it with a description of the quantity instead.
- Keep it concise: aim for roughly one focused paragraph.

Put ONLY the methodology inside <skills></skills>.
"""

# P2：六思路混合模板——主体二选一（toy 类比 / 路线卡片）+ 永远加纪律后缀；限长。
P2_COMBO = """\
You are a skill-generation model. Your <skills> block is the ONLY thing a separate executor model will see; it must help the executor solve the problem quickly within a tight token budget.

First think privately and solve the problem in your head. Then write a SHORT <skills> block (under 120 words) with exactly this structure:
1. MAIN PART - pick ONE of the two forms, whichever fits the problem better:
   (a) Toy example: invent a tiny problem of the SAME type but with DIFFERENT, smaller numbers, solve the toy completely in 2-4 lines showing the key trick, then add one sentence: "Your problem has the same shape - apply the same steps to its numbers."
   (b) Route card: name the problem type, then give the key formula / recurrence / lemma / reduction that cracks it (no derivation, no solving), and say what single quantity to compute.
2. LAST LINE - always end with exactly this discipline line: "Single pass: no re-deriving, no re-checking; once computed, box a bare number immediately."

Never use the original problem's own numbers in the main part; never state or imply the final answer.

Put everything inside <skills></skills>.
"""

# P3：纯 toy 类比——完整解一道异数字同型玩具题，靠示范迁移；天然 answer-free。
P3_TOY = """\
You are a skill-generation model. A separate executor model will solve the problem; it only sees your <skills> block.

First think privately and identify the core technique this problem needs. Then, inside <skills></skills>, do exactly one thing: invent a MINIATURE problem of the same type with DIFFERENT and much smaller numbers, and solve that miniature completely in at most 5 short lines, making the key trick explicit. Finish with one transfer sentence: "Your problem has the same shape - repeat these steps with its own numbers, then box a bare number."

Hard rules: never mention or use any number that appears in the original problem; never state the original problem's answer; keep the whole block under 100 words.
"""

# P4：路线卡片——极简结构化卡片（类型/公式/起点/目标），无叙述。
P4_CARD = """\
You are a skill-generation model. A separate executor model will solve the problem; it only sees your <skills> block.

First think privately and find the standard route for this problem. Then output, inside <skills></skills>, an ultra-compact ROUTE CARD with at most 4 lines:
TYPE: <the problem type in a few words>
KEY: <the one formula / recurrence / lemma / substitution that cracks it>
START: <what to set up first>
COMPUTE: <the single final quantity the executor must evaluate and box as a bare number>

No derivations, no explanations, no solving, never state the final answer, never copy the problem's numbers into KEY.
"""

# P5：预判纠错——无失败情报版“纠错针”：预判本题最可能的错误走向并拦截。
P5_PITFALL = """\
You are a skill-generation model. A separate executor model will solve the problem; it only sees your <skills> block.

First think privately: solve the problem in your head AND identify the single most likely way a solver goes wrong on this type (a tempting but wrong turn, an off-by-one, a wasteful brute-force, a wrong branch). Then, inside <skills></skills>, write under 90 words:
- WARNING: name that most likely mistake concretely and say why it is wrong.
- INSTEAD: one or two sentences pointing to the correct turn (technique name + where to apply it), without solving the problem or revealing any numeric result.
- End with: "Avoid re-checking loops; box a bare number as soon as it is computed."
"""

# P6：SEAM 经验风格（英文，输出改为 <skills> 统一解析）——概念/策略/易错三段式对照组。
P6_SEAM = """\
You are a problem-solving guidance model. Read the math problem and distill a concise, reusable piece of solving experience that will help a SEPARATE solver model reach the correct answer.
Rules:
- Do NOT solve the problem and do NOT reveal or compute the final answer.
- State the key concepts/theorems, the recommended strategy/steps, and the common pitfalls to avoid.
- Output ONLY the experience, wrapped EXACTLY as <skills> ... </skills>.
"""

# P7：一句话下界对照——只准一句话点出关键恒等式/技巧。
P7_MINIMAL = """\
You are a skill-generation model. A separate executor model will solve the problem; it only sees your <skills> block.
Inside <skills></skills>, write EXACTLY ONE sentence (max 30 words) naming the single key identity, theorem, or technique that cracks this problem. Nothing else. Never state the answer.
"""

PROMPTS = {
    'P1_narrative': P1_NARRATIVE,
    'P2_combo': P2_COMBO,
    'P3_toy': P3_TOY,
    'P4_card': P4_CARD,
    'P5_pitfall': P5_PITFALL,
    'P6_seam': P6_SEAM,
    'P7_minimal': P7_MINIMAL,
}


def skillgen_prompt(system: str, problem: str) -> Dict:
    # 与训练 _skillgen_prompt 同构：system + user('Problem:\n...')
    return {'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': f'Problem:\n{problem}'}]}


# ===========================================================================
# 题目分层抽样：与 200 题集 baseline 通过率同配比
# ===========================================================================
def load_problems(path, n, seed):
    probs = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('record_type') != 'eval_problem' or r.get('chunk') != -1:
            continue
        did = r.get('data_id')
        if did and did not in probs:
            probs[did] = {'data_id': did, 'problem': r['problem'],
                          'reference_answer': r['reference_answer'],
                          'baseline_pass': float(r.get('baseline_pass', 0))}
    items = sorted(probs.values(), key=lambda x: x['data_id'])
    passed = [x for x in items if x['baseline_pass'] > 0]
    failed = [x for x in items if x['baseline_pass'] == 0]
    ratio = len(passed) / max(1, len(items))
    n_pass = round(n * ratio)
    rng = random.Random(seed)
    sample = rng.sample(passed, min(n_pass, len(passed))) + \
        rng.sample(failed, min(n - n_pass, len(failed)))
    sample.sort(key=lambda x: x['data_id'])
    print(f'[抽样] 全集 {len(items)} 题 baseline率 {ratio:.3f} -> 抽 {len(sample)} 题 '
          f'(pass {sum(1 for x in sample if x["baseline_pass"] > 0)} / fail '
          f'{sum(1 for x in sample if x["baseline_pass"] == 0)})，seed={seed}')
    return sample


def run_batch(sampler, prompts, max_tokens, temperature, top_p, dp, num_samples=1):
    if not prompts:
        return []
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature,
                            top_p=top_p, num_samples=num_samples)
    padded = prompts
    if dp > 1 and 0 < len(prompts) < dp:
        padded = prompts + [copy.deepcopy(prompts[-1]) for _ in range(dp - len(prompts))]
    responses = sampler.sample(padded, params)[:len(prompts)]
    return [list(r.sequences) if (r and r.sequences) else [] for r in responses]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skill-thinking', choices=('on', 'off'), required=True)
    ap.add_argument('--prompts', default=','.join(PROMPTS.keys()))
    ap.add_argument('--n-problems', type=int, default=200)
    ap.add_argument('--n-rollouts', type=int, default=8, help='每题每思路的 skill 采样数（T>0）；executor 对每条 skill 各跑一次 greedy')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skill-temperature', type=float, default=0.5)
    ap.add_argument('--skill-top-p', type=float, default=0.95)
    ap.add_argument('--skill-max-tokens', type=int, default=8192)
    ap.add_argument('--max-tokens', type=int, default=8192, help='executor，对齐 eval')
    ap.add_argument('--eval-records', default=DEFAULT_EVAL_RECORDS)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    names = [x for x in args.prompts.split(',') if x]
    for x in names:
        if x not in PROMPTS:
            print(f'未知 prompt: {x}，可选: {list(PROMPTS)}')
            sys.exit(1)
    problems = load_problems(args.eval_records, args.n_problems, args.seed)
    think = args.skill_thinking == 'on'
    out_path = args.out or os.path.join(SCRIPT_DIR, f'skillcfg_{args.skill_thinking}.jsonl')

    # 两个采样器：skill(thinking 可切) + exec(enable_thinking=True, T=0, 对齐 v2 eval 的 base_sampler)
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

    # ---- 1) 所有配置的 skill-gen 一次性 batch ----
    sg_prompts, meta = [], []
    for name in names:
        for p in problems:
            sg_prompts.append(skillgen_prompt(PROMPTS[name], p['problem']))
            meta.append((name, p))
    print(f'[skill-gen] {len(sg_prompts)} 条 x {args.n_rollouts} rollouts (prompts={len(names)} x 题={len(problems)}), '
          f'thinking={args.skill_thinking}, T={args.skill_temperature}')
    sg_out = run_batch(skill_sampler, sg_prompts, args.skill_max_tokens,
                       args.skill_temperature, args.skill_top_p, SKILL_GPUS,
                       num_samples=args.n_rollouts)

    trials = []
    for (name, p), seqs in zip(meta, sg_out):
        seqs = list(seqs or [])
        for si in range(args.n_rollouts):
            seq = seqs[si] if si < len(seqs) else None
            full = _clean_text(getattr(seq, 'decoded', '') or '') if seq is not None else ''
            sk = extract_skill(full) or ''
            trials.append({'config': name, 'data_id': p['data_id'], 'sample_idx': si,
                           'problem': p['problem'],
                           'reference_answer': p['reference_answer'], 'baseline_pass': p['baseline_pass'],
                           'skill': sk, 'parseable': bool(sk), 'skill_chars': len(sk),
                           'leaked': answer_leaked(sk, p['reference_answer']),
                           'skillgen_full': full,
                           'skillgen_stop': getattr(seq, 'stop_reason', None) if seq is not None else 'empty',
                           'skillgen_tokens': len(getattr(seq, 'tokens', None) or []) if seq is not None else 0})

    # ---- 2) 所有配置的 executor 一次性 batch（空 skill 走 direct，等价 baseline 口径）----
    ex_prompts = [build_skill_solve_prompt(t['problem'], t['skill']) for t in trials]
    print(f'[executor] {len(ex_prompts)} 条, T=0, max_tokens={args.max_tokens}')
    ex_out = run_batch(exec_sampler, ex_prompts, args.max_tokens, 0.0, 1.0, EXEC_GPUS)

    with open(out_path, 'w') as f:
        for t, seqs in zip(trials, ex_out):
            roll = grade(seqs[0], t['reference_answer']) if seqs else {
                'pred': None, 'correct': False, 'stop_reason': 'empty', 'gen_tokens': 0, 'text': ''}
            t.update({'withskill_pred': roll['pred'], 'withskill_correct': roll['correct'],
                      'withskill_stop': roll['stop_reason'], 'withskill_tokens': roll['gen_tokens'],
                      'withskill_text': roll['text'],
                      'skill_thinking': args.skill_thinking})
            f.write(json.dumps(t, ensure_ascii=False) + '\n')

    # ---- 3) 汇总（含题级 pass@k）----
    print('\n' + '=' * 118)
    print('%-14s %-5s %-6s %-6s %-6s %-8s %-7s %-6s %-8s %-8s %-10s' % (
        'config', 'n', 'parse', 'leak', 'trunc', 'skill字符', 'mean@1', 'base', 'lift', 'pass@k', 'hard救活@k'))
    print('-' * 118)
    for name in names:
        sub = [t for t in trials if t['config'] == name]
        n = len(sub)
        parse = sum(t['parseable'] for t in sub) / n
        leak = sum(t['leaked'] for t in sub) / n
        trunc = sum(1 for t in sub if t['withskill_stop'] == 'length') / n
        chars = int(st.median([t['skill_chars'] for t in sub if t['parseable']] or [0]))
        acc = sum(t['withskill_correct'] for t in sub) / n
        base = sum(t['baseline_pass'] for t in sub) / n
        byq = {}
        for t in sub:
            byq.setdefault(t['data_id'], []).append(t)
        p_at_k = sum(1 for v in byq.values() if any(x['withskill_correct'] for x in v)) / len(byq)
        hardq = {d: v for d, v in byq.items() if v[0]['baseline_pass'] == 0}
        rescue_k = (sum(1 for v in hardq.values() if any(x['withskill_correct'] for x in v)) / len(hardq)) if hardq else 0.0
        print('%-14s %-5d %-6.2f %-6.2f %-6.2f %-8d %-7.3f %-6.3f %+-8.3f %-8.3f %-10.3f' % (
            name, n, parse, leak, trunc, chars, acc, base, acc - base, p_at_k, rescue_k))
    print('-' * 118)
    print(f'逐 trial 明细（含 skill-gen/executor 全文）已写入 {out_path}')


if __name__ == '__main__':
    main()
