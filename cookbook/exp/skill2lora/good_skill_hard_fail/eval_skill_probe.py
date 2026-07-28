#!/usr/bin/env python3
"""eval_skill_probe.py — 自包含的"单次 eval"探针，用于手工迭代 skill。

目的：固定一批难题（executor 持续解不出的），保持 executor 侧 prompt/采样/判分与
train_skill_v2.py 的 eval 完全一致（v2 模式：单 user turn + \\boxed{} 答案格式，
executor 用 base_sampler、enable_thinking=True、greedy 温度 0），但允许自由替换每题
的 skill(experience) 内容。反复替换 skill 重跑，一旦某题解对，就把该 skill 落盘到
winning_skills.jsonl，从而观察"能让 executor 生效的 skill 到底长什么样"。

不 import train_skill_v2.py（自包含）；executor 侧逻辑逐段复刻自该文件（2026-07）。

用法：
  # 1) 生成 trials 模板（默认 10 道难题，skill 待填）：
  python3 eval_skill_probe.py --init
  # 2) 编辑 trials.jsonl，给每题填不同的 skill，然后跑：
  python3 eval_skill_probe.py
  # 只测某题 / 调大 max_tokens（验证"截断"型失败是否靠加长度能救）：
  python3 eval_skill_probe.py --only seam:val:128 --max-tokens 12000

trials.jsonl 每行一个 JSON（# 开头的行会被忽略，可当注释）：
  {"data_id": "seam:val:128", "skill": "..."}            # 用该 skill 解题
  {"data_id": "seam:val:128", "skill": ""}               # 空 skill = baseline
  {"data_id": "seam:val:128", "tag": "v3", "skill":"..."}# tag 便于区分同题多次试验
problem / reference_answer 默认按 data_id 从 eval_records.jsonl 解析；也可在行内直接
提供 "problem" / "reference_answer" 覆盖（用于测 eval 集之外的题）。

GPU：仅起一个 executor sampler。默认 EXEC_GPUS=2；若训练在占卡，先用
CUDA_VISIBLE_DEVICES 指定空闲卡，或设 EXEC_GPUS=1。
"""
import argparse
import copy
import json
import os
import re
import sys
from typing import Dict, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.environ.get('GEN_MODEL_ID', 'Qwen/Qwen3-4B')
GPU_MEM = float(os.environ.get('GEN_GPU_MEM', 0.8))
EXEC_GPUS = int(os.environ.get('EXEC_GPUS', 2))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 16384))
DEFAULT_EVAL_RECORDS = os.path.join(SCRIPT_DIR, '..', 'skill_v2', 'eval_records.jsonl')

# 与本文件夹 10 个 case 对应的难题（executor 全程持续解不出、baseline 也全 0）。
DEFAULT_DATA_IDS = ['seam:val:81', 'seam:val:92', 'seam:val:148', 'seam:val:128', 'seam:val:127',
                    'seam:val:29', 'seam:val:94', 'seam:val:176', 'seam:val:12', 'seam:val:151']

# ===========================================================================
# executor prompt / answer format —— 逐字复刻 train_skill_v2.py 的 v2 分支
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


# ===========================================================================
# boxed 抽取 + SEAM lpem 风格数值判分 —— 逐字复刻
# ===========================================================================
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


def _parse_seq(seq, gold: str) -> Dict:
    text = _clean_text(getattr(seq, 'decoded', '') or '')
    raw = extract_boxed(text)
    pred = _seam_sanitize(raw) if raw else None
    correct = bool(pred) and (pred == _seam_sanitize(str(gold)))
    terminated = getattr(seq, 'stop_reason', None) != 'length'
    return {'pred': pred, 'correct': correct, 'terminated': terminated,
            'stop_reason': getattr(seq, 'stop_reason', None),
            'gen_tokens': len(getattr(seq, 'tokens', None) or []), 'text': text}


def _run_samples(sampler, prompts, max_tokens, gen_dp):
    """greedy（T=0）单样本，对齐 v2 eval。gen_dp>len 时按最后一条 padding 补齐。"""
    if not prompts:
        return []
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0, num_samples=1)
    padded = prompts
    if gen_dp > 1 and 0 < len(prompts) < gen_dp:
        padded = prompts + [copy.deepcopy(prompts[-1]) for _ in range(gen_dp - len(prompts))]
    responses = sampler.sample(padded, params)[:len(prompts)]
    return [list(r.sequences) if (r and r.sequences) else [] for r in responses]


# ===========================================================================
# problem 查表 + trials 载入
# ===========================================================================
def load_problems(eval_records_path) -> Dict[str, Dict]:
    probs: Dict[str, Dict] = {}
    if not os.path.exists(eval_records_path):
        return probs
    for line in open(eval_records_path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('record_type') != 'eval_problem':
            continue
        did = r.get('data_id')
        if did and did not in probs:
            probs[did] = {'problem': r['problem'], 'reference_answer': r['reference_answer']}
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', default=os.path.join(SCRIPT_DIR, 'trials.jsonl'))
    ap.add_argument('--eval-records', default=DEFAULT_EVAL_RECORDS)
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'winning_skills.jsonl'))
    ap.add_argument('--max-tokens', type=int, default=8192, help='对齐 v2 eval 的 --max-tokens')
    ap.add_argument('--only', default=None, help='只跑某个 data_id')
    ap.add_argument('--init', action='store_true', help='生成 trials.jsonl 模板后退出')
    ap.add_argument('--dump-text', action='store_true', help='把每个 trial 的 executor 全文落到 probe_texts/')
    args = ap.parse_args()

    problems = load_problems(args.eval_records)

    if args.init:
        with open(args.trials, 'w') as f:
            f.write('# 每行一个 trial；# 开头的行被忽略。skill="" 即 baseline。改 skill 后重跑本脚本。\n')
            for did in DEFAULT_DATA_IDS:
                p = problems.get(did, {})
                f.write(json.dumps({'data_id': did, 'tag': 'baseline', 'skill': '',
                                    'reference_answer': p.get('reference_answer')}, ensure_ascii=False) + '\n')
        print(f'已写模板 {args.trials}（{len(DEFAULT_DATA_IDS)} 题，skill 待填）。编辑后去掉 --init 再跑。')
        return

    if not os.path.exists(args.trials):
        print(f'找不到 {args.trials}，先跑：python3 eval_skill_probe.py --init')
        sys.exit(1)

    trials = []
    for line in open(args.trials):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        trials.append(json.loads(line))
    if args.only:
        trials = [t for t in trials if t.get('data_id') == args.only]
    for t in trials:
        p = problems.get(t.get('data_id'), {})
        t.setdefault('problem', p.get('problem'))
        t.setdefault('reference_answer', p.get('reference_answer'))
    skipped = [t for t in trials if t.get('problem') is None]
    for t in skipped:
        print(f'[warn] data_id={t.get("data_id")} 无题面（eval_records 找不到且行内未给 problem），跳过')
    trials = [t for t in trials if t.get('problem') is not None]
    if not trials:
        print('没有可跑的 trial。')
        return

    # 仅起一个 executor sampler（enable_thinking=True，对齐 v2 eval 的 base_sampler）
    twinkle.initialize(mode='ray', nproc_per_node=EXEC_GPUS, lazy_collect=False,
                       groups=[DeviceGroup(name='exec', ranks=list(range(EXEC_GPUS)), device_type='GPU')])
    sampler = vLLMSampler(model_id=MODEL_ID,
                          engine_args={'gpu_memory_utilization': GPU_MEM,
                                       'max_model_len': MAX_MODEL_LEN, 'tensor_parallel_size': 1},
                          device_mesh=DeviceMesh.from_sizes(world_size=EXEC_GPUS, dp_size=EXEC_GPUS),
                          remote_group='exec')
    sampler.set_template(Template, model_id=MODEL_ID, enable_thinking=True, max_length=MAX_MODEL_LEN)

    prompts = [build_skill_solve_prompt(t['problem'], t.get('skill', '')) for t in trials]
    outs = _run_samples(sampler, prompts, args.max_tokens, EXEC_GPUS)

    if args.dump_text:
        os.makedirs(os.path.join(SCRIPT_DIR, 'probe_texts'), exist_ok=True)

    n_ok = 0
    print('\n' + '=' * 94)
    print('%-16s %-10s %-4s %-10s %-10s %-7s %s' % ('data_id', 'tag', '对?', 'pred', 'gold', 'tokens', 'note'))
    print('-' * 94)
    win_f = open(args.out, 'a')
    for idx, (t, seqs) in enumerate(zip(trials, outs)):
        roll = _parse_seq(seqs[0], t['reference_answer']) if seqs else {
            'pred': None, 'correct': False, 'terminated': False,
            'stop_reason': 'empty', 'gen_tokens': 0, 'text': ''}
        ok = '✓' if roll['correct'] else '✗'
        note = '[截断]' if roll['stop_reason'] == 'length' else ''
        if roll['correct']:
            n_ok += 1
        print('%-16s %-10s %-4s %-10s %-10s %-7d %s' % (
            t.get('data_id', ''), str(t.get('tag', ''))[:10], ok,
            str(roll['pred'])[:10], str(t['reference_answer'])[:10], roll['gen_tokens'], note))
        if args.dump_text:
            fn = os.path.join(SCRIPT_DIR, 'probe_texts',
                              'trial_%02d_%s_%s.txt' % (idx, str(t.get('data_id', '')).replace(':', '_'),
                                                        str(t.get('tag', ''))))
            with open(fn, 'w') as tf:
                tf.write('SKILL:\n' + (t.get('skill') or '') + '\n\n' + '=' * 60 + '\nEXECUTOR OUTPUT:\n' + roll['text'])
        if roll['correct']:
            win_f.write(json.dumps({'data_id': t.get('data_id'), 'tag': t.get('tag'),
                                    'reference_answer': t['reference_answer'], 'pred': roll['pred'],
                                    'gen_tokens': roll['gen_tokens'], 'max_tokens': args.max_tokens,
                                    'skill': t.get('skill', '')}, ensure_ascii=False) + '\n')
    win_f.close()
    print('-' * 94)
    print(f'共 {len(trials)} 个 trial，解对 {n_ok} 个。成功的 skill 已追加到 {args.out}')


if __name__ == '__main__':
    main()
