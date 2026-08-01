#!/usr/bin/env python3
"""logp_corr_probe.py — 找"低噪声、强预测 skill 好坏"的指标（E15 数据驱动 reward 选型）。

问题：E15 用 mean ΔlogP(GT|题+skill) 当稠密 reward，20 步 delta 不爬。要回答两件事：
  1) ΔlogP（及各种 per-token 聚合变体）到底和 skill 的真实有效性（executor 多 rollout
     通过率）相关吗？相关性多强？
  2) 各候选指标的噪声多大？（greedy×1 判分 vs 8-rollout 真值 的一致性 = 老 0/1 reward 的噪声）

数据：E15 gen_records（题/skill/GT 参考解/训练期 fp32 mean logps 都在盘上）。
三阶段（分开进程跑，互不污染 Ray/vllm）：
  --phase rollout   twinkle vLLMSampler dp=8：每对 (题,skill) T=0.5×8 rollout + greedy×1，
                    外加每题 baseline（无 skill）同口径 → 真值 pass_rate / lift。
  --phase logps     原生 vllm prompt_logprobs=0 + twinkle Template.encode 的 labels 定位
                    response 段（与训练 _score_executor_mean_logps 同一模板/同一切位），
                    对 base(题+GT) 与 skill(题+skill+GT) 各算一遍逐 token logp → npz。
  --phase analyze   CPU：各指标 vs 真值的 Spearman/AUC（全局 + 组内），噪声对比表。

用法（8 卡空闲时）：
  cd cookbook/exp/skill2lora
  PYTHONPATH=../../src python3 logp_corr_probe.py --phase rollout
  PYTHONPATH=../../src python3 logp_corr_probe.py --phase logps
  PYTHONPATH=../../src python3 logp_corr_probe.py --phase analyze
"""
import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E15_DIR = os.path.join(SCRIPT_DIR, 'output.ablate12', 'E15_logp_gt_on_narrative')
OUT_DIR = os.path.join(SCRIPT_DIR, 'logp_corr')
MODEL_ID = os.environ.get('GEN_MODEL_ID', 'Qwen/Qwen3-4B')
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 16384))
MAX_TOKENS = 8192          # executor 解题预算，对齐 v2
N_PROBLEMS = int(os.environ.get('PROBE_PROBLEMS', 64))
N_ROLLOUTS = int(os.environ.get('PROBE_ROLLOUTS', 8))
SEED = 42

# ---- executor prompt / 判分：逐字复刻 train_skill_v2 v2 分支（与 eval_skill_probe 相同） ----
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
    # bugfix 2026-07-29：\dfrac/\tfrac/\cfrac 归一，同 train_skill_v2._seam_sanitize
    txt = txt.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac').replace(r'\cfrac', r'\frac')
    txt = re.sub(r'\\frac\s*\{\s*([^}]+?)\s*}\s*\{\s*([^}]+?)\s*}', r'\1/\2', txt)
    if (m := _SEAM_FRAC_RE.search(txt)):
        p, q = map(float, m.groups())
        if q:
            return _seam_norm(str(p / q))
    if (m := _SEAM_NUM_RE.search(txt)):
        return _seam_norm(m.group())
    return txt


def _judge(decoded, gold):
    text = _SPECIAL_TOKEN_RE.sub('', decoded or '').rstrip()
    raw = extract_boxed(text)
    pred = _seam_sanitize(raw) if raw else None
    return bool(pred) and (pred == _seam_sanitize(str(gold)))


# ---- 配对采样：E15 gen_records -> pairs.jsonl -----------------------------------------
def load_pairs():
    """64 题（seeded）× 组内全部 parseable 且有 delta 的候选（<=8）；GT 取每题首候选 rolls[0].text。"""
    by_id = {}
    for line in open(os.path.join(E15_DIR, 'gen_records.jsonl')):
        r = json.loads(line)
        if r.get('record_type') != 'problem' or not r.get('candidates'):
            continue
        by_id.setdefault(r['data_id'], r)   # data_id 在 epoch 内唯一
    ids = sorted(by_id)
    rng = np.random.RandomState(SEED)
    pick = list(rng.permutation(len(ids))[:N_PROBLEMS])
    pairs, problems = [], []
    for k in pick:
        r = by_id[ids[k]]
        gt = next((c['rolls'][0]['text'] for c in r['candidates']
                   if c.get('rolls') and c['rolls'][0].get('text')), '')
        if not gt:
            continue
        cands = [c for c in r['candidates'] if c['parseable'] and c.get('logp_delta') is not None]
        if len(cands) < 4:
            continue
        problems.append({'data_id': r['data_id'], 'problem': r['problem'],
                         'reference_answer': r['reference_answer'], 'gt': gt})
        for j, c in enumerate(cands[:8]):
            pairs.append({'pair_id': f'{r["data_id"]}#{j}', 'data_id': r['data_id'],
                          'skill': c['skills'], 'leaked': bool(c['leaked']),
                          'skill_chars': len(c['skills']),
                          'skillgen_tokens': c.get('skillgen_tokens'),
                          'logp_base_train': c['logp_base'], 'logp_skill_train': c['logp_skill'],
                          'logp_delta_train': c['logp_delta']})
    return problems, pairs


# ---- phase: rollout -------------------------------------------------------------------
def phase_rollout():
    import twinkle
    from twinkle import DeviceGroup, DeviceMesh
    from twinkle.data_format import SamplingParams
    from twinkle.sampler import vLLMSampler
    from twinkle.template import Template

    problems, pairs = load_pairs()
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(problems, open(os.path.join(OUT_DIR, 'problems.json'), 'w'))
    with open(os.path.join(OUT_DIR, 'pairs.jsonl'), 'w') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f'[rollout] problems={len(problems)} pairs={len(pairs)}', flush=True)

    n_gpu = int(os.environ.get('EXEC_GPUS', 8))
    twinkle.initialize(mode='ray', nproc_per_node=n_gpu, lazy_collect=False,
                       groups=[DeviceGroup(name='exec', ranks=list(range(n_gpu)), device_type='GPU')])
    sampler = vLLMSampler(model_id=MODEL_ID,
                          engine_args={'gpu_memory_utilization': 0.85,
                                       'max_model_len': MAX_MODEL_LEN, 'tensor_parallel_size': 1},
                          device_mesh=DeviceMesh.from_sizes(world_size=n_gpu, dp_size=n_gpu),
                          remote_group='exec')
    sampler.set_template(Template, model_id=MODEL_ID, enable_thinking=True, max_length=MAX_MODEL_LEN)

    prob_by = {p['data_id']: p for p in problems}
    prompts, metas = [], []
    for p in problems:                                        # baseline（无 skill）
        prompts.append(build_direct_prompt(p['problem']))
        metas.append(('base', p['data_id']))
    for pr in pairs:                                          # with-skill
        prompts.append(build_skill_solve_prompt(prob_by[pr['data_id']]['problem'], pr['skill']))
        metas.append(('skill', pr['pair_id']))

    def run(params, tag):
        padded = prompts if len(prompts) % n_gpu == 0 else \
            prompts + [copy.deepcopy(prompts[-1])] * (n_gpu - len(prompts) % n_gpu)
        outs = sampler.sample(padded, params)[:len(prompts)]
        rows = []
        for (kind, key), resp in zip(metas, outs):
            gold = prob_by[key.split('#')[0]]['reference_answer'] if '#' in key \
                else prob_by[key]['reference_answer']
            seqs = list(resp.sequences) if (resp and resp.sequences) else []
            rows.append({'kind': kind, 'key': key, 'mode': tag,
                         'n': len(seqs),
                         'pass': [bool(_judge(getattr(s, 'decoded', '') or '', gold)) for s in seqs],
                         'trunc': [getattr(s, 'stop_reason', None) == 'length' for s in seqs],
                         'tokens': [len(getattr(s, 'tokens', None) or []) for s in seqs]})
        return rows

    rows = run(SamplingParams(max_tokens=MAX_TOKENS, temperature=0.5, top_p=1.0,
                              num_samples=N_ROLLOUTS), f't05x{N_ROLLOUTS}')
    rows += run(SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0,
                               num_samples=1), 'greedy')
    with open(os.path.join(OUT_DIR, 'rollout_results.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'[rollout] done: {len(rows)} rows -> rollout_results.jsonl', flush=True)


# ---- phase: logps ---------------------------------------------------------------------
def phase_logps():
    """原生 vllm prompt_logprobs=0；token 布局与训练一致：twinkle Template.encode 的 labels
    != -100 即 response(GT) 段位置。base 与 skill 两条轨迹各存一行 float32。"""
    from twinkle.template import Template
    from vllm import LLM, SamplingParams as VSP
    from vllm.inputs import TokensPrompt

    problems = json.load(open(os.path.join(OUT_DIR, 'problems.json')))
    pairs = [json.loads(l) for l in open(os.path.join(OUT_DIR, 'pairs.jsonl'))]
    prob_by = {p['data_id']: p for p in problems}
    tmpl = Template(model_id=MODEL_ID, enable_thinking=True, max_length=MAX_MODEL_LEN,
                    truncation_strategy='delete')

    def encode(problem, skill, gt):
        msgs = [dict(m) for m in build_skill_solve_prompt(problem, skill)['messages']]
        enc = tmpl.encode({'messages': msgs + [{'role': 'assistant', 'content': gt}],
                           'user_data': {'key_rounds': [len(msgs)]}})
        if enc is None:
            return None, None
        ids = [int(x) for x in enc['input_ids']]   # numpy int64 -> int（vllm msgspec 拒收 np 类型）
        pos = np.where(np.asarray(enc['labels']) != -100)[0]
        return ids, pos

    jobs, keys = [], []          # key: ('base', data_id) / ('skill', pair_id)
    for p in problems:
        ids, pos = encode(p['problem'], '', p['gt'])
        if ids is not None and len(pos):
            jobs.append((ids, pos))
            keys.append(('base', p['data_id']))
    for pr in pairs:
        p = prob_by[pr['data_id']]
        ids, pos = encode(p['problem'], pr['skill'], p['gt'])
        if ids is not None and len(pos):
            jobs.append((ids, pos))
            keys.append(('skill', pr['pair_id']))
    print(f'[logps] encoded jobs={len(jobs)} (skipped {len(problems)+len(pairs)-len(jobs)})', flush=True)

    llm = LLM(model=_local_model_path(), max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.85, tensor_parallel_size=1)
    sp = VSP(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids, _ in jobs], sp)

    store = {}
    for (ids, pos), (kind, key), out in zip(jobs, keys, outs):
        plp = out.prompt_logprobs
        row = np.full(len(pos), np.nan, dtype=np.float32)
        for i, p_ in enumerate(pos):
            d = plp[int(p_)] if int(p_) < len(plp) else None
            if d:
                lp = d.get(ids[int(p_)])
                if lp is not None:
                    row[i] = lp.logprob
        store[f'{kind}|{key}'] = row
    np.savez_compressed(os.path.join(OUT_DIR, 'token_logps.npz'), **store)
    print(f'[logps] saved {len(store)} rows -> token_logps.npz', flush=True)


def _local_model_path():
    from modelscope.hub.snapshot_download import snapshot_download
    return snapshot_download(MODEL_ID, local_files_only=True)


# ---- phase: analyze -------------------------------------------------------------------
def _rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):                      # 平均并列名次
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    ra, rb = _rank(a[m]), _rank(b[m])
    sa, sb = ra.std(), rb.std()
    return np.nan if sa == 0 or sb == 0 else float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    m = ~np.isnan(score)
    score, label = score[m], label[m]
    if label.sum() == 0 or (~label).sum() == 0:
        return np.nan
    r = _rank(score)
    return float((r[label].sum() - label.sum() * (label.sum() - 1) / 2) / (label.sum() * (~label).sum()))


def phase_analyze():
    problems = json.load(open(os.path.join(OUT_DIR, 'problems.json')))
    pairs = [json.loads(l) for l in open(os.path.join(OUT_DIR, 'pairs.jsonl'))]
    rolls = [json.loads(l) for l in open(os.path.join(OUT_DIR, 'rollout_results.jsonl'))]
    npz = np.load(os.path.join(OUT_DIR, 'token_logps.npz'))

    base_pass, pair_pass, pair_greedy, base_greedy, pair_trunc = {}, {}, {}, {}, {}
    for r in rolls:
        rate = float(np.mean(r['pass'])) if r['n'] else np.nan
        if r['kind'] == 'base' and r['mode'].startswith('t05'):
            base_pass[r['key']] = rate
        elif r['kind'] == 'skill' and r['mode'].startswith('t05'):
            pair_pass[r['key']] = rate
            pair_trunc[r['key']] = float(np.mean(r['trunc'])) if r['n'] else np.nan
        elif r['kind'] == 'skill' and r['mode'] == 'greedy':
            pair_greedy[r['key']] = float(r['pass'][0]) if r['n'] else np.nan
        elif r['kind'] == 'base' and r['mode'] == 'greedy':
            base_greedy[r['key']] = float(r['pass'][0]) if r['n'] else np.nan

    rows = []
    for pr in pairs:
        key, did = pr['pair_id'], pr['data_id']
        b = npz[f'base|{did}'] if f'base|{did}' in npz.files else None
        s = npz[f'skill|{key}'] if f'skill|{key}' in npz.files else None
        if key not in pair_pass or did not in base_pass:
            continue
        row = {'pair_id': key, 'data_id': did,
               'truth_pass8': pair_pass[key], 'truth_lift': pair_pass[key] - base_pass[did],
               'base_pass8': base_pass[did], 'greedy1': pair_greedy.get(key, np.nan),
               'trunc_rate': pair_trunc.get(key, np.nan),
               'delta_train': pr['logp_delta_train'],
               'leaked': float(pr['leaked']), 'skill_chars': float(pr['skill_chars'])}
        if b is not None and s is not None and len(b) == len(s):
            d = s - b
            ok = ~(np.isnan(d))
            d, bb = d[ok], b[ok]
            if len(d):
                row['delta_mean'] = float(d.mean())
                row['delta_sum'] = float(d.sum())
                k = min(50, len(d))
                row['delta_top50'] = float(d[np.argsort(-np.abs(d))[:k]].mean())
                unc = bb < -1.0                       # executor 本来拿不准的 token
                row['delta_uncertain'] = float(d[unc].mean()) if unc.sum() >= 5 else np.nan
                row['delta_tail100'] = float(d[-min(100, len(d)):].mean())
                row['frac_improved'] = float((d > 0).mean())
                row['base_mean_ck'] = float(bb.mean())
        rows.append(row)
    print(f'[analyze] usable pairs={len(rows)}')
    json.dump(rows, open(os.path.join(OUT_DIR, 'pair_table.json'), 'w'))

    # 交叉校验：vllm 重算 delta vs 训练 fp32 delta
    dm = [r.get('delta_mean', np.nan) for r in rows]
    dt = [r['delta_train'] for r in rows]
    print(f'\n[校验] corr(delta_vllm, delta_train) spearman={spearman(dm, dt):.3f}')

    metrics = ['delta_train', 'delta_mean', 'delta_sum', 'delta_top50', 'delta_uncertain',
               'delta_tail100', 'frac_improved', 'greedy1', 'skill_chars', 'leaked', 'trunc_rate']
    truth = np.array([r['truth_pass8'] for r in rows])
    lift = np.array([r['truth_lift'] for r in rows])
    helped = lift > 0

    print('\n=== 全局相关性（n=%d 对）：指标 vs 8-rollout 真值 ===' % len(rows))
    print('%-16s %-14s %-14s %-10s' % ('metric', 'sp(pass8)', 'sp(lift)', 'AUC(lift>0)'))
    for m in metrics:
        v = np.array([r.get(m, np.nan) for r in rows], float)
        print('%-16s %-14s %-14s %-10s' % (
            m, f'{spearman(v, truth):+.3f}', f'{spearman(v, lift):+.3f}', f'{auc(v, helped):.3f}'))

    # 组内（GRPO 真正用的信号）：每题 >=4 候选的组内 spearman 均值
    print('\n=== 组内相关性（每题组内 spearman 的均值±se）===')
    by_p = defaultdict(list)
    for r in rows:
        by_p[r['data_id']].append(r)
    for m in metrics:
        cs = []
        for did, rs in by_p.items():
            if len(rs) < 4:
                continue
            v = [r.get(m, np.nan) for r in rs]
            t = [r['truth_pass8'] for r in rs]
            c = spearman(v, t)
            if not np.isnan(c):
                cs.append(c)
        if cs:
            cs = np.array(cs)
            print('%-16s mean=%+.3f  se=%.3f  n_groups=%d' % (m, cs.mean(), cs.std() / np.sqrt(len(cs)), len(cs)))

    # 噪声对比：greedy×1 vs 真值；基线 greedy vs 基线8rollout
    g = np.array([r.get('greedy1', np.nan) for r in rows], float)
    m = ~np.isnan(g)
    hard_wrong = np.abs(g[m] - truth[m]) > 0.5
    print(f'\n[噪声] greedy×1 与 8-rollout 真值强不一致率（|diff|>0.5）: {hard_wrong.mean():.3f} (n={m.sum()})')
    bg = np.array([base_greedy.get(p["data_id"], np.nan) for p in problems], float)
    bp = np.array([base_pass.get(p["data_id"], np.nan) for p in problems], float)
    mm = ~(np.isnan(bg) | np.isnan(bp))
    print(f'[噪声] baseline greedy 与 baseline pass8 强不一致率: {(np.abs(bg[mm]-bp[mm])>0.5).mean():.3f} (n={mm.sum()})')
    print(f'[分布] truth_pass8 mean={np.nanmean(truth):.3f}  lift>0 比例={np.mean(helped):.3f}  '
          f'lift<0 比例={np.mean(lift<0):.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('rollout', 'logps', 'analyze'), required=True)
    args = ap.parse_args()
    {'rollout': phase_rollout, 'logps': phase_logps, 'analyze': phase_analyze}[args.phase]()


if __name__ == '__main__':
    main()
