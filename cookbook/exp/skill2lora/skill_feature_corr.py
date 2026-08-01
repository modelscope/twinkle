#!/usr/bin/env python3
"""skill_feature_corr.py — 探"什么样的 skill 文本特征决定 executor 好坏"。

对象：logp_corr/ 已有的 476 对 (题,skill) + 8-rollout 真值（pass_rate / trunc_rate / mean_tokens）。
在 skill 文本上抽一批可解释特征，与真值做组内 spearman（GRPO 实际用的口径）+ 全局 spearman。
纯 CPU，无 GPU。用法：/usr/local/bin/python3 skill_feature_corr.py
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'logp_corr')

pairs = [json.loads(l) for l in open(os.path.join(OUT, 'pairs.jsonl'))]
rolls = [json.loads(l) for l in open(os.path.join(OUT, 'rollout_results.jsonl'))]
pas = {r['key']: float(np.mean(r['pass'])) for r in rolls if r['kind'] == 'skill' and r['mode'].startswith('t05')}
trc = {r['key']: float(np.mean(r['trunc'])) for r in rolls if r['kind'] == 'skill' and r['mode'].startswith('t05')}
tok = {r['key']: float(np.mean(r['tokens'])) for r in rolls if r['kind'] == 'skill' and r['mode'].startswith('t05')}

# ---- skill 文本特征 -------------------------------------------------------------------
_NUM = re.compile(r'\d')
_FORMULA = re.compile(r'[=+\-*/^]|\\frac|\\sqrt|\\sum|\\int|\$')
_STEP = re.compile(r'(?im)^\s*(step\s*\d|[0-9]+[.)]|first|second|third|next|then|finally)\b')
_IMPER = re.compile(r'(?i)\b(use|apply|consider|note|remember|compute|calculate|check|verify|'
                    r'identify|recall|avoid|ensure|find|start|begin|rewrite|simplify|substitute)\b')
_HEDGE = re.compile(r'(?i)\b(might|maybe|perhaps|possibly|could|may|try)\b')
_LATEX = re.compile(r'\\[a-zA-Z]+')
_PITFALL = re.compile(r'(?i)\b(mistake|error|pitfall|careful|caution|wrong|avoid|common|trap|'
                      r'incorrect|forget|overlook)\b')


def feats(skill):
    s = skill or ''
    words = s.split()
    nw = max(1, len(words))
    sents = [x for x in re.split(r'[.!?\n]', s) if x.strip()]
    return {
        'chars': len(s),
        'words': nw,
        'sent_len': nw / max(1, len(sents)),            # 平均句长（可读性）
        'num_density': len(_NUM.findall(s)) / nw,        # 数字密度（具体计算 vs 抽象）
        'formula_density': len(_FORMULA.findall(s)) / nw,
        'latex_density': len(_LATEX.findall(s)) / nw,
        'step_markers': len(_STEP.findall(s)),           # 步骤/编号结构
        'imper_density': len(_IMPER.findall(s)) / nw,    # 祈使动词（指令性）
        'hedge_density': len(_HEDGE.findall(s)) / nw,    # 模糊限定词（不确定）
        'pitfall_density': len(_PITFALL.findall(s)) / nw,
        'uniq_ratio': len(set(words)) / nw,              # 词汇多样性（低=啰嗦重复）
    }


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    ra, rb = rank(a[m]), rank(b[m])
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (ra.std() * rb.std()))


rows = []
for pr in pairs:
    k = pr['pair_id']
    if k not in pas:
        continue
    f = feats(pr['skill'])
    f.update({'pair_id': k, 'data_id': pr['data_id'], 'pass': pas[k],
              'trunc': trc.get(k, np.nan), 'tok': tok.get(k, np.nan),
              'leaked': float(pr['leaked'])})
    rows.append(f)

FKEYS = ['chars', 'words', 'sent_len', 'num_density', 'formula_density', 'latex_density',
         'step_markers', 'imper_density', 'hedge_density', 'pitfall_density', 'uniq_ratio']

print(f'[skill_feat] n_pairs={len(rows)}')
# 全局
print('\n=== 全局 spearman：skill 特征 vs 真值 ===')
print('%-16s %-10s %-10s %-10s' % ('feature', 'sp(pass)', 'sp(trunc)', 'sp(tok)'))
for fk in FKEYS:
    v = [r[fk] for r in rows]
    print('%-16s %+.3f     %+.3f     %+.3f' % (
        fk, sp(v, [r['pass'] for r in rows]),
        sp(v, [r['trunc'] for r in rows]), sp(v, [r['tok'] for r in rows])))

# 组内
by = defaultdict(list)
for r in rows:
    by[r['data_id']].append(r)
groups = [g for g in by.values() if len(g) >= 4]
print(f'\n=== 组内 spearman（每题内，n_groups={len(groups)}，mean±se）vs pass_rate ===')
for fk in FKEYS:
    cs = []
    for g in groups:
        c = sp([r[fk] for r in g], [r['pass'] for r in g])
        if not np.isnan(c):
            cs.append(c)
    if cs:
        cs = np.array(cs)
        print('%-16s mean=%+.3f  se=%.3f  n=%d' % (fk, cs.mean(), cs.std() / np.sqrt(len(cs)), len(cs)))

print(f'\n=== 组内 spearman vs trunc_rate（截断通道）===')
for fk in FKEYS:
    cs = []
    for g in groups:
        c = sp([r[fk] for r in g], [r['trunc'] for r in g])
        if not np.isnan(c):
            cs.append(c)
    if cs:
        cs = np.array(cs)
        print('%-16s mean=%+.3f  se=%.3f  n=%d' % (fk, cs.mean(), cs.std() / np.sqrt(len(cs)), len(cs)))
