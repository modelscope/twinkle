#!/usr/bin/env python3
"""analyze_more.py — logp_corr 数据的补充相关性分析（7 项此前未做的）。纯 CPU。
用法：/usr/local/bin/python3 analyze_more.py
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logp_corr')
pairs = [json.loads(l) for l in open(os.path.join(D, 'pairs.jsonl'))]
problems = json.load(open(os.path.join(D, 'problems.json')))
rolls = [json.loads(l) for l in open(os.path.join(D, 'rollout_results.jsonl'))]
npz = np.load(os.path.join(D, 'token_logps.npz'))

pas = {r['key']: float(np.mean(r['pass'])) for r in rolls if r['kind'] == 'skill' and r['mode'].startswith('t05')}
tok = {r['key']: float(np.mean(r['tokens'])) for r in rolls if r['kind'] == 'skill' and r['mode'].startswith('t05')}
npass = {r['key']: int(np.sum(r['pass'])) for r in rolls if r['kind'] == 'skill' and r['mode'].startswith('t05')}
gr = {r['key']: float(r['pass'][0]) for r in rolls if r['kind'] == 'skill' and r['mode'] == 'greedy' and r['n']}
bpas = {r['key']: float(np.mean(r['pass'])) for r in rolls if r['kind'] == 'base' and r['mode'].startswith('t05')}
btok = {r['key']: float(np.mean(r['tokens'])) for r in rolls if r['kind'] == 'base' and r['mode'].startswith('t05')}
prob_by = {p['data_id']: p for p in problems}


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


# ============ ① 题目侧特征 -> skill 收益（筛题实证） ============
print('=' * 70)
print('① 题目侧特征 -> skill 平均收益（n=61 题；spearman 跨题）')
by_p = defaultdict(list)
for pr in pairs:
    if pr['pair_id'] in pas:
        by_p[pr['data_id']].append(pr)
prows = []
for did, prs in by_p.items():
    p = prob_by[did]
    lv = int(did.split(':')[1])
    lifts = [pas[x['pair_id']] - bpas[did] for x in prs]
    passes = [pas[x['pair_id']] for x in prs]
    prows.append({'level': lv, 'base_pass': bpas[did], 'base_tok': btok[did],
                  'prob_chars': len(p['problem']), 'gt_chars': len(p['gt']),
                  'mean_lift': float(np.mean(lifts)), 'grp_std': float(np.std(passes)),
                  'frac_helped': float(np.mean([l > 0 for l in lifts]))})
for fk in ['level', 'base_pass', 'base_tok', 'prob_chars', 'gt_chars']:
    v = [r[fk] for r in prows]
    print('%-12s vs mean_lift %+0.3f | vs 组可分性(grp_std) %+0.3f | vs frac_helped %+0.3f' % (
        fk, sp(v, [r['mean_lift'] for r in prows]), sp(v, [r['grp_std'] for r in prows]),
        sp(v, [r['frac_helped'] for r in prows])))
bt = np.array([r['base_tok'] for r in prows])
ml = np.array([r['mean_lift'] for r in prows])
for lo, hi in [(0, 3000), (3000, 5000), (5000, 9999)]:
    m = (bt >= lo) & (bt < hi)
    if m.sum():
        print('  base_tok[%d,%d): n=%d  mean_lift=%+.3f  frac(lift>0)=%.2f' % (
            lo, hi, m.sum(), ml[m].mean(), np.mean([r['frac_helped'] for r, mm in zip(prows, m) if mm])))

# ============ ② skill 生成 think 长度 -> 好坏 ============
print('\n' + '=' * 70)
print('② skillgen_tokens（skill 生成总 token 含 think）组内 vs pass/输出长')
cs1, cs2 = [], []
for did, prs in by_p.items():
    if len(prs) < 4:
        continue
    sg = [x.get('skillgen_tokens') or np.nan for x in prs]
    c = sp(sg, [pas[x['pair_id']] for x in prs])
    if not np.isnan(c):
        cs1.append(c)
    c = sp(sg, [tok[x['pair_id']] for x in prs])
    if not np.isnan(c):
        cs2.append(c)
print('  vs pass8: mean=%+.3f se=%.3f n=%d' % (np.mean(cs1), np.std(cs1) / np.sqrt(len(cs1)), len(cs1)))
print('  vs exec_tokens: mean=%+.3f se=%.3f n=%d' % (np.mean(cs2), np.std(cs2) / np.sqrt(len(cs2)), len(cs2)))

# ============ ③ |delta| 当干预强度计：|delta| vs |lift| ============
print('\n' + '=' * 70)
print('③ |ΔlogP| 是否预测"干预幅度"|lift|（不看方向）')
ad, al, dtk = [], [], []
for pr in pairs:
    k = pr['pair_id']
    if k not in pas or pr.get('logp_delta_train') is None:
        continue
    ad.append(abs(pr['logp_delta_train']))
    al.append(abs(pas[k] - bpas[pr['data_id']]))
    dtk.append(abs(tok[k] - btok[pr['data_id']]))
print('  |delta| vs |lift| 全局 sp=%+.3f (n=%d)' % (sp(ad, al), len(ad)))
print('  |delta| vs |Δexec_tokens| 全局 sp=%+.3f' % sp(ad, dtk))
cs = []
for did, prs in by_p.items():
    if len(prs) < 4:
        continue
    a = [abs(x['logp_delta_train']) for x in prs]
    b = [abs(pas[x['pair_id']] - bpas[did]) for x in prs]
    c = sp(a, b)
    if not np.isnan(c):
        cs.append(c)
print('  组内: mean=%+.3f se=%.3f n=%d' % (np.mean(cs), np.std(cs) / np.sqrt(len(cs)), len(cs)))

# ============ ④ delta 的位置衰减：skill 影响是否集中在 GT 前段 ============
print('\n' + '=' * 70)
print('④ per-token delta 的位置分布（四分位段的 mean|delta|，跨 476 对平均）')
qsum = np.zeros(4)
qcnt = 0
for pr in pairs:
    k, did = pr['pair_id'], pr['data_id']
    if f'base|{did}' not in npz.files or f'skill|{k}' not in npz.files:
        continue
    b, s = npz[f'base|{did}'], npz[f'skill|{k}']
    if len(b) != len(s) or len(b) < 40:
        continue
    d = np.abs(s - b)
    d = d[~np.isnan(d)]
    if len(d) < 40:
        continue
    qs = np.array_split(d, 4)
    qsum += np.array([q.mean() for q in qs])
    qcnt += 1
print('  Q1(前1/4)=%.4f  Q2=%.4f  Q3=%.4f  Q4(末1/4)=%.4f  (n=%d)' % (*(qsum / qcnt), qcnt))

# ============ ⑤ pass8 分布形态：混沌（U形/过散）还是二项噪声 ============
print('\n' + '=' * 70)
print('⑤ 混合组 (0<pass8<1) 的 k/8 直方图（U 形=题级混沌，钟形=独立二项）')
ks = [npass[k] for k in npass if 0 < npass[k] < 8]
hist = np.bincount(ks, minlength=9)[1:8]
print('  k=1..7: %s  (n=%d)' % (hist.tolist(), len(ks)))
# 对照：以每对自身 p=k/8 的独立二项，条件在 0<k<8 上的期望形状
exp = np.zeros(7)
for k in ks:
    p = k / 8
    probs = np.array([math.comb(8, i) * p**i * (1 - p)**(8 - i) for i in range(1, 8)])
    exp += probs / probs.sum()
print('  二项参照: %s' % np.round(exp, 1).tolist())

# ============ ⑥ greedy vs pass8 强分歧案例的特征（危险区验证） ============
print('\n' + '=' * 70)
print('⑥ greedy 与真值强分歧（|greedy-pass8|>0.5）案例 vs 其余：输出长度')
dis, rest = [], []
for k in pas:
    if k in gr:
        (dis if abs(gr[k] - pas[k]) > 0.5 else rest).append(tok[k])
print('  分歧组 n=%d mean_tok=%d p75=%d | 其余 n=%d mean_tok=%d p75=%d' % (
    len(dis), np.mean(dis), np.percentile(dis, 75), len(rest), np.mean(rest), np.percentile(rest, 75)))

# ============ ⑦ leak clean 分解：leak 到底贡献多少 lift ============
print('\n' + '=' * 70)
print('⑦ leak 分解（lift = pass8 - base_pass8）')
for name, cond in [('leaked', lambda p: p['leaked']), ('clean', lambda p: not p['leaked'])]:
    ls = [pas[p['pair_id']] - bpas[p['data_id']] for p in pairs if p['pair_id'] in pas and cond(p)]
    print('  %-7s n=%-4d mean_lift=%+.4f  frac(lift>0)=%.2f' % (
        name, len(ls), np.mean(ls), np.mean([x > 0 for x in ls])))
