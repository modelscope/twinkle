#!/usr/bin/env python3
"""rubric_effect.py — rubric 作用的三路数据分析（纯 CPU，现有数据）。
① 臂级：E5/E6/E7(rl_ab, rubric 条件) vs E1/E2/E3(bnpo, query-only) eval lift 对照
② 题级：同一错题上，rubric 条件生成的 skill vs query-only 生成的 skill 的 executor 通过率
③ rubric 文本特征 vs A 线拯救率（组内 any-pass）
"""
import json
import hashlib
import os
import re
from collections import defaultdict

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output.ablate12')

# gen_records 未存 rubric 文本；用全局缓存反查（key = md5('rubric_global\x1f'+data_id)）。
# 能查到 => 该题被诊断过 => 裸题答错、属 A 线（rl_ab 只诊断错题）。
_RUBRIC = {}
with open(os.path.join(BASE, 'rubric_cache_global.jsonl')) as f:
    for l in f:
        d = json.loads(l)
        _RUBRIC[d['key']] = d.get('value') or ''


def rubric_of(data_id):
    k = hashlib.md5(('\x1f'.join(['rubric_global', str(data_id)])).encode('utf-8')).hexdigest()
    return _RUBRIC.get(k)


def load_gen(exp):
    rows = []
    with open(os.path.join(BASE, exp, 'gen_records.jsonl')) as f:
        for l in f:
            r = json.loads(l)
            rows.append(r)
    return rows


def probe_schema(exp):
    rows = load_gen(exp)
    tps = defaultdict(int)
    for r in rows:
        tps[r.get('record_type')] += 1
    print(exp, dict(tps))
    for r in rows:
        if r.get('record_type') == 'problem':
            print(' problem keys:', sorted(r.keys())[:30])
            cands = r.get('cands') or r.get('_cands') or []
            if cands:
                print(' cand keys:', sorted(cands[0].keys()))
            break


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'schema':
        probe_schema('E7_rl_ab_on_pitfall')
        probe_schema('E3_bnpo_on_pitfall')
        sys.exit(0)

    # ---------- ② 题级同题对照 ----------
    # E7 A 线（rubric 非空）的候选 vs E3 同 data_id 的候选（query-only），配对比较组均值
    for pair in [('E7_rl_ab_on_pitfall', 'E3_bnpo_on_pitfall'),
                 ('E5_rl_ab_off_pitfall', 'E1_bnpo_off_pitfall'),
                 ('E6_rl_ab_off_narrative', 'E2_bnpo_off_narrative')]:
        ea, eb = pair
        ga, gb = load_gen(ea), load_gen(eb)

        def group_pass(rows, need_rubric=None):
            out = {}
            for r in rows:
                if r.get('record_type') != 'problem':
                    continue
                did = r.get('data_id', '')
                rub = rubric_of(did)
                if need_rubric is True and not rub:
                    continue
                cands = [c for c in (r.get('candidates') or []) if c.get('parseable')
                         and c.get('with_pass') is not None]
                if not cands:
                    continue
                # 同一题可能多 chunk 出现，取第一次（早期，policy 漂移最小）
                if did not in out:
                    out[did] = (np.mean([float(c['with_pass']) for c in cands]), rub or '')
            return out

        pa = group_pass(ga, need_rubric=True)   # A 线（rubric 条件）
        pb = group_pass(gb)                     # query-only
        common = sorted(set(pa) & set(pb))
        if not common:
            print(f'[②] {ea} vs {eb}: 无同题交集')
            continue
        da = np.array([pa[d][0] for d in common])
        db = np.array([pb[d][0] for d in common])
        diff = da - db
        print(f'[②] {ea.split("_")[0]}(rubric) vs {eb.split("_")[0]}(query-only) 同题 n={len(common)}: '
              f'rubric臂组均pass={da.mean():.3f} qonly臂={db.mean():.3f} '
              f'配对差={diff.mean():+.4f}±{diff.std()/np.sqrt(len(diff)):.4f} '
              f'win/tie/lose={int((diff>0).sum())}/{int((diff==0).sum())}/{int((diff<0).sum())}')

    # ---------- ③ rubric 特征 vs 拯救率 ----------
    ga = load_gen('E7_rl_ab_on_pitfall')
    rows = []
    for r in ga:
        if r.get('record_type') != 'problem':
            continue
        rub = (rubric_of(r.get('data_id', '')) or '').strip()
        if not rub:
            continue
        cands = [c for c in (r.get('candidates') or []) if c.get('parseable')
                 and c.get('with_pass') is not None]
        if not cands:
            continue
        n_fail = len(re.findall(r'\[FAIL\]', rub))
        n_pass = len(re.findall(r'\[PASS\]', rub))
        rows.append({'len': len(rub), 'n_fail': n_fail, 'n_pass': n_pass,
                     'n_crit': n_fail + n_pass,
                     'has_fix': int('fix:' in rub),
                     'rescue': float(np.mean([float(c['with_pass']) for c in cands])),
                     'any': float(any(c['with_pass'] for c in cands))})
    if rows:
        def sp(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            ra = np.argsort(np.argsort(a)).astype(float)
            rb = np.argsort(np.argsort(b)).astype(float)
            if ra.std() == 0 or rb.std() == 0:
                return np.nan
            return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (ra.std() * rb.std()))
        print(f'\n[③] E7 A线 rubric 特征 vs 拯救率 (n={len(rows)} 题, '
              f'mean rescue={np.mean([r["rescue"] for r in rows]):.3f}, '
              f'any-pass={np.mean([r["any"] for r in rows]):.3f})')
        for fk in ['len', 'n_fail', 'n_pass', 'n_crit', 'has_fix']:
            v = [r[fk] for r in rows]
            print('  %-8s vs rescue %+0.3f | vs any-pass %+0.3f' % (
                fk, sp(v, [r['rescue'] for r in rows]), sp(v, [r['any'] for r in rows])))
        ls = np.array([r['len'] for r in rows])
        print('  rubric len 分布: p25=%d p50=%d p75=%d' % tuple(np.percentile(ls, [25, 50, 75])))
