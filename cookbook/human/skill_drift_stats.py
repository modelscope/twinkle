#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""统计 E23 skill 随训练 step 的「长度」与「词频」漂移，量化「skill 是否越来越空泛」。

⚠️ 数据来源与偏差
------------------
本脚本读 output.e23/zero_reward_groups.jsonl —— 它**只落整组 reward=0 的题**（8 个候选全错）。
这不是全部 skill，是一个偏难的子集。所以：
  * 「长度随 step 涨」的结论对这个子集成立，但推广到全体 skill 时要记住这一点；
  * 词频漂移（guarantee->confirm 等）同理。
若要无偏统计需改 e23 落全量 skill；当前只有这一份带 step 标签的 skill 文本。

口径
----
* skill = 候选的 <skills> 块（extract_skill 已抽好，存在 candidates[].skills）。
* INSTEAD 段单独抽出来看动词，因为「空泛化」主要发生在这一段（WARNING 段是描述过去的错误）。
* 词频对比早期（step<=2）vs 晚期（step>=13）两窗，报 log2 比值最大的上升/下降词。
"""
import collections
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, 'output.e23', 'zero_reward_groups.jsonl')

# 英文停用词（够用即可，不引第三方）。
STOP = set('the a an of to in on for and or is are be it its this that with as by from at '
           'you your not no if then when will would should must can may a an s t re ve'.split())
INSTEAD_RE = re.compile(r'INSTEAD:\s*(.*?)(?:\n(?:Deliver|WARNING)|\Z)', re.S)
WARNING_RE = re.compile(r'WARNING:\s*(.*?)(?:\n(?:INSTEAD|Deliver)|\Z)', re.S)
# 空泛/认知性措辞：不要求 executor 改任何代码，只要求「认同一条命题」。
VAGUE_RE = re.compile(
    r'\b(confirm that|in any|for any|general principle|it may|might|unpredictab\w+|'
    r'violat\w+ the principle|be aware|keep in mind|note that|understand that|'
    r'is a general|in general|conceptually|principle)\b', re.I)
# 祈使/可执行动词：直接命令 executor 怎么写。
IMPER_RE = re.compile(r'^(use|set|derive|write|apply|replace|compute|return|add|remove|'
                      r'ensure|guarantee|call|pass|cast|convert|assign|initialize|import|'
                      r'check|handle|raise|match)\b', re.I)


def words(text):
    return [w for w in re.findall(r"[a-zA-Z_][a-zA-Z_']+", (text or '').lower())
            if w not in STOP and len(w) > 2]


def load():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    # 每个候选一条记录：(step, skill_text)
    out = []
    for r in rows:
        s = r['step']
        for c in r['candidates']:
            out.append((s, c.get('skills') or ''))
    return out, path


def instead_seg(sk):
    m = INSTEAD_RE.search(sk or '')
    return (m.group(1).strip() if m else '')


def warning_seg(sk):
    m = WARNING_RE.search(sk or '')
    return (m.group(1).strip() if m else '')


def per_step_table(data):
    by = collections.defaultdict(list)
    for s, sk in data:
        by[s].append(sk)
    steps = sorted(by)
    print('=' * 88)
    print('一、skill 长度与结构 随 step（每行 = 该 step 全部候选的均值；n=候选数）')
    print('=' * 88)
    print(f"{'step':>4} {'n':>4} {'skill字符':>9} {'skill词数':>9} {'INSTEAD词数':>11} "
          f"{'空泛词/条':>9} {'祈使开头%':>9}")
    rows_for_trend = []
    for s in steps:
        sks = by[s]
        chars = st.mean(len(x) for x in sks)
        nwords = st.mean(len(words(x)) for x in sks)
        ins = [instead_seg(x) for x in sks]
        inw = st.mean(len(i.split()) for i in ins) if ins else 0
        vague = st.mean(len(VAGUE_RE.findall(x)) for x in sks)
        imper = sum(1 for i in ins if IMPER_RE.match(i)) / len(ins) if ins else 0
        print(f'{s:>4} {len(sks):>4} {chars:>9.1f} {nwords:>9.1f} {inw:>11.1f} '
              f'{vague:>9.2f} {imper:>8.0%}')
        rows_for_trend.append((s, chars, vague, imper))
    return by, steps, rows_for_trend


def trend(rows_for_trend):
    """对 (step, y) 做最小二乘斜率 + t，判断长度/空泛度/祈使率是否真在漂移。"""
    print('\n' + '=' * 88)
    print('二、趋势显著性（OLS 斜率 / step，|t|>2 才算真漂移）')
    print('=' * 88)
    xs = [r[0] for r in rows_for_trend]
    n = len(xs)
    mx = st.mean(xs)
    sxx = sum((x - mx) ** 2 for x in xs)
    for idx, name in ((1, 'skill 字符数'), (2, '空泛词/条'), (3, 'INSTEAD 祈使开头率')):
        ys = [r[idx] for r in rows_for_trend]
        my = st.mean(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
        s2 = sum(e * e for e in resid) / (n - 2)
        se = math.sqrt(s2 / sxx) if sxx else 0.0
        t = slope / se if se else 0.0
        tag = '显著' if abs(t) > 2 else '不显著'
        print(f'  {name:20s} 斜率={slope:+.4f}/step  t={t:+.2f}  [{tag}]  '
              f'首={ys[0]:.3f} 尾={ys[-1]:.3f}')


def word_freq_shift(by, steps):
    """早期 vs 晚期两窗的词频比。"""
    print('\n' + '=' * 88)
    print('三、词频漂移：早期(step<=2) vs 晚期(step>=13)')
    print('=' * 88)
    early_steps = [s for s in steps if s <= 2]
    late_steps = [s for s in steps if s >= 13]

    def counts(win):
        c = collections.Counter()
        ntok = 0
        for s in win:
            for sk in by[s]:
                w = words(instead_seg(sk))     # 只看 INSTEAD 段
                c.update(w)
                ntok += len(w)
        return c, ntok
    ce, ne = counts(early_steps)
    cl, nl = counts(late_steps)
    print(f'早期窗 step={early_steps} INSTEAD总词={ne}；晚期窗 step={late_steps} 总词={nl}')
    # 频率（每千词），加平滑
    vocab = set(ce) | set(cl)
    rows = []
    for w in vocab:
        fe = (ce[w] + 0.5) / (ne + 1) * 1000
        fl = (cl[w] + 0.5) / (nl + 1) * 1000
        if ce[w] + cl[w] < 4:                  # 太稀疏的词不看
            continue
        rows.append((math.log2(fl / fe), w, ce[w], cl[w], fe, fl))
    rows.sort(reverse=True)
    print(f'\n{"↑晚期变多的词":22s}{"早/千":>8}{"晚/千":>8}{"log2比":>8}')
    for lr, w, e, l, fe, fl in rows[:15]:
        print(f'  {w:20s}{fe:8.1f}{fl:8.1f}{lr:+8.2f}')
    print(f'\n{"↓晚期变少的词":22s}{"早/千":>8}{"晚/千":>8}{"log2比":>8}')
    for lr, w, e, l, fe, fl in rows[-15:][::-1]:
        print(f'  {w:20s}{fe:8.1f}{fl:8.1f}{lr:+8.2f}')


def main():
    data, path = load()
    print(f'[数据] {path}')
    print(f'[数据] {len(data)} 个候选 skill（来自整组 reward=0 的题，是偏难子集，非全量）\n')
    by, steps, rows_for_trend = per_step_table(data)
    trend(rows_for_trend)
    word_freq_shift(by, steps)


if __name__ == '__main__':
    main()
