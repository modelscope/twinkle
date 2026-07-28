#!/usr/bin/env python3
# leak_decomp.py — 严格重审 "think 提升是否由答案泄漏驱动"。
# 上次教训: 宽松判定曾导致严重误判。本脚本做四路独立检验:
#   T1 安慰剂测试: 用"别题答案"跑同一 leak 规则 → 估计 leak 标记的偶然假阳性地板
#      (think skill 长且数字密集, 答案数字偶然出现的概率天然更高)
#   T2 条件分解: parseable 记录拆 leaked/clean, 分别算 acc + 占比 → 贡献分解
#   T3 反事实: 把 leaked 样本的 acc 替换成同组 clean acc → think 优势还剩多少
#   T4 严格判分: 对 withskill_text 用严格 boxed 精确匹配重新判分,
#      检验现行 grade(_seam_sanitize 取首数字等宽松步骤) 是否给 think 虚增 acc
# 难度控制: T2/T3 同时在 baseline_pass==0 (hard) 子集上重复, 排除"泄漏样本恰好是简单题"。
import json, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {'A_old_off': 'skillcfg_full_off.jsonl',
         'B_new_on':  'skillcfg_full_on.jsonl',
         'C_new_off': 'env_runs/vllm_0.23.0/skillcfg_full_off.jsonl'}
CFGS = ['P1_narrative','P2_combo','P3_toy','P4_card','P5_pitfall','P6_seam','P7_minimal']

_NUM = re.compile(r'-?\d+(\.\d+)?')

def sanitize(x):
    s = str(x).strip()
    m = _NUM.search(s)
    if m and m.group() == s:
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f)
        except Exception:
            pass
    return s

def leak_rule(skill, ans):
    """逐字复刻 skill_config_probe.answer_leaked 的数字边界规则"""
    if not skill:
        return False
    g = sanitize(ans)
    if not g or not re.fullmatch(r'-?\d+(\.\d+)?', g):
        return None  # 不适用(非纯数字答案)
    return bool(re.search(r'(?<![\d.])' + re.escape(g) + r'(?!\d)(?!\.\d)', skill))

_BOXED = re.compile(r'\\boxed\s*\{')

def last_boxed(text):
    last = None
    for m in _BOXED.finditer(text or ''):
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            depth += (text[i] == '{') - (text[i] == '}')
            i += 1
        if depth == 0:
            last = text[m.end():i-1].strip()
    return last

def strict_correct(text, gold):
    """严格口径: 最后一个 boxed 的内容去掉 latex 修饰后必须与 gold 精确相等(串或数值)。
    不做 '从文本中捞第一个数字' 这类宽松回退。"""
    raw = last_boxed(text)
    if raw is None:
        return False
    s = raw.replace('\\!','').replace('\\,','').replace('\\ ',' ')
    s = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)
    s = s.replace('$','').replace('{','').replace('}','').replace('\\','').strip()
    g = str(gold).strip()
    if s == g:
        return True
    try:
        return float(s) == float(g)
    except Exception:
        return False

def main():
    # 先取 200 题的答案表(placebo 用): data_id -> answer
    answers = {}
    with open(os.path.join(HERE, FILES['A_old_off'])) as f:
        for line in f:
            r = json.loads(line)
            answers.setdefault(r['data_id'], r['reference_answer'])
    dids = sorted(answers)
    placebo = {}
    for i, d in enumerate(dids):
        # 找下一个"值不同"的答案做安慰剂
        for j in range(1, len(dids)):
            cand = answers[dids[(i+j) % len(dids)]]
            if sanitize(cand) != sanitize(answers[d]):
                placebo[d] = cand
                break

    for grp, path in FILES.items():
        # 聚合器: cfg -> 统计
        S = defaultdict(lambda: defaultdict(float))
        with open(os.path.join(HERE, path)) as f:
            for line in f:
                r = json.loads(line)
                cfg = r['config']; s = S[cfg]
                skill = r['skill'] or ''
                corr = 1.0 if r['withskill_correct'] else 0.0
                hard = (r.get('baseline_pass') or 0) == 0
                s['n'] += 1
                # --- T4 严格判分 ---
                sc = 1.0 if strict_correct(r.get('withskill_text',''), r['reference_answer']) else 0.0
                s['acc_loose'] += corr; s['acc_strict'] += sc
                s['loose_only'] += 1.0 if (corr and not sc) else 0.0
                # --- T1 安慰剂 ---
                lk = leak_rule(skill, r['reference_answer'])
                if lk is not None and r['parseable']:
                    s['n_lk'] += 1
                    s['leak'] += 1.0 if lk else 0.0
                    pl = leak_rule(skill, placebo[r['data_id']])
                    s['placebo'] += 1.0 if pl else 0.0
                # --- T2 条件分解 ---
                if r['parseable']:
                    key = 'L' if r['leaked'] else 'Cn'
                    s[f'n_{key}'] += 1; s[f'acc_{key}'] += corr
                    if hard:
                        s[f'nh_{key}'] += 1; s[f'acch_{key}'] += corr
                else:
                    s['n_U'] += 1; s['acc_U'] += corr
                    if hard:
                        s['nh_U'] += 1; s['acch_U'] += corr
        print(f"\n{'='*100}\n### {grp}  ({path})")
        print("%-14s %6s | %7s %7s %9s | %5s %6s | %5s %6s | %5s %6s | %8s %8s %9s" % (
            'config','n','leak%','placebo%','净leak%','nL','accL','nCn','accCn','nU','accU','accLoose','accStrict','looseOnly%'))
        for cfg in CFGS:
            s = S[cfg]
            n = s['n'] or 1
            nlk = s['n_lk'] or 1
            lk, pl = s['leak']/nlk, s['placebo']/nlk
            aL  = s['acc_L']/s['n_L'] if s['n_L'] else float('nan')
            aC  = s['acc_Cn']/s['n_Cn'] if s['n_Cn'] else float('nan')
            aU  = s['acc_U']/s['n_U'] if s['n_U'] else float('nan')
            print("%-14s %6d | %7.3f %7.3f %9.3f | %5d %6.3f | %5d %6.3f | %5d %6.3f | %8.3f %8.3f %9.3f" % (
                cfg, s['n'], lk, pl, lk-pl, s['n_L'], aL, s['n_Cn'], aC, s['n_U'], aU,
                s['acc_loose']/n, s['acc_strict']/n, s['loose_only']/n))
        # hard 子集(排除"泄漏样本挑了简单题")
        print("  --- hard(baseline=0) 子集: leaked vs clean 的 acc ---")
        for cfg in CFGS:
            s = S[cfg]
            ahL = s['acch_L']/s['nh_L'] if s['nh_L'] else float('nan')
            ahC = s['acch_Cn']/s['nh_Cn'] if s['nh_Cn'] else float('nan')
            ahU = s['acch_U']/s['nh_U'] if s['nh_U'] else float('nan')
            print("    %-14s hard: leaked %4d/%.3f  clean %4d/%.3f  unparse %4d/%.3f" % (
                cfg, s['nh_L'], ahL, s['nh_Cn'], ahC, s['nh_U'], ahU))

if __name__ == '__main__':
    main()
