#!/usr/bin/env python3
# analyze_3way.py — 三路探针对比分析（A/B/C），全部结论可复现、可引用。
#   A = 老环境 nothink  : skillcfg_full_off.jsonl        + reflexion_full_off.jsonl        (根目录)
#   B = 新环境 think     : skillcfg_full_on.jsonl         + reflexion_full_on.jsonl         (根目录)
#   C = 新环境 nothink   : env_runs/vllm_0.23.0/skillcfg_full_off.jsonl + reflexion_full_off.jsonl
# 对比轴:
#   vLLM/环境影响 = A vs C (都 nothink)
#   think 影响    = C vs B (都新环境)
# 用法: python3 analyze_3way.py > analysis_out/report.txt 2>&1
import json, os, math, statistics as st, re, random
from collections import defaultdict, Counter

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {
    'A_old_nothink': ('skillcfg_full_off.jsonl', 'reflexion_full_off.jsonl'),
    'B_new_think':   ('skillcfg_full_on.jsonl',  'reflexion_full_on.jsonl'),
    'C_new_nothink': ('env_runs/vllm_0.23.0/skillcfg_full_off.jsonl',
                      'env_runs/vllm_0.23.0/reflexion_full_off.jsonl'),
}
SKILL_CFGS = ['P1_narrative','P2_combo','P3_toy','P4_card','P5_pitfall','P6_seam','P7_minimal']
REFL_CFGS  = ['R4_blind','D1_needle','D2_narr','D3_toyfix']


def load(path):
    """流式读取 -> list[dict]（只保留分析需要的字段，省内存）"""
    keep = ('config','data_id','sample_idx','baseline_pass','parseable','skill_chars',
            'leaked','skillgen_stop','skillgen_tokens','withskill_correct','withskill_stop',
            'withskill_tokens','skill')
    rows = []
    with open(os.path.join(HERE, path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({k: r.get(k) for k in keep})
    return rows


def agg_metrics(rows, cfgs):
    """返回 {config: metrics}"""
    out = {}
    for name in cfgs:
        sub = [t for t in rows if t['config'] == name]
        n = len(sub)
        if n == 0:
            continue
        parse = sum(bool(t['parseable']) for t in sub)/n
        leak  = sum(bool(t['leaked']) for t in sub)/n
        trunc = sum(1 for t in sub if t['withskill_stop']=='length')/n
        sg_trunc = sum(1 for t in sub if t['skillgen_stop']=='length')/n
        chars = [t['skill_chars'] for t in sub if t['parseable']]
        med_chars = int(st.median(chars)) if chars else 0
        acc  = sum(bool(t['withskill_correct']) for t in sub)/n
        base = sum(_f(t['baseline_pass']) for t in sub)/n
        # 题级 pass@k / hard 救活@k
        byq = defaultdict(list)
        for t in sub:
            byq[t['data_id']].append(t)
        p_at_k = sum(1 for v in byq.values() if any(x['withskill_correct'] for x in v))/len(byq)
        hardq = {d:v for d,v in byq.items() if _f(v[0]['baseline_pass'])==0}
        rescue = (sum(1 for v in hardq.values() if any(x['withskill_correct'] for x in v))/len(hardq)) if hardq else 0.0
        # 去混杂子集: leaked=0 且 parseable=1
        clean = [t for t in sub if (not t['leaked']) and t['parseable']]
        acc_clean = (sum(bool(t['withskill_correct']) for t in clean)/len(clean)) if clean else float('nan')
        out[name] = dict(n=n, parse=parse, leak=leak, trunc=trunc, sg_trunc=sg_trunc,
                         med_chars=med_chars, acc=acc, base=base, lift=acc-base,
                         p_at_k=p_at_k, rescue=rescue, n_clean=len(clean), acc_clean=acc_clean,
                         sg_tokens_med=int(st.median([t['skillgen_tokens'] or 0 for t in sub])))
    return out


def diversity(rows, cfgs):
    """题内 8 rollout 多样性: 去重率 + 词级 pairwise Jaccard(距离) + 字符长度 CV"""
    _word = re.compile(r"[A-Za-z]+|\d+")
    out = {}
    for name in cfgs:
        sub = [t for t in rows if t['config']==name]
        byq = defaultdict(list)
        for t in sub:
            byq[t['data_id']].append(t)
        uniq_ratios, jac_dists, all_chars = [], [], []
        for v in byq.values():
            skills = [(t['skill'] or '') for t in v]
            all_chars += [len(s) for s in skills]
            uniq_ratios.append(len(set(skills))/len(skills))
            sets = [set(_word.findall(s.lower())) for s in skills]
            ds = []
            for i in range(len(sets)):
                for j in range(i+1, len(sets)):
                    a,b = sets[i],sets[j]
                    if not a and not b:
                        ds.append(0.0); continue
                    inter=len(a&b); uni=len(a|b) or 1
                    ds.append(1 - inter/uni)   # 1=完全不同,0=完全相同
            if ds:
                jac_dists.append(sum(ds)/len(ds))
        cv = (st.pstdev(all_chars)/ (sum(all_chars)/len(all_chars))) if all_chars and sum(all_chars) else 0.0
        out[name] = dict(uniq=sum(uniq_ratios)/len(uniq_ratios),
                         jac=sum(jac_dists)/len(jac_dists) if jac_dists else 0.0,
                         char_cv=cv)
    return out


def fmt_table(title, gm, cfgs, cols):
    print(f"\n### {title}")
    head = "%-14s " % "config" + " ".join("%-9s" % c for c,_ in cols)
    print(head); print("-"*len(head))
    for name in cfgs:
        if name not in gm:
            continue
        m = gm[name]
        row = "%-14s " % name + " ".join(("%-9.3f" if isinstance(m[k],float) else "%-9d") % m[k] for _,k in cols)
        print(row)


def main():
    os.makedirs(os.path.join(HERE,'analysis_out'), exist_ok=True)
    data = {}
    for g,(sf,rf) in GROUPS.items():
        data[g] = dict(skill=load(sf), refl=load(rf))
        print(f"[load] {g}: skillcfg={len(data[g]['skill'])}  reflexion={len(data[g]['refl'])}")

    # ---------- 对齐校验: 三组是否同题同 idx ----------
    print("\n" + "="*80 + "\n[对齐校验] 三组 (config,data_id,sample_idx) 键集合是否一致")
    def keyset(rows):
        return set((t['config'],t['data_id'],t['sample_idx']) for t in rows)
    ka,kb,kc = keyset(data['A_old_nothink']['skill']),keyset(data['B_new_think']['skill']),keyset(data['C_new_nothink']['skill'])
    print(f"  A∩C 交集/并集(nothink 对比): {len(ka&kc)}/{len(ka|kc)}  A独有={len(ka-kc)} C独有={len(kc-ka)}")
    print(f"  C∩B 交集/并集(think 对比):   {len(kc&kb)}/{len(kc|kb)}  C独有={len(kc-kb)} B独有={len(kb-kc)}")

    cols_skill = [('n','n'),('parse','parse'),('leak','leak'),('trunc','trunc'),
                  ('sgTrunc','sg_trunc'),('chars','med_chars'),('base','base'),
                  ('acc@1','acc'),('lift','lift'),('pass@k','p_at_k'),('rescue@k','rescue'),
                  ('accClean','acc_clean')]
    # ---------- Q1+Q4: 每组每 config 指标 ----------
    print("\n" + "="*80 + "\n[Q1/Q4] skillcfg 每类准确率与质量指标")
    GM = {}
    for g in GROUPS:
        GM[g] = agg_metrics(data[g]['skill'], SKILL_CFGS)
        fmt_table(f"{g}  (skillcfg)", GM[g], SKILL_CFGS, cols_skill)

    print("\n" + "="*80 + "\n[Q1/Q4] reflexion 每类救活指标")
    RM = {}
    for g in GROUPS:
        RM[g] = agg_metrics(data[g]['refl'], REFL_CFGS)
        fmt_table(f"{g}  (reflexion)", RM[g], REFL_CFGS, cols_skill)

    # ---------- Q2: 横向差分 ----------
    print("\n" + "="*80 + "\n[Q2] 环境(vLLM/栈)影响 = C - A (同 nothink)   ；think 影响 = B - C (同新环境)")
    print("%-14s %-22s %-22s" % ("config","env(C-A) acc/lift/parse","think(B-C) acc/lift/parse"))
    for name in SKILL_CFGS:
        a,c,b = GM['A_old_nothink'].get(name),GM['C_new_nothink'].get(name),GM['B_new_think'].get(name)
        if not(a and c and b): continue
        env = f"{c['acc']-a['acc']:+.3f}/{c['lift']-a['lift']:+.3f}/{c['parse']-a['parse']:+.3f}"
        thk = f"{b['acc']-c['acc']:+.3f}/{b['lift']-c['lift']:+.3f}/{b['parse']-c['parse']:+.3f}"
        print("%-14s %-22s %-22s" % (name, env, thk))

    # ---------- Q3: 多样性 ----------
    print("\n" + "="*80 + "\n[Q3] skill 生成多样性 (题内 8 rollout；uniq=去重率 jac=词级平均两两距离 char_cv=长度变异)")
    for g in GROUPS:
        dv = diversity(data[g]['skill'], SKILL_CFGS)
        print(f"\n### {g}")
        print("%-14s %-8s %-8s %-8s" % ("config","uniq","jac","char_cv"))
        for name in SKILL_CFGS:
            m=dv[name]; print("%-14s %-8.3f %-8.3f %-8.3f" % (name,m['uniq'],m['jac'],m['char_cv']))

    print("\n[done] 详见各节表格；抽样交叉验证见 sample_probe.py 输出")

if __name__ == '__main__':
    main()
