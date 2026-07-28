#!/usr/bin/env python3
# sample_probe.py — 抽样交叉验证，为统计结论提供可人工核对的具体证据（含 文件:行号:data_id）。
# 5 项交叉验证:
#   V1 A vs C 样本级逐字一致率 (环境影响的最硬证据; 同 key 对比 skill 与 withskill_correct)
#   V2 leak 标记真伪 (随机抽 leaked=True/False 各若干, 核对 reference_answer 是否真出现在 skill)
#   V3 think 泄漏机制 (B 组 leaked=True 样本, 展示 think 段算出答案->写进 skill)
#   V4 parse 失败成因 (B 组 P7/parse=False 样本, 确认 skillgen 被 think 吃满预算而截断)
#   V5 accClean 样本量 (打印各 config n_clean, 防止小样本误读高 accClean)
import json, os, re, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    'A': 'skillcfg_full_off.jsonl',
    'B': 'skillcfg_full_on.jsonl',
    'C': 'env_runs/vllm_0.23.0/skillcfg_full_off.jsonl',
}
random.seed(0)


def load_indexed(path):
    """返回 {(config,data_id,sample_idx): (lineno, record)}"""
    d = {}
    with open(os.path.join(HERE, path)) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[(r['config'], r['data_id'], r['sample_idx'])] = (i, r)
    return d


def leaked_check(skill, ref):
    """独立复现 answer_leaked 的判定意图: 整数答案是否以数字边界出现在 skill"""
    s = str(ref).strip()
    if not re.fullmatch(r'-?\d+(\.\d+)?', s):
        return None  # 非纯数字答案, 泄漏判定本就不适用
    return re.search(r'(?<!\d)' + re.escape(s) + r'(?!\d)', skill or '') is not None


def main():
    A = load_indexed(FILES['A'])
    B = load_indexed(FILES['B'])
    C = load_indexed(FILES['C'])
    fa, fb, fc = FILES['A'], FILES['B'], FILES['C']

    print("="*90)
    print("[V1] A vs C 样本级逐字一致率（同 config/data_id/sample_idx；环境=torch/cuda/transformers 变更的净效应）")
    keys = [k for k in A if k in C]
    same_skill = sum(1 for k in keys if (A[k][1]['skill'] or '') == (C[k][1]['skill'] or ''))
    same_exec  = sum(1 for k in keys if (A[k][1]['withskill_text'] or '') == (C[k][1]['withskill_text'] or ''))
    same_corr  = sum(1 for k in keys if bool(A[k][1]['withskill_correct']) == bool(C[k][1]['withskill_correct']))
    print(f"  N={len(keys)}  skill 逐字一致={same_skill/len(keys):.3f}  "
          f"executor 全文逐字一致={same_exec/len(keys):.3f}  correct 一致={same_corr/len(keys):.3f}")
    # 展示一个 skill 不同但 correct 相同 / 一个逐字相同 的实例
    diff_ex = next((k for k in keys if (A[k][1]['skill'] or '')!=(C[k][1]['skill'] or '')), None)
    if diff_ex:
        la,_ = A[diff_ex]; lc,_ = C[diff_ex]
        print(f"  例(skill 不同): key={diff_ex}  A={fa}:{la}  C={fc}:{lc}")
        print(f"     A.skill[:90]={ (A[diff_ex][1]['skill'] or '')[:90]!r}")
        print(f"     C.skill[:90]={ (C[diff_ex][1]['skill'] or '')[:90]!r}")

    print("\n"+"="*90)
    print("[V2] leak 标记真伪核对（随机抽样, 独立重算数字边界匹配）")
    for grp,(D,fn) in {'A':(A,fa),'B':(B,fb),'C':(C,fc)}.items():
        pos = [k for k in D if D[k][1]['leaked']]
        neg = [k for k in D if not D[k][1]['leaked']]
        random.shuffle(pos); random.shuffle(neg)
        tp = 0; checked_pos = pos[:60]
        for k in checked_pos:
            _,r = D[k]
            v = leaked_check(r['skill'], r['reference_answer'])
            if v: tp += 1
        fp_free = 0; checked_neg = neg[:60]
        for k in checked_neg:
            _,r = D[k]
            v = leaked_check(r['skill'], r['reference_answer'])
            if v is False or v is None:
                fp_free += 1
        print(f"  [{grp}] leaked=True 抽{len(checked_pos)} 复算确含答案={tp}/{len(checked_pos)}  "
              f"leaked=False 抽{len(checked_neg)} 复算确不含={fp_free}/{len(checked_neg)}")

    print("\n"+"="*90)
    print("[V3] think 泄漏机制实例（B 组 leaked=True, 展示 think 段->skill 搬答案）")
    shown = 0
    for k in B:
        _,r = B[k]
        if r['config']=='P1_narrative' and r['leaked'] and str(r['reference_answer']).lstrip('-').isdigit():
            ln,_ = B[k]
            full = r['skillgen_full'] or ''
            ans = str(r['reference_answer'])
            think_end = full.lower().find('</think>')
            in_think = ans in full[:think_end] if think_end>0 else False
            in_skill = ans in (r['skill'] or '')
            print(f"  {fb}:{ln}  key={k}  ref={ans}  答案在think段={in_think} 在skill={in_skill}")
            idx = (r['skill'] or '').find(ans)
            if idx>=0:
                print(f"     skill 命中片段: ...{(r['skill'])[max(0,idx-45):idx+len(ans)+25]!r}...")
            shown += 1
            if shown>=3: break

    print("\n"+"="*90)
    print("[V4] parse 失败成因（B 组 P7_minimal, parseable=False）")
    cnt_len=0; cnt_noskill=0; shown=0
    for k in B:
        _,r = B[k]
        if r['config']!='P7_minimal' or r['parseable']:
            continue
        full = r['skillgen_full'] or ''
        has_close = '</skills>' in full.lower()
        if r['skillgen_stop']=='length': cnt_len+=1
        if not has_close: cnt_noskill+=1
        if shown<3:
            ln,_ = B[k]
            print(f"  {fb}:{ln} key={k} stop={r['skillgen_stop']} sg_tokens={r['skillgen_tokens']} "
                  f"含</skills>={has_close} full尾50={full[-50:]!r}")
            shown+=1
    total_pf = sum(1 for k in B if B[k][1]['config']=='P7_minimal' and not B[k][1]['parseable'])
    print(f"  P7 parse失败共 {total_pf}: 其中 stop=length {cnt_len}  无</skills> {cnt_noskill}")

    print("\n"+"="*90)
    print("[V5] accClean 样本量核对（leaked=0 且 parseable=1 的 n_clean, 防小样本误读）")
    for grp,(D,fn) in {'A':(A,fa),'B':(B,fb),'C':(C,fc)}.items():
        by=defaultdict(lambda:[0,0])
        for k in D:
            _,r=D[k]
            if (not r['leaked']) and r['parseable']:
                by[r['config']][0]+=1
                by[r['config']][1]+= 1 if r['withskill_correct'] else 0
        cells=" ".join(f"{c.split('_')[0]}={by[c][0]}({(by[c][1]/by[c][0] if by[c][0] else 0):.2f})"
                       for c in ['P1_narrative','P5_pitfall','P7_minimal'])
        print(f"  [{grp}] n_clean(acc):  {cells}")

if __name__ == '__main__':
    main()
