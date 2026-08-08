# -*- coding: utf-8 -*-
"""E21：rubric 值多少钱？在**同一批题**上做成功复盘 vs 失败诊断的配对对照。

目标题型 = 「首次成功但不稳定」（0 < base_pass_rate < 1）。
⭐ 为什么只能用这类题：它们**同时拥有**成功轨迹和失败轨迹，所以同一道题可以同时喂给
两个 arm，构成**配对设计**（paired design）。base=1.0 的题没有失败轨迹（无法出 rubric），
base=0 的题没有成功轨迹（无法做成功复盘）—— 只有这个交集能做干净对照。
配对比独立分组强得多：题目难度是最大的方差来源，配对把它消掉了。

两个 arm，除了输入信号完全同构（同题、同 N_SKILLS、同 executor、同温度、同 rollout 数）：

  arm SUCCESS : 成功代码           -> narrative skill（无 rubric）   [E20 的 prompt]
  arm RUBRIC  : 失败代码 + 报错 -> 教师诊断出 rubric -> narrative skill  [E18 的 prompt]

判据统一为 J2（升到全对）：base<1 的题加 skill 后 with_pass_rate 是否达到 1.0。
这是唯一在两个 arm 上都可测、且不受选择偏差污染的口径。

⚠️ 已知的不对称（诚实记录，不是 bug）：
  1. RUBRIC arm 多消耗一次教师 API 调用（不占 GPU，但不是零成本）。
  2. 两个 arm 的 prompt 家族不同（SKILLGEN_SYSTEM vs SKILLGEN_SYSTEM_SUCCESS），
     所以测的是「成功复盘管线」vs「失败诊断管线」的**整体**差异，
     不是「rubric 这一个字段」的净效应。要拆到字段级需要第三个 arm
     （失败代码但不给 rubric），本脚本用 FAILONLY arm 补上。
  3. FAILONLY arm 复用 SKILLGEN_SYSTEM 但把 rubric 位填成「无诊断」的官方 fallback
     （skillgen_prompt 内建该分支），所以 arm 间 system prompt 一致，
     RUBRIC vs FAILONLY 的差值才是 rubric 字段的净贡献。
"""
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, '..', 'human'))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e18_kodcode import clean_text, judge_seqs, load_records  # noqa: E402
from e18_prompts import direct_prompt, skill_solve_prompt, skillgen_prompt  # noqa: E402
from e20_success_skill import (extract_skill, run_samples, seq_text,  # noqa: E402
                               skillgen_success_prompt, _mean, _pass_rate)
from e23_rubric import RubricCache, build_checker  # noqa: E402

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3-4B')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(_HERE, 'output.e21.paired'))
SEED = int(os.environ.get('SEED', 42))
SKILL_GPUS = int(os.environ.get('SKILL_GPUS', 4))
EXEC_GPUS = int(os.environ.get('EXEC_GPUS', 4))
NUM_GPUS = SKILL_GPUS + EXEC_GPUS
GPU_MEM = float(os.environ.get('GPU_MEM', 0.85))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 24000))

POOL = int(os.environ.get('POOL', 400))            # 题池；不稳定题约占 4-5%
N_SKILLS = int(os.environ.get('N_SKILLS', 4))
BARE_ROLLOUTS = int(os.environ.get('BARE_ROLLOUTS', 4))
EXEC_ROLLOUTS = int(os.environ.get('EXEC_ROLLOUTS', 8))
SKILL_MAX_TOKENS = int(os.environ.get('SKILL_MAX_TOKENS', 8192))
EXEC_MAX_TOKENS = int(os.environ.get('EXEC_MAX_TOKENS', 15000))
EXEC_TEMPERATURE = float(os.environ.get('EXEC_TEMPERATURE', 0.6))
EXEC_TOP_P = float(os.environ.get('EXEC_TOP_P', 0.95))
SKILL_TEMPERATURE = float(os.environ.get('SKILL_TEMPERATURE', 1.0))


def build_runtime():
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, lazy_collect=False, groups=[
        DeviceGroup(name='skill', ranks=list(range(SKILL_GPUS)), device_type='GPU'),
        DeviceGroup(name='exec', ranks=list(range(SKILL_GPUS, NUM_GPUS)),
                    device_type='GPU')])

    def mk(group, world, thinking):
        s = vLLMSampler(model_id=MODEL_ID, remote_group=group,
                        device_mesh=DeviceMesh.from_sizes(world_size=world, dp_size=world),
                        engine_args={'gpu_memory_utilization': GPU_MEM,
                                     'max_model_len': MAX_MODEL_LEN,
                                     'tensor_parallel_size': 1})
        s.set_template(Template, model_id=MODEL_ID, enable_thinking=thinking,
                       max_length=MAX_MODEL_LEN)
        return s

    return mk('skill', SKILL_GPUS, True), mk('exec', EXEC_GPUS, False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    ds, _ = load_records(SEED, 0, OUTPUT_DIR)
    pool = [r for i, r in enumerate(ds.dataset) if i < POOL]
    logger.info(f'E21 start: 题池 {len(pool)}  n_skills={N_SKILLS} '
                f'bare={BARE_ROLLOUTS} exec={EXEC_ROLLOUTS}')
    skill_sampler, exec_sampler = build_runtime()

    # ---- 1. 裸解，挑「首次成功但不稳定」的题 ----
    bare = run_samples(exec_sampler, [direct_prompt(r['problem']) for r in pool],
                       BARE_ROLLOUTS, EXEC_MAX_TOKENS, EXEC_GPUS,
                       temperature=EXEC_TEMPERATURE, top_p=EXEC_TOP_P)
    pairs, spans = [], []
    for r, seqs in zip(pool, bare):
        seqs = list(seqs or [])
        spans.append(len(seqs))
        pairs.extend((s, r['reference_answer']) for s in seqs)
    judged = judge_seqs(pairs) if pairs else []
    picked, i, n_first_ok = [], 0, 0
    for r, n in zip(pool, spans):
        rolls = judged[i:i + n]
        i += n
        if not rolls or not rolls[0]['correct']:
            continue
        n_first_ok += 1
        rate = _pass_rate(rolls)
        if rate >= 1.0:
            continue                       # 稳定成功 -> 无失败轨迹，做不了配对
        bad = next((x for x in rolls[1:] if not x['correct']), None)
        if bad is None:
            continue
        picked.append({'rec': r, 'base_rate': rate,
                       'good_code': rolls[0].get('code') or '',
                       'bad_roll': bad,
                       'base_tokens': _mean([x.get('gen_tokens') for x in rolls])})
    logger.info(f'[bare] 首次成功 {n_first_ok}/{len(pool)}；'
                f'其中**不稳定**（可配对）{len(picked)} 题')
    if not picked:
        raise RuntimeError('没有可配对的不稳定题')

    # ---- 2. 教师诊断出 rubric（纯 API，不占 GPU）----
    # ⭐ 缓存必须用**本 run 私有**的路径，不能复用全局 RUBRIC_CACHE_PATH：
    #    RubricCache 的键里**不含轨迹**，其跨 run 复用的前提是「executor 冻结在 T=0，
    #    同一题裸解逐字相同」（见 e23_rubric.py:227 的说明）。本实验裸解用 T=0.6，
    #    轨迹每次不同，共用全局缓存会拿到**别的轨迹**的诊断，静默污染 RUBRIC arm。
    cache = RubricCache(os.path.join(OUTPUT_DIR, 'diag_cache.jsonl'))
    checker = build_checker()
    rubrics = cache.diagnose_many(checker, [(p['rec'], p['bad_roll']) for p in picked])
    n_rub = sum(1 for x in rubrics for _ in [x] if x)
    logger.info(f'[rubric] {n_rub}/{len(picked)} 题拿到诊断')
    for p, rb in zip(picked, rubrics):
        p['rubric'] = rb or ''

    # ---- 3. 三个 arm 生成 skill ----
    # ⭐ 三个 arm 一次性拼进同一个 sample 调用，保证同一批权重、同一批 KV cache 状态，
    #    避免"先跑完 A 再跑 B"引入的引擎状态差异。
    arms: List[str] = ['SUCCESS', 'RUBRIC', 'FAILONLY']
    prompts, meta = [], []
    for p in picked:
        prompts.append(skillgen_success_prompt(p['rec']['problem'], p['good_code']))
        meta.append((p, 'SUCCESS'))
        prompts.append(skillgen_prompt(p['rec']['problem'], p['rubric'], False))
        meta.append((p, 'RUBRIC'))
        # FAILONLY：同 system prompt，rubric 位走内建的「无诊断」fallback
        prompts.append(skillgen_prompt(p['rec']['problem'], '', False))
        meta.append((p, 'FAILONLY'))
    sg = run_samples(skill_sampler, prompts, N_SKILLS, SKILL_MAX_TOKENS,
                     SKILL_GPUS, temperature=SKILL_TEMPERATURE)
    flat = []
    for (p, arm), seqs in zip(meta, sg):
        for ci in range(N_SKILLS):
            seq = seqs[ci] if seqs and ci < len(seqs) else None
            flat.append({'p': p, 'arm': arm, 'cand_idx': ci,
                         'skill': extract_skill(seq_text(seq))})
    for arm in arms:
        g = [f for f in flat if f['arm'] == arm]
        n = sum(1 for f in g if f['skill'])
        logger.info(f'[skillgen] {arm}: {n}/{len(g)} 可解析 ({100*n/max(1,len(g)):.0f}%)')

    # ---- 4. 带 skill 重解 ----
    todo = [f for f in flat if f['skill']]
    ws = run_samples(exec_sampler,
                     [skill_solve_prompt(f['p']['rec']['problem'], f['skill']) for f in todo],
                     EXEC_ROLLOUTS, EXEC_MAX_TOKENS, EXEC_GPUS,
                     temperature=EXEC_TEMPERATURE, top_p=EXEC_TOP_P)
    pairs, spans = [], []
    for f, seqs in zip(todo, ws):
        seqs = list(seqs or [])
        spans.append(len(seqs))
        pairs.extend((s, f['p']['rec']['reference_answer']) for s in seqs)
    judged = judge_seqs(pairs) if pairs else []
    i = 0
    for f, n in zip(todo, spans):
        rolls = judged[i:i + n]
        i += n
        f['with_rate'] = _pass_rate(rolls)
        f['with_tokens'] = _mean([x.get('gen_tokens') for x in rolls])

    # ---- 5. 落盘 ----
    out = os.path.join(OUTPUT_DIR, 'e21_candidates.jsonl')
    with open(out, 'w', encoding='utf-8') as fh:
        for f in flat:
            p = f['p']
            row = {'data_id': p['rec']['data_id'], 'arm': f['arm'],
                   'cand_idx': f['cand_idx'], 'base_rate': p['base_rate'],
                   'parseable': bool(f['skill']),
                   'with_rate': f.get('with_rate'),
                   'base_tokens': round(p['base_tokens'], 1),
                   'with_tokens': round(f.get('with_tokens') or 0, 1),
                   'has_rubric': bool(p['rubric']),
                   'skill_chars': len(f['skill']), 'skill': f['skill']}
            if f.get('with_rate') is not None:
                row['delta'] = round(f['with_rate'] - p['base_rate'], 6)
                row['J2_robust'] = f['with_rate'] >= 1.0 - 1e-9
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    cache.close()
    logger.info(f'[done] 落盘 {out}，用时 {(time.time()-t0)/60:.1f} 分钟')


if __name__ == '__main__':
    main()
