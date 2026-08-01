# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepMath-103K loader with difficulty-stratified train/eval split.

Dataset: AI-ModelScope/DeepMath-103K (columns: question / final_answer / difficulty / topic /
r1_solution_1..3). We keep only rows whose final_answer normalizes to a number via v2
``_numeric_value`` (the \\boxed{} judging pipeline is numeric-exact; answers like ``\\phi^4``
cannot be scored and are dropped).

Stratified split (skill_quality_analysis.md 组成漂移修正): difficulty is bucketed to its
rounded integer level; ``eval_size`` problems are sampled with per-bucket quotas proportional
to the pool (largest-remainder rounding), the rest form the train pool — so train and eval
difficulty proportions match by construction. All sampling is seeded and file-order stable:
``data_id = dm:<level>:<global_row_index>`` is reproducible across runs/experiments.

Train-only difficulty floor (``--min-level``): E1/E5 gradient audit showed level<=5 groups are
dominated by all-pass (level 3: 63-74% all-pass, corr(level, mixed_rate)=0.92), i.e. mostly
zero-gradient. The floor drops those rows from the *train pool only*; the eval split keeps the
full-level stratification so eval/baseline stay comparable across experiments.
"""
import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

import train_skill_v2 as v2


def _read_rows(deepmath_dir: str) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    paths = sorted(glob.glob(os.path.join(deepmath_dir, '**', '*.parquet'), recursive=True))
    if not paths:
        raise FileNotFoundError(f'no parquet files under --deepmath-dir {deepmath_dir}')
    rows: List[Dict[str, Any]] = []
    for p in paths:
        # r1_solution_1: E14-ref 的 logP 目标文本（外部 R1 参考解，强于 executor 自采解）
        t = pq.read_table(p, columns=['question', 'final_answer', 'difficulty', 'r1_solution_1'])
        rows.extend(t.to_pylist())
    return rows


def load_deepmath_records(args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """-> (train_records, eval_records), each record {'data_id','problem','reference_answer'}."""
    rows = _read_rows(args.deepmath_dir)
    pool: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):  # global row index over sorted files = stable id
        problem = (r.get('question') or '').strip()
        num = v2._numeric_value(r.get('final_answer'))
        if not problem or num is None:
            continue
        lvl = int(round(float(r.get('difficulty') or 0)))
        pool.append({'data_id': f'dm:{lvl}:{i}', 'problem': problem,
                     'reference_answer': num, '_level': lvl,
                     'solution': (r.get('r1_solution_1') or '').strip()})

    # bucket by level, seeded shuffle inside each bucket
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in pool:
        buckets[rec['_level']].append(rec)
    rng = np.random.RandomState(args.seed)
    for lvl in sorted(buckets):
        rng.shuffle(buckets[lvl])

    # eval quota per bucket: proportional, largest-remainder rounding.
    # --eval-min-level>0 把配额限在难度不低于它的桶里（默认 0 = 全难度混合，与旧臂逐字一致）。
    # E17 需要它：该臂的指标只在【裸解做错的题】上有信息量，而全难度混合的错题率只有
    # ~26%（E16 实测 128 -> 33 道，SE 0.087，趋势读不出来）；level>=6 上是 ~43%，同样的
    # GPU 成本能换到 1.6 倍的有效样本，同时与训练池（min_level）同分布。
    eval_min_level = int(getattr(args, 'eval_min_level', 0) or 0)
    elig = {lvl: b for lvl, b in buckets.items() if lvl >= eval_min_level}
    n_elig = sum(len(b) for b in elig.values())
    eval_n = min(args.eval_size, n_elig) if (args.eval_size > 0 and n_elig) else 0
    quota = {lvl: eval_n * len(b) / n_elig for lvl, b in elig.items()} if n_elig else {}
    take = {lvl: int(q) for lvl, q in quota.items()}
    for lvl in sorted(quota, key=lambda x: quota[x] - int(quota[x]), reverse=True):
        if sum(take.values()) >= eval_n:
            break
        take[lvl] += 1

    eval_records, train_records = [], []
    for lvl in sorted(buckets):
        b = buckets[lvl]
        n_ev = take.get(lvl, 0)
        eval_records.extend(b[:n_ev])
        train_records.extend(b[n_ev:])
    min_level = int(getattr(args, 'min_level', 0) or 0)
    if min_level > 0:  # train-only floor; eval keeps full-level mix (see module docstring)
        n_before = len(train_records)
        train_records = [r for r in train_records if r['_level'] >= min_level]
        v2.logger.info(f'[data] min_level={min_level}: train pool {n_before} -> {len(train_records)}')
    rng.shuffle(train_records)  # ProblemPool reshuffles too; this decorrelates level runs
    if args.n > 0:  # optional stratified-in-expectation downsample (pool already shuffled)
        train_records = train_records[:args.n]

    def _lvls(rs):
        c = defaultdict(int)
        for r in rs:
            c[r['_level']] += 1
        return {k: round(v / len(rs), 3) for k, v in sorted(c.items())}
    v2.logger.info(f'[data] DeepMath: pool={len(pool)} (numeric-only of {len(rows)}) '
                   f'train={len(train_records)} eval={len(eval_records)}')
    v2.logger.info(f'[data] level mix train={_lvls(train_records)} eval={_lvls(eval_records)}')
    for r in eval_records + train_records:
        r.pop('_level', None)
    return train_records, eval_records
