# Copyright (c) ModelScope Contributors. All rights reserved.
"""BigCodeBench loader for the ablation package (code task family).

Dataset: bigcodebench/bcb.parquet (v0.1.4, 1140 tasks). Record shape matches the math loader
so nothing downstream changes shape:
    data_id          = task_id            (e.g. BigCodeBench/42)
    problem          = instruct_prompt    (ends with the exact imports + signature to reproduce)
    reference_answer = code_task.payload_of(task)   # 判分载荷，不是数值答案

与 DeepMath loader 的三处结构差异：
1. **没有 difficulty**，所以没有分层抽样，也不存在 --min-level / --eval-min-level（在 code
   模式下被显式忽略并告警）。train/eval 就是同一个 seed 洗牌后的前后切分。
2. **题池极小**：1140 题，剔掉缺库/需外网GUI子进程的题后约 900，再减 eval 后训练池只有几百题。
   E17 用 chunk 48 × 50 updates = 2400 次抽取 ≈ 3 个 epoch 重复题（用户 2026-07-31 拍板接受）。
   重复题的 rubric 全部缓存命中，所以重复的成本只在 GPU rollout。
3. **有"沙箱自检"这道闸**：参考解答跑不过它自己的单测 = 环境不可判定（缺库的边角、随机种子、
   matplotlib 后端等），这类题的 0 分与模型能力无关，必须剔掉，否则它会给每个臂加一层同样的
   噪声底并稀释 lift。probe 实测 7.5% 属于这一类。自检结果按 parquet 落一个 json 缓存，
   之后各臂零成本复用。
"""
import json
import os
from typing import Any, Dict, List, Tuple

import code_task
import train_skill_v2 as v2


def _broken_task_ids(args, tasks: List[Dict[str, Any]]) -> set:
    """参考解答跑不过自己单测的题（缓存到 --output-dir 的父目录，跨臂复用）。"""
    base = (getattr(args, 'rubric_global_dir', None)
            or os.path.dirname(os.path.abspath(str(args.output_dir).rstrip('/'))))
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, 'bcb_broken_tasks.json')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                cached = json.load(f)
            if int(cached.get('n_tasks', -1)) == len(tasks):
                return set(cached.get('broken') or [])
            v2.logger.info(f'[data] {path} 的题数 {cached.get("n_tasks")} != 当前 {len(tasks)}，'
                           f'重跑自检')
        except Exception as exc:
            v2.logger.warning(f'[data] 读取 {path} 失败（{exc}），重跑自检')
    v2.logger.info(f'[data] 沙箱自检：{len(tasks)} 道题跑参考解答（一次性，之后走缓存）…')
    broken = code_task.selftest(tasks, args.test_workers, args.test_timeout)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'n_tasks': len(tasks), 'broken': sorted(broken)}, f, indent=1)
    return set(broken)


def load_code_records(args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """-> (train_records, eval_records)，每条 {'data_id','problem','reference_answer'}。"""
    tasks, stats = code_task.load_tasks(args.bcb_parquet, args.seed)
    if not tasks:
        raise FileNotFoundError(f'no usable BigCodeBench task in {args.bcb_parquet}')
    v2.logger.info(f"[data] BigCodeBench: 全集 {stats['raw']}，剔除依赖缺失 "
                   f"{stats['drop_missing_lib']}、需外网/GUI/子进程 {stats['drop_needs_net_or_gui']}"
                   f" -> {stats['kept']}")
    if getattr(args, 'code_selftest', True):
        broken = _broken_task_ids(args, tasks)
        if broken:
            tasks = [t for t in tasks if t['task_id'] not in broken]
            v2.logger.info(f'[data] 剔除参考解答自己跑不过单测的题 {len(broken)} 道 '
                           f'（沙箱不可判定，非模型能力）-> 可用 {len(tasks)}')
    for k in ('min_level', 'eval_min_level'):
        if int(getattr(args, k, 0) or 0):
            v2.logger.warning(f'[data] --{k.replace("_", "-")} 在 code 任务下无效（'
                              f'BigCodeBench 没有 difficulty 字段），已忽略')
    recs = [{'data_id': t['task_id'], 'problem': t['instruct_prompt'],
             'reference_answer': code_task.payload_of(t)} for t in tasks]
    eval_n = min(args.eval_size, len(recs)) if args.eval_size > 0 else 0
    eval_records, train_records = recs[:eval_n], recs[eval_n:]
    if args.n > 0:
        train_records = train_records[:args.n]
    if not train_records:
        raise ValueError(f'--eval-size {args.eval_size} 吃掉了整个题池（可用 {len(recs)}）')
    epochs = (args.chunk_size * args.max_updates) / max(1, len(train_records))
    v2.logger.info(f'[data] train={len(train_records)} eval={len(eval_records)}；'
                   f'按 chunk={args.chunk_size} x max_updates={args.max_updates} 估算约 '
                   f'{epochs:.1f} 个 epoch（题会重复，rubric 走缓存）')
    return train_records, eval_records
