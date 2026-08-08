# -*- coding: utf-8 -*-
"""多机分片采集的两个配套工具：导出「已跑过的题」种子 + 合并多机产物。

  python3 shard_tool.py seed  <src_dir> <out.jsonl>        # 在 A 机跑，产出给 B 机的种子
  python3 shard_tool.py merge <out_dir> <dir1> [dir2 ...]  # 合并任意台机器的产物

为什么需要 seed：
  crc32 分片只保证「以后」两台机器不撞，但 A 机已经跑掉的题会均匀落在所有分片上
  （实测 7374 题在 SHARD_N=2 下是 3667/3707）。B 机目录是空的，resume_done_ids()
  读不到任何东西，就会把属于自己分片的那 3707 题重跑一遍 —— 白烧一半算力，
  而且合并时同一 data_id 出现两份。种子文件就是把 A 机的 done id 搬给 B 机。
  ❗ 种子只需要 e18_candidates.jsonl 的 data_id 字段（resume_done_ids 只读这个），
  所以导出的是**精简行**，不是整个 27000 行的候选文件 —— B 机不需要 A 机的 skill 正文。
"""
import collections
import json
import os
import sys

FILES = ('e18_sft_dataset.jsonl', 'e18_candidates.jsonl', 'collect_log.jsonl')


def _rows(path):
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


def cmd_seed(src, out):
    """把 src 里所有跑过的 data_id 导成最小的 candidates 行。"""
    ids = {r['data_id'] for r in _rows(os.path.join(src, 'e18_candidates.jsonl'))
           if r.get('data_id')}
    with open(out, 'w', encoding='utf-8') as f:
        for i in sorted(ids):
            # resume_done_ids() 只取 data_id；其余字段给最小合法值，
            # 保证这些行即使被别的分析脚本读到也不会伪装成真实候选：
            # kept=False + parseable=False + seed=True 一眼可滤。
            f.write(json.dumps({'data_id': i, 'kept': False, 'parseable': False,
                                'seed': True, 'run': 'SEED', 'chunk': -1},
                               ensure_ascii=False) + '\n')
    print('导出 %d 个已跑 data_id -> %s' % (len(ids), out))
    print('用法：拷到 B 机的 OUTPUT_DIR/e18_candidates.jsonl，再用 KOD_RESUME=1 启动')


def cmd_merge(out_dir, dirs):
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    # ---- 1. sft_dataset：按 data_id 去重（同题多机重复时保留 pass_gain 更高者）----
    best, dup = {}, 0
    for d in dirs:
        for r in _rows(os.path.join(d, 'e18_sft_dataset.jsonl')):
            k = r.get('data_id')
            if not k:
                continue
            if k in best:
                dup += 1
                # 保留 gain 高的：重复只可能来自"种子没同步"的意外，
                # 此时保留更优样本比保留先到者更合理
                if (r.get('pass_gain') or -9) <= (best[k].get('pass_gain') or -9):
                    continue
            best[k] = r
    with open(os.path.join(out_dir, 'e18_sft_dataset.jsonl'), 'w', encoding='utf-8') as f:
        for r in best.values():
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report['sft'] = (len(best), dup)

    # ---- 2. candidates：按 (data_id, run, cand_idx) 去重，丢掉 seed 占位行 ----
    seen, rows, nseed = set(), [], 0
    for d in dirs:
        for r in _rows(os.path.join(d, 'e18_candidates.jsonl')):
            if r.get('seed'):
                nseed += 1
                continue
            k = (r.get('data_id'), r.get('run'), r.get('cand_idx'))
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    with open(os.path.join(out_dir, 'e18_candidates.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report['cand'] = (len(rows), nseed)

    # ---- 3. collect_log：chunk 编号两机都从 0 起，直接 cat 会产生歧义 ----
    # 不重编号（会破坏与 run.log 的对照），改为加 src 字段标机器来源，
    # 并按 (run, chunk) 唯一化。分析脚本本来就该按 run 分组看 chunk。
    lg, seen2 = [], set()
    for d in dirs:
        tag = os.path.basename(os.path.normpath(d))
        for r in _rows(os.path.join(d, 'collect_log.jsonl')):
            k = (r.get('run'), r.get('chunk'))
            if k in seen2:
                continue
            seen2.add(k)
            r['src'] = tag
            lg.append(r)
    lg.sort(key=lambda r: (str(r.get('run')), r.get('chunk') or 0))
    with open(os.path.join(out_dir, 'collect_log.jsonl'), 'w', encoding='utf-8') as f:
        for r in lg:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report['log'] = (len(lg), 0)

    print('=== 合并完成 -> %s ===' % out_dir)
    print('  e18_sft_dataset.jsonl  %6d 条   (跨机重复丢弃 %d)' % report['sft'])
    print('  e18_candidates.jsonl   %6d 条   (滤掉种子占位 %d)' % report['cand'])
    print('  collect_log.jsonl      %6d 条' % report['log'][0])
    # 交叉校验：胜者的 data_id 必须都能在候选里找到
    cid = {r.get('data_id') for r in rows}
    miss = [k for k in best if k not in cid]
    print('  一致性：胜者 data_id 在候选中缺失 %d 个 %s'
          % (len(miss), '(OK)' if not miss else '<- 异常'))
    per = collections.Counter(str(r.get('run')).split('.')[-1] for r in rows)
    print('  按分片计候选数：%s' % dict(per))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == 'seed':
        cmd_seed(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'merge':
        cmd_merge(sys.argv[2], sys.argv[3:])
    else:
        print(__doc__)
        sys.exit(2)
