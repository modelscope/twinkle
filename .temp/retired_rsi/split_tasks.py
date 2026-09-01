# Copyright (c) ModelScope Contributors. All rights reserved.
"""Split a challenge.py task file into a training set and a held-out eval set.

The eval set is stratified by ``n_pass`` -- the number of solver attempts that
succeeded when the task was filtered. Difficulty is the whole point of the
filter, so a random split can easily hand the eval set every task the model
already solves 3 times in 4, and a pass rate on those says nothing about the
hard end. Stratifying keeps both halves the same shape.

    python cookbook/rsi/agentic/split_tasks.py \\
        output/rsi_agentic/run3/challenge_flows.jsonl --eval-frac 0.25
"""
import argparse
import json
import os
import random


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('flows', help='challenge_flows.jsonl')
    p.add_argument('--eval-frac', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train-out', default='', help='default: <flows dir>/train_tasks.jsonl')
    p.add_argument('--eval-out', default='', help='default: <flows dir>/eval_tasks.jsonl')
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.flows))
    train_out = args.train_out or os.path.join(out_dir, 'train_tasks.jsonl')
    eval_out = args.eval_out or os.path.join(out_dir, 'eval_tasks.jsonl')

    with open(args.flows, encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    if not tasks:
        raise SystemExit(f'{args.flows} contains no tasks')

    strata = {}
    for task in tasks:
        strata.setdefault(task.get('n_pass'), []).append(task)

    rng = random.Random(args.seed)
    train, held = [], []
    for n_pass in sorted(strata, key=lambda x: (x is None, x)):
        group = strata[n_pass][:]
        rng.shuffle(group)
        # round() rather than int(): with 3 tasks at a difficulty and a quarter
        # held out, truncating would give the eval set none of them.
        n_eval = min(len(group) - 1, round(len(group) * args.eval_frac)) if len(group) > 1 else 0
        held.extend(group[:n_eval])
        train.extend(group[n_eval:])

    for path, rows in ((train_out, train), (eval_out, held)):
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def dist(rows):
        out = {}
        for row in rows:
            out[row.get('n_pass')] = out.get(row.get('n_pass'), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (kv[0] is None, kv[0])))

    print(f'{len(tasks)} tasks, n_pass dist {dist(tasks)}')
    print(f'train {len(train)} -> {train_out}  dist {dist(train)}')
    print(f'eval  {len(held)} -> {eval_out}  dist {dist(held)}')


if __name__ == '__main__':
    main()
