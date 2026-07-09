"""Compare MATH direct vs RAG by difficulty level.

Re-grades both result files with the production ``answers_match`` (so the
stored ``is_correct`` is never trusted) and prints the per-level accuracy
plus the RAG gain (delta) so you can see how it varies with difficulty.

Defaults to the raw-RAG output (``math_rag_results.jsonl``); pass a second
arg to compare a different rag file (e.g. ``math_rag_hint_results.jsonl``).

Usage:
  python cookbook/exp/embedding/compare_math_levels.py \
      [direct.jsonl] [rag.jsonl]
"""
import importlib.util
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_grader():
    spec = importlib.util.spec_from_file_location(
        'egr', os.path.join(_HERE, 'eval_gpqa_rag.py'))
    egr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(egr)
    return egr.answers_match


def _load(path):
    return {json.loads(l)['idx']: json.loads(l)
            for l in open(path, encoding='utf-8') if l.strip()}


def main():
    direct_path = sys.argv[1] if len(sys.argv) > 1 else \
        './output/thinking_rag/math_direct_results.jsonl'
    hint_path = sys.argv[2] if len(sys.argv) > 2 else \
        './output/thinking_rag/math_rag_results.jsonl'

    answers_match = _load_grader()
    D = _load(direct_path)
    H = _load(hint_path)
    common = sorted(set(D) & set(H))
    print(f'direct={len(D)}  rag+hint={len(H)}  common={len(common)}')

    def runaway(rec):
        mo = rec.get('model_output') or ''
        return ('</think>' not in mo) or (
            not (rec.get('predicted') or '').strip() and len(mo) > 40000)

    def correct(rec):
        return answers_match(rec.get('predicted') or '',
                             rec['reference_answer'])

    # level -> counters
    per = defaultdict(lambda: {'n': 0, 'd': 0, 'h': 0,
                               'd_run': 0, 'h_run': 0})
    for i in common:
        lv = H[i].get('level') or D[i].get('level') or 'Unknown'
        c = per[lv]
        c['n'] += 1
        c['d'] += int(correct(D[i]))
        c['h'] += int(correct(H[i]))
        c['d_run'] += int(runaway(D[i]))
        c['h_run'] += int(runaway(H[i]))

    print(f'\n{"level":>10} | {"n":>4} | {"direct":>7} | {"rag+hint":>8} | '
          f'{"delta":>7} | {"d_run":>6} | {"h_run":>6}')
    print('-' * 68)
    tot = {'n': 0, 'd': 0, 'h': 0, 'd_run': 0, 'h_run': 0}
    for lv in sorted(per.keys()):
        c = per[lv]
        for k in tot:
            tot[k] += c[k]
        n = c['n']
        dacc, hacc = c['d'] / n, c['h'] / n
        print(f'{lv:>10} | {n:>4} | {dacc:>7.3f} | {hacc:>8.3f} | '
              f'{hacc - dacc:>+7.3f} | {c["d_run"]/n:>6.1%} | '
              f'{c["h_run"]/n:>6.1%}')
    print('-' * 68)
    n = tot['n']
    if n:
        print(f'{"OVERALL":>10} | {n:>4} | {tot["d"]/n:>7.3f} | '
              f'{tot["h"]/n:>8.3f} | {(tot["h"]-tot["d"])/n:>+7.3f} | '
              f'{tot["d_run"]/n:>6.1%} | {tot["h_run"]/n:>6.1%}')


if __name__ == '__main__':
    main()
