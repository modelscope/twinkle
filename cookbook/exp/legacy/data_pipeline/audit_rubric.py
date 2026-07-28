"""Audit rubric scoring accuracy: re-score a spread of kept agent trajectories
with the SAME teacher RubricVerifier used in the pipeline, and print, per chosen
trajectory, the generated rubric + per-criterion pass rate + a readable segment
summary so a human can judge whether the score is *right* (good and bad alike).

Run (same env as the pipeline):
    LLM_BACKUP_MODEL=qwen3.7-max \
    LLM_BACKUP_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
    LLM_BACKUP_API_KEY=sk-... \
    python cookbook/exp/data_pipeline/audit_rubric.py
"""
import json
import os
from typing import Any, Dict, List, Optional

PROCESSED = os.environ.get(
    'PROCESSED_PATH', './output/data_pipeline/processed_20260531_200.jsonl')
N_HIGH = int(os.environ.get('AUDIT_N_HIGH', 3))
N_LOW = int(os.environ.get('AUDIT_N_LOW', 3))
N_MID = int(os.environ.get('AUDIT_N_MID', 2))
# Same stabilization policy as the pipeline (skeleton|fixed|off).
RUBRIC_MODE = os.environ.get('TRAJ_RUBRIC_MODE', 'skeleton').strip().lower()
# Repeat each re-score REPEAT times to measure score variance (jitter). >1 to
# check whether the stabilization actually lowered the spread.
REPEAT = int(os.environ.get('AUDIT_REPEAT', 1))


def _lab(row: Dict[str, Any], key: str) -> Optional[Any]:
    for kv in (row.get('user_data') or []):
        if isinstance(kv, list) and len(kv) == 2 and kv[0] == key:
            try:
                return json.loads(kv[1])
            except Exception:
                return kv[1]
    return None


def _seg_summary(messages: List[dict], max_chars: int = 900) -> str:
    parts = []
    for m in messages:
        role = m.get('role', '?')
        content = m.get('content') or ''
        tc = m.get('tool_calls')
        if tc:
            try:
                calls = json.loads(tc) if isinstance(tc, str) else tc
                names = ','.join(c.get('function', {}).get('name', '?') for c in calls)
                content = (content + f'  [tool_calls: {names}]').strip()
            except Exception:
                pass
        content = content.replace('\n', ' ')
        if len(content) > 200:
            content = content[:200] + '…'
        parts.append(f'  {role}: {content}')
    text = '\n'.join(parts)
    return text if len(text) <= max_chars else text[:max_chars] + '\n  …(truncated)'


def _infer_intent(messages: List[dict]) -> Optional[str]:
    """Structural intent for the whole trajectory (same detectors as the scorer)."""
    from twinkle_agentic.preprocessor.intent_classifier import (
        CodeDetector, MathDetector, ToolCallDetector)
    # tool_calls arrive JSON-encoded in the processed jsonl; decode so the
    # ToolCallDetector (which reads normalized tool_calls) can see them.
    norm = []
    for m in messages:
        m = dict(m)
        tc = m.get('tool_calls')
        if isinstance(tc, str) and tc.strip():
            try:
                m['tool_calls'] = json.loads(tc)
            except Exception:
                m['tool_calls'] = []
        norm.append(m)
    for det in (ToolCallDetector(), CodeDetector(), MathDetector()):
        try:
            if det(norm):
                return det.intent
        except Exception:
            continue
    return None


def main() -> None:
    from twinkle_agentic.verifier import (RubricVerifier,
                                          default_intent_base_rubrics,
                                          default_intent_fixed_rubrics)

    rows = [json.loads(l) for l in open(PROCESSED)]
    scored = [r for r in rows if _lab(r, 'traj_score') is not None]
    scored.sort(key=lambda r: _lab(r, 'traj_score'))
    if not scored:
        print('no scored rows found')
        return

    picks: List[Dict[str, Any]] = []
    picks += scored[:N_LOW]                      # lowest
    mid = len(scored) // 2
    picks += scored[mid:mid + N_MID]             # middle
    picks += scored[-N_HIGH:]                    # highest
    # de-dup by id, preserve order
    seen = set()
    uniq = []
    for r in picks:
        if r.get('id') not in seen:
            seen.add(r.get('id'))
            uniq.append(r)

    intent_base = intent_fixed = None
    if RUBRIC_MODE == 'skeleton':
        intent_base = default_intent_base_rubrics()
    elif RUBRIC_MODE == 'fixed':
        intent_fixed = default_intent_fixed_rubrics()
    rv = RubricVerifier(
        max_votes=5, max_votes_long=3, min_votes_long=2, long_margin_threshold=0.18,
        min_votes_high=3, high_score_threshold=0.85,
        intent_base_rubrics=intent_base, intent_rubrics=intent_fixed)
    print(f'[audit] rubric_mode={RUBRIC_MODE}  repeat={REPEAT}')

    for r in uniq:
        stored = _lab(r, 'traj_score')
        seg_scores = _lab(r, 'segment_scores')
        messages = r.get('messages') or []
        intent = _infer_intent(messages)
        print('=' * 100)
        print(f"id={r.get('id')}  model={r.get('model_id')}  n_msgs={len(messages)}  intent={intent}")
        print(f"stored traj_score={stored}  level={_lab(r,'traj_level')}  "
              f"segment_scores={seg_scores}  safety={_lab(r,'safety_score')}")
        print('-- trajectory summary --')
        print(_seg_summary(messages))
        traj = {'messages': messages}
        if r.get('tools'):
            traj['tools'] = r['tools']

        scalars: List[float] = []
        det = None
        for _ in range(max(1, REPEAT)):
            try:
                det = rv.score_detail(traj, intent=intent)
            except Exception as e:
                print(f'!! re-score failed: {e}')
                det = None
                break
            scalars.append(det.scalar)
        if det is None:
            continue
        print('-- teacher re-score --')
        print(f"  llm_scalar={det.llm_scalar:.3f}  hard_pass_rate={det.hard_pass_rate:.3f}  "
              f"scalar={det.scalar:.3f}  gated={det.gated}  n_votes={det.n_votes}")
        if REPEAT > 1:
            lo, hi = min(scalars), max(scalars)
            mean = sum(scalars) / len(scalars)
            var = sum((s - mean) ** 2 for s in scalars) / len(scalars)
            print(f"  [variance over {REPEAT}] mean={mean:.3f}  spread={hi - lo:.3f}  "
                  f"std={var ** 0.5:.3f}  scalars={[round(s, 3) for s in scalars]}")
        rubric = det.rubric or []
        rates = det.per_item_pass_rate or []
        for i, it in enumerate(rubric):
            rate = rates[i] if i < len(rates) else None
            kind = 'HARD' if getattr(it, 'is_hard', False) else 'prin'
            rate_s = 'n/a' if rate is None else f'{rate:.2f}'
            print(f'   [{kind}] pass={rate_s}  {it.text}')
    print('=' * 100)


if __name__ == '__main__':
    main()
