"""Diagnosis parsing must survive truncated JSON.

The diagnostic pass emits a full per-criterion (verdict+reason+fix) JSON. When a
teacher reply is cut off mid-stream (too small a token budget), the outer object
never closes. We must still recover every COMPLETE item object rather than
degrade to an all-FAIL, reason-less vector (which is useless as SFT data).
"""
from twinkle_agentic.verifier.rubric_verifier import (RubricVerifier,
                                                      _salvage_diag_items)

# A realistic truncated response: 3 complete items (one reason contains LaTeX
# braces to exercise string-aware brace matching), then cut off mid-string.
TRUNCATED = (
    '{\n  "items": [\n'
    '    {"index": 1, "verdict": "PASS", "reason": "No tool calls needed.", "fix": ""},\n'
    '    {"index": 2, "verdict": "FAIL", "reason": "Duplicated \\\\subsubsection*{ans} block.", '
    '"fix": "Remove the duplicate."},\n'
    '    {"index": 3, "verdict": "PASS", "reason": "Correctly omits \\\\begin{document}.", "fix": ""},\n'
    '    {"index": 4, "verdict": "FAIL", "reason": "The response is cut off right he'
)

WELL_FORMED = (
    '{"items": [{"index": 1, "verdict": "PASS", "reason": "ok", "fix": ""},'
    '{"index": 2, "verdict": "FAIL", "reason": "bad", "fix": "do x"}],'
    ' "overall": "issues", "summary": "one failure"}'
)


def test_salvage_recovers_complete_items_from_truncated_json():
    items = _salvage_diag_items(TRUNCATED)
    # 3 complete objects; the 4th (truncated) is dropped.
    assert len(items) == 3
    assert [it['verdict'] for it in items] == ['PASS', 'FAIL', 'PASS']
    # braces inside the reason string must not corrupt matching
    assert 'subsubsection' in items[1]['reason']


def test_parse_diagnosis_truncated_keeps_reasons_and_verdicts():
    items, overall_ok, summary = RubricVerifier._parse_diagnosis(TRUNCATED, n=7)
    assert len(items) == 3                       # not the all-FAIL length-7 fallback
    assert items[0].verdict is True
    assert items[1].verdict is False
    assert items[1].reason                       # the "why" survives
    assert items[1].fix == 'Remove the duplicate.'
    assert overall_ok is False                   # a FAIL present -> not ok


def test_parse_diagnosis_well_formed_still_works():
    items, overall_ok, summary = RubricVerifier._parse_diagnosis(WELL_FORMED, n=2)
    assert len(items) == 2
    assert items[0].verdict is True and items[1].verdict is False
    assert items[1].fix == 'do x'
    assert overall_ok is False
    assert summary == 'one failure'


def test_diag_sampling_params_bigger_than_scoring():
    rv = RubricVerifier(diag_max_tokens=2048)
    diag = rv._diagnose_sampling_params(None, temperature=0.0)
    score = rv._score_sampling_params(None, temperature=0.0)
    assert diag.max_tokens >= 2048
    assert diag.max_tokens > score.max_tokens    # diagnosis needs a bigger budget
