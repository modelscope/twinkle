"""Rubric stabilization: skeleton prepend, intent routing, high-band voting.

These exercise :class:`RubricVerifier` without any real LLM by stubbing the two
distilled hooks (``_gen_rubric`` / ``_score_once``) and forcing
``_llm_available`` True, so we test the *assembly + voting* logic in isolation.
"""
from twinkle_agentic.preprocessor.intents import INTENT_CODE, INTENT_TOOL_CALL
from twinkle_agentic.verifier import RubricItem, RubricVerifier
from twinkle_agentic.verifier.rubric_library import default_intent_base_rubrics


def _segment():
    return {'messages': [
        {'role': 'user', 'content': 'do the thing'},
        {'role': 'assistant', 'content': 'here is the result'},
    ]}


def _stub_llm(rv, *, gen_lines, score_seq):
    """Force LLM-available and deterministic gen/score outputs.

    ``score_seq`` is a list of raw verdict strings returned by successive
    ``_score_once`` calls (so we can count how many votes were spent).
    """
    rv._llm_available = lambda: True  # type: ignore[method-assign]
    rv._gen_rubric = lambda **kw: gen_lines  # type: ignore[method-assign]
    calls = {'n': 0}

    def _score_once(**kw):
        i = min(calls['n'], len(score_seq) - 1)
        calls['n'] += 1
        return score_seq[i]

    rv._score_once = _score_once  # type: ignore[method-assign]
    return calls


def test_base_rubric_prepended_and_dedup():
    base = [RubricItem('The agent calls tools with valid JSON', True)]
    rv = RubricVerifier(base_rubric=base, min_rubrics=1, max_rubrics=6)
    gen = ('1. The agent calls tools with valid JSON [Hard Rule]\n'  # dup of skeleton
           '2. The response advances the sub-goal [Principle]')
    _stub_llm(rv, gen_lines=gen, score_seq=['1: PASS\n2: PASS'])
    detail = rv.score_detail(_segment())
    texts = [it.text for it in detail.rubric]
    # skeleton first, duplicate from generation dropped -> exactly 2 items
    assert texts[0] == 'The agent calls tools with valid JSON'
    assert len(detail.rubric) == 2


def test_intent_fixed_rubric_skips_generation():
    fixed = {INTENT_TOOL_CALL: [RubricItem('The agent uses tools correctly', True)]}
    rv = RubricVerifier(intent_rubrics=fixed)
    # gen would raise if called -> proves generation is skipped for this intent
    rv._llm_available = lambda: True  # type: ignore[method-assign]
    rv._gen_rubric = lambda **kw: (_ for _ in ()).throw(AssertionError('gen called'))  # type: ignore
    rv._score_once = lambda **kw: '1: PASS'  # type: ignore[method-assign]
    detail = rv.score_detail(_segment(), intent=INTENT_TOOL_CALL)
    assert len(detail.rubric) == 1
    assert detail.rubric[0].text == 'The agent uses tools correctly'


def test_intent_base_rubric_routes_by_intent():
    rv = RubricVerifier(intent_base_rubrics=default_intent_base_rubrics(),
                        min_rubrics=1, max_rubrics=8)
    _stub_llm(rv, gen_lines='1. The response is coherent [Principle]',
              score_seq=['1: PASS\n2: PASS\n3: PASS\n4: PASS'])
    detail = rv.score_detail(_segment(), intent=INTENT_CODE)
    # the CODE skeleton leads the rubric
    assert detail.rubric[0].text.startswith('The response produces code')


def test_high_band_forces_more_votes():
    rv = RubricVerifier(min_rubrics=1, max_rubrics=4, min_votes_high=3,
                        high_score_threshold=0.85, max_votes=5)
    gen = '1. The response is correct [Hard Rule]'
    # first pass all-PASS -> scalar 1.0 (>= 0.85) -> must escalate to >=3 votes
    calls = _stub_llm(rv, gen_lines=gen, score_seq=['1: PASS'])
    detail = rv.score_detail(_segment())
    assert detail.n_votes >= 3
    assert calls['n'] >= 3


def test_low_band_single_vote():
    rv = RubricVerifier(min_rubrics=1, max_rubrics=4, min_votes_high=3,
                        high_score_threshold=0.85, max_votes=5, margin_threshold=0.1)
    gen = '1. The response is correct [Hard Rule]'
    # first pass FAIL -> scalar 0.0, decisive (far from 0.5) -> single vote
    calls = _stub_llm(rv, gen_lines=gen, score_seq=['1: FAIL'])
    detail = rv.score_detail(_segment())
    assert detail.n_votes == 1
    assert calls['n'] == 1
