from twinkle_agentic.verifier.aggregation import _combine


def test_hard_soft_blend_high_hard_not_one_shot_veto():
    # product would be 0.9 * 0.1 = 0.09
    blended = _combine(0.9, 0.1, 'hard_soft_blend')
    assert blended > 0.35
    assert blended < 0.9


def test_hard_soft_blend_low_hard_uses_soft():
    assert _combine(0.5, 0.4, 'hard_soft_blend') == 0.5 * 0.4
