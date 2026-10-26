# GSM8K-family reward tests (plan T1-01..T1-13, T1-14..T1-15).
import pytest

from twinkle.reward.gsm8k import (GSM8KAccuracyBrevityReward, GSM8KAccuracyReward, GSM8KBrevityReward,
                                  GSM8KFormatReward, MathVerifyAccuracyReward, _extract_last_boxed)


def _traj(content, gt):
    return {'messages': [{'role': 'assistant', 'content': content}], 'user_data': [('ground_truth', gt)]}


class TestExtractLastBoxed:
    """T1-01/T1-02: nested-brace depth counting and the unbalanced guard."""

    def test_nested_braces_extracted_intact(self):
        # Depth counting must return the FULL inner span, not stop at the first '}'.
        assert _extract_last_boxed('x \\boxed{\\frac{1}{2}} y') == '\\frac{1}{2}'
        assert _extract_last_boxed('\\boxed{{a}{b}}') == '{a}{b}'

    def test_unbalanced_returns_empty(self):
        # A truncated \boxed{ must never yield a half answer (it would score as wrong-but-parseable).
        assert _extract_last_boxed('\\boxed{3') == ''
        assert _extract_last_boxed('no answer') == ''

    def test_last_boxed_wins(self):
        assert _extract_last_boxed('\\boxed{1} mid \\boxed{2}') == '2'


class TestGSM8KExtractAnswer:
    """T1-03/T1-04/T1-05: precedence, the 500-char tail window, and the #### regex classes."""

    def test_boxed_preferred_over_hash(self):
        assert GSM8KAccuracyReward.extract_answer('#### 41 and \\boxed{99}') == '99'

    def test_tail_window_drops_early_boxed_keeps_hash(self):
        # boxed at offset ~20 falls OUTSIDE the [-500:] window; #### at ~590 stays inside.
        completion = 'a' * 20 + '\\boxed{99}' + 'b' * 560 + '#### 41'
        assert GSM8KAccuracyReward.extract_answer(completion) == '41'

    def test_hash_negative_thousands(self):
        assert GSM8KAccuracyReward.extract_answer('#### -1,234.5') == '-1234.5'

    def test_no_answer_returns_empty(self):
        assert GSM8KAccuracyReward.extract_answer('just reasoning') == ''


class TestGSM8KAccuracyReward:
    """T1-06/T1-07/T1-08: numeric tolerance, string fallback, empty inputs."""

    def test_numeric_tolerance(self):
        reward = GSM8KAccuracyReward()
        assert reward([_traj('\\boxed{42}', '42.0')]) == [1.0]
        assert reward([_traj('\\boxed{42}', '43')]) == [0.0]

    def test_non_numeric_string_fallback(self):
        reward = GSM8KAccuracyReward()
        # float('x=1') raises ValueError -> the string-equality branch must catch it.
        assert reward([_traj('\\boxed{x=1}', 'x=1')]) == [1.0]
        assert reward([_traj('\\boxed{x=1}', 'x=2')]) == [0.0]

    def test_missing_prediction_or_gt_scores_zero(self):
        reward = GSM8KAccuracyReward()
        assert reward([_traj('no answer here', '42')]) == [0.0]
        assert reward([{'messages': [{'role': 'assistant', 'content': '\\boxed{42}'}], 'user_data': []}]) == [0.0]


class TestGSM8KBrevityReward:
    """T1-09/T1-10: the answer gate and the linear decay, hand-computed."""

    def test_no_answer_zero(self):
        assert GSM8KBrevityReward()([_traj('pure reasoning, no answer', '42')]) == [0.0]

    def test_linear_decay_hand_computed(self):
        reward = GSM8KBrevityReward(full_reward_length=300, decay_length=3000)
        assert reward([_traj('#### 1', '1')]) == [1.0]  # short answer, len << full
        # len = 300 + 1500 -> 1 - 1500/3000 = 0.5. '#### 1' is 6 chars, so pad to exactly 1800.
        completion = 'x' * (1800 - 6) + '#### 1'
        assert len(completion) == 1800
        assert reward([_traj(completion, '1')])[0] == pytest.approx(0.5)
        # Past the decay horizon the floor is exactly zero.
        long = 'y' * (3400 - 6) + '#### 1'
        assert reward([_traj(long, '1')]) == [0.0]


class TestGSM8KAccuracyBrevityReward:
    """T1-11/T1-12: weighted composition and the metric payload keys."""

    def test_weighted_composition_hand_computed(self):
        reward = GSM8KAccuracyBrevityReward(accuracy_weight=2.0, brevity_weight=0.5)
        # accuracy=1.0 (correct), brevity=1.0 (short) -> 2.0*1 + 0.5*1 = 2.5
        assert reward([_traj('\\boxed{42}', '42')]) == [2.5]
        # accuracy=0, brevity=0 (no answer) -> 0.0
        assert reward([_traj('no answer', '42')]) == [0.0]

    def test_metric_payload_keys(self):
        reward = GSM8KAccuracyBrevityReward()
        trajs = [_traj('\\boxed{42}', '42'), _traj('no answer', '42')]
        out = reward.metric_payload(trajs, rewards=reward(trajs))
        assert set(out) == {'total_reward', 'accuracy_reward', 'brevity_reward'}
        assert out['accuracy_reward'] == 0.5


class TestGSM8KFormatReward:
    """T1-13: format presence for either anchor."""

    def test_either_anchor_scores_one(self):
        reward = GSM8KFormatReward()
        assert reward([_traj('#### 41', '41')]) == [1.0]
        assert reward([_traj('\\boxed{9}', '9')]) == [1.0]
        assert reward([_traj('neither', '9')]) == [0.0]


class TestMathVerifyAccuracyReward:
    """T1-14/T1-15: symbolic equivalence and the never-raises guarantee."""

    def test_symbolic_equivalence(self):
        pytest.importorskip('math_verify')
        reward = MathVerifyAccuracyReward()
        assert reward([_traj('\\boxed{1/2}', '0.5')]) == [1.0]
        assert reward([_traj('\\boxed{0.6}', '1/2')]) == [0.0]

    def test_garbage_never_raises(self):
        pytest.importorskip('math_verify')
        reward = MathVerifyAccuracyReward()
        # Unparseable completion or gold must score 0.0, never bubble into the rollout loop.
        assert reward([_traj('#### ??###', '42'), _traj('\\boxed{1}', '')]) == [0.0, 0.0]
