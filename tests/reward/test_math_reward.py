# MathReward / BoxedMathAccuracyReward tests (plan T1-16..T1-22).
import pytest

from twinkle.reward.boxed_math import BoxedMathAccuracyReward
from twinkle.reward.math_reward import MathReward


def _traj(content, gt_key='solution', gt=None, flat_gt=None):
    traj = {'messages': [{'role': 'assistant', 'content': content}]}
    if gt is not None:
        traj['user_data'] = [(gt_key, gt)]
    if flat_gt is not None:
        traj[gt_key] = flat_gt
    return traj


class TestExtractBoxedResult:
    """T1-16: no-boxed input returns the ORIGINAL text (the documented fallback)."""

    def test_no_boxed_returns_original(self):
        assert MathReward.extract_boxed_result('plain answer') == 'plain answer'

    def test_boxed_extraction_nested(self):
        assert MathReward.extract_boxed_result('a \\boxed{\\frac{1}{2}} b') == '\\frac{1}{2}'

    def test_unbalanced_boxed_returns_original(self):
        assert MathReward.extract_boxed_result('\\boxed{3') == '\\boxed{3'


class TestMathRewardScoring:
    """T1-17/T1-18: gold fallback, flat-column gold, and the '# Answer' split."""

    def test_missing_gold_falls_back_to_self_comparison(self):
        # No user_data and no flat column: scoring against itself must yield 1.0, not raise.
        assert MathReward()([_traj('\\boxed{7}')]) == [1.0]

    def test_flat_column_gold(self):
        # ground_truth() reads the packed user_data first, the flat column second.
        reward = MathReward(ground_truth_key='solution')
        assert reward([_traj('\\boxed{7}', flat_gt='7')]) == [1.0]
        assert reward([_traj('\\boxed{8}', flat_gt='7')]) == [0.0]

    def test_answer_marker_split(self):
        reward = MathReward()
        pred = 'reasoning # Answer\n\\boxed{7}'
        gold = 'work # Answer\n\\boxed{7}'
        assert reward([_traj(pred, gt=gold)]) == [1.0]
        # Only one side carrying the marker still compares the boxed content.
        assert reward([_traj('reasoning # Answer\n\\boxed{8}', gt='\\boxed{7}')]) == [0.0]

    def test_gold_of_zero_is_a_real_label(self):
        # `is None` -- not falsy -- decides absence, so a gold of 0 must NOT fall back to self.
        reward = MathReward(ground_truth_key='ground_truth')
        assert reward([_traj('\\boxed{0}', gt_key='ground_truth', gt=0)]) == [1.0]


class TestCompareConsecutive:
    """T1-19/T1-20: sympy equivalence, and the cleaned-string fallback."""

    def test_sympy_equivalence(self):
        parse = pytest.importorskip('sympy.parsing.latex').parse_latex
        try:
            parse('x+x')
        except ImportError:
            pytest.skip('antlr4 runtime version incompatible with sympy parse_latex')
        assert MathReward.compare_consecutive('x+x', '2x')
        assert not MathReward.compare_consecutive('x+x', '3x')

    def test_fallback_string_comparison(self):
        # Non-LaTeX inputs: parse fails on both sides -> cleaned string equality decides.
        assert MathReward.compare_consecutive('abc', 'abc') is True
        assert MathReward.compare_consecutive('abc', 'abd') is False


class TestBoxedMathAccuracyReward:
    """T1-21/T1-22: the boxed gate and Decimal thousands-separator equality."""

    def test_no_boxed_scores_zero_even_with_answer_line(self):
        reward = BoxedMathAccuracyReward()
        assert reward([_traj('Answer: 42', gt_key='ground_truth', gt='42')]) == [0.0]

    def test_decimal_thousands_separator(self):
        reward = BoxedMathAccuracyReward()
        assert reward([_traj('\\boxed{1,000}', gt_key='ground_truth', gt='1000')]) == [1.0]
        assert reward([_traj('\\boxed{1,000}', gt_key='ground_truth', gt='1001')]) == [0.0]

    def test_boxed_decimal_mismatch_zero(self):
        reward = BoxedMathAccuracyReward()
        assert reward([_traj('\\boxed{3.5}', gt_key='ground_truth', gt='3.6')]) == [0.0]
