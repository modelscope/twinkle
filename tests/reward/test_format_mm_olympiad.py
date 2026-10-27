# FormatReward / MultiModalAccuracyReward / OlympiadBench rewards (plan T1-29..T1-32).
import pytest

from twinkle.reward.format_reward import FormatReward
from twinkle.reward.mm_reward import MultiModalAccuracyReward
from twinkle.reward.olympiad_bench import OlympiadBenchAccuracyReward, OlympiadBenchFormatReward, OlympiadBenchQualityReward


def _traj(content, gt=None, extra_user_data=()):
    user_data = []
    if gt is not None:
        user_data.append(('ground_truth', gt))
    user_data.extend(extra_user_data)
    traj = {'messages': [{'role': 'assistant', 'content': content}]}
    if user_data:
        traj['user_data'] = user_data
    return traj


class TestFormatReward:
    """T1-29: both the think and the answer tag are required."""

    def test_both_tags_score_one(self):
        assert FormatReward()([_traj('<think>so</think><answer>42</answer>')]) == [1.0]

    def test_missing_either_tag_scores_zero(self):
        assert FormatReward()([_traj('<answer>42</answer>')]) == [0.0]
        assert FormatReward()([_traj('<think>so</think>')]) == [0.0]
        assert FormatReward()([_traj('plain')]) == [0.0]


class TestMultiModalAccuracyReward:
    """T1-30: <answer> extraction, case-insensitive match, numeric tolerance."""

    def test_answer_tag_extraction_case_insensitive(self):
        reward = MultiModalAccuracyReward()
        assert reward([_traj('<answer>blue</answer>', 'Blue')]) == [1.0]
        assert reward([_traj('<answer>red</answer>', 'Blue')]) == [0.0]

    def test_numeric_tolerance_branch(self):
        reward = MultiModalAccuracyReward()
        assert reward([_traj('<answer>0.5</answer>', '0.500001')]) == [1.0]
        assert reward([_traj('<answer>0.5</answer>', '0.6')]) == [0.0]

    def test_missing_answer_tag_scores_zero(self):
        reward = MultiModalAccuracyReward()
        assert reward([_traj('blue', 'Blue')]) == [0.0]

    def test_multimodal_part_list_content(self):
        # last_assistant_text walks list-of-parts contents; the <answer> tag must still be found.
        traj = {'messages': [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'look'}]}, {
            'role': 'assistant', 'content': [{'type': 'text', 'text': '<answer>2</answer>'}]}],
            'user_data': [('ground_truth', '2')]}
        assert MultiModalAccuracyReward()([traj]) == [1.0]


class TestOlympiadBenchAccuracyReward:
    """T1-31: normalization, numeric tolerance with partial credit, and multi-answer split."""

    def test_single_answer_exact_and_normalization(self):
        reward = OlympiadBenchAccuracyReward()
        assert reward([_traj('\\boxed{42}', '42')]) == [1.0]
        # Unit suffix and $..$ shells normalize away.
        assert reward([_traj('\\boxed{$42 cm$}', '42')]) == [1.0]

    def test_numeric_match_within_tolerance(self):
        reward = OlympiadBenchAccuracyReward()
        assert reward([_traj('\\boxed{100.4}', '100')]) == [1.0]  # rel err 0.4% < 1%

    def test_partial_credit_decays_with_error(self):
        reward = OlympiadBenchAccuracyReward()
        close = reward([_traj('\\boxed{105}', '100')])[0]
        far = reward([_traj('\\boxed{200}', '100')])[0]
        assert 0.0 < close < 1.0
        assert far < close

    def test_multiple_answer_any_hit(self):
        reward = OlympiadBenchAccuracyReward()
        # _split_answers splits on commas only, so the multi-answer gt uses a comma.
        traj = _traj('\\boxed{1} and \\boxed{2}', '1,2', extra_user_data=[('is_multiple_answer', True)])
        assert reward([traj]) == [1.0]
        # One of two gt parts hit -> 0.5
        traj_half = _traj('\\boxed{1}', '1,2', extra_user_data=[('is_multiple_answer', True)])
        assert reward([traj_half]) == [0.5]
        # Neither part hit -> 0.0
        traj_none = _traj('\\boxed{9}', '1,2', extra_user_data=[('is_multiple_answer', True)])
        assert reward([traj_none]) == [0.0]

    def test_no_boxed_scores_zero(self):
        reward = OlympiadBenchAccuracyReward()
        assert reward([_traj('Answer: 3', '3')]) == [0.0]


class TestOlympiadBenchFormatReward:
    """T1-32 (format half): presence + near-end + consistency scoring, hand-computed."""

    def test_perfect_format(self):
        # boxed present (0.4) + near end (0.3) + single consistent answer (0.3) = 1.0
        assert OlympiadBenchFormatReward()([_traj('work \\boxed{42}')]) == [1.0]

    def test_conflicting_answers_penalized(self):
        # boxed (0.4) + near end (0.3) - three distinct answers (0.3) = 0.4
        assert OlympiadBenchFormatReward()([_traj('\\boxed{1} \\boxed{2} \\boxed{3}')])[0] == pytest.approx(0.4)

    def test_no_boxed_scores_zero(self):
        assert OlympiadBenchFormatReward()([_traj('no answer')]) == [0.0]


class TestOlympiadBenchQualityReward:
    """T1-32 (quality half): the three scoring components, hand-computed."""

    def test_reasoning_markers_score(self):
        reward = OlympiadBenchQualityReward()
        # Four reasoning ingredients: 3+ bare '=', 5+ operators, a Step marker, a conclusion word
        # followed by \boxed. Length 300..1500 -> 1.0; unique content -> 1.0.
        pad = ''.join(f'filler line {i} with token{chr(97 + i % 26)}\n' for i in range(20))
        # Two DIFFERENT operator lines: >=5 operator characters without duplicated n-grams.
        completion = ('Step 1\n' + 'a = 1\nb = 2\nc = 3\n' + 'x + y - z * w ^ 2\n' + 'p / q * r ^ s\n'
                      + pad + '所以 \\boxed{42}')
        assert 300 <= len(completion) <= 1500
        score = reward([_traj(completion)])[0]
        assert score == pytest.approx(1.0)

    def test_empty_completion_low_score(self):
        reward = OlympiadBenchQualityReward()
        score = reward([_traj('x')])[0]
        # reasoning=0, length=1/100*0.3, repetition=1 (<100 chars) -> 0.3*0.003 + 0.3*1 = 0.3009
        assert abs(score - 0.3009) < 1e-9

    def test_length_curve_favour_zone(self):
        reward = OlympiadBenchQualityReward()
        assert reward._length_score(800) == 1.0
        assert reward._length_score(50) == 0.15
        assert reward._length_score(2000) < 1.0
        assert reward._length_score(2000) > 0.0
