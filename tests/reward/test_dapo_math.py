# DAPOMathAccuracyReward / DAPOMathReward tests (plan T1-23..T1-28).
import pytest

from twinkle.reward.dapo_math import DAPOMathAccuracyReward, DAPOMathReward


def _traj(content, gt, completion_length=None):
    traj = {'messages': [{'role': 'assistant', 'content': content}], 'user_data': [('ground_truth', gt)]}
    if completion_length is not None:
        traj['completion_length'] = completion_length
    return traj


class TestExtractAnswer:
    """T1-23: the Answer: line grammar -- last match, case-insensitive, trailing dot, $..$ shell."""

    def test_last_answer_line_wins(self):
        text = 'Answer: 1\nthen more\nAnswer: 7'
        assert DAPOMathAccuracyReward.extract_answer(text) == '7'

    def test_case_insensitive_and_trailing_dot(self):
        assert DAPOMathAccuracyReward.extract_answer('answer: 42.') == '42'

    def test_dollar_shell_stripped(self):
        assert DAPOMathAccuracyReward.extract_answer('Answer: $42$') == '42'

    def test_boxed_preferred_over_answer_line(self):
        assert DAPOMathAccuracyReward.extract_answer('\\boxed{9} Answer: 1') == '9'

    def test_no_anchor_returns_empty(self):
        assert DAPOMathAccuracyReward.extract_answer('nothing here') == ''


class TestDAPOMathAccuracyReward:
    def test_decimal_match_and_miss(self):
        reward = DAPOMathAccuracyReward()
        assert reward([_traj('Answer: 1,000', '1000')]) == [1.0]
        assert reward([_traj('Answer: 1000', '1001')]) == [0.0]

    def test_missing_prediction_or_gt(self):
        reward = DAPOMathAccuracyReward()
        assert reward([_traj('no anchor', '42'), _traj('Answer: 42', '')]) == [0.0, 0.0]


class TestDAPOMathRewardConstruction:
    """T1-24: all four constructor guards fire loudly."""

    @pytest.mark.parametrize('kwargs', [
        {'max_response_length': 0, 'overlong_buffer_length': 1},
        {'max_response_length': 10, 'overlong_buffer_length': 0},
        {'max_response_length': 10, 'overlong_buffer_length': 11},
        {'max_response_length': 10, 'overlong_buffer_length': 5, 'overlong_penalty_factor': -1.0},
        {'max_response_length': 10, 'overlong_buffer_length': 5, 'score_tail_chars': 0},
    ])
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            DAPOMathReward(**kwargs)


class TestDAPOMathRewardScoring:
    """T1-25/T1-26/T1-27: the overlong penalty slope, the +/-1 sign convention, and the
    completion_length contract. max=100, buffer=20 -> expected=80, slope 1/20 per token."""

    def _reward(self):
        return DAPOMathReward(max_response_length=100, overlong_buffer_length=20, overlong_penalty_factor=1.0)

    def test_overlong_penalty_hand_computed(self):
        reward = self._reward()
        assert reward._overlong_reward(_traj('Answer: 1', '1', completion_length=80)) == 0.0
        assert reward._overlong_reward(_traj('Answer: 1', '1', completion_length=70)) == 0.0  # below: clamped
        assert reward._overlong_reward(_traj('Answer: 1', '1', completion_length=90)) == -0.5
        assert reward._overlong_reward(_traj('Answer: 1', '1', completion_length=100)) == -1.0

    def test_penalty_factor_scales_slope(self):
        reward = DAPOMathReward(max_response_length=100, overlong_buffer_length=20, overlong_penalty_factor=2.0)
        assert reward._overlong_reward(_traj('Answer: 1', '1', completion_length=90)) == -1.0

    def test_sign_convention(self):
        reward = self._reward()
        assert reward([_traj('Answer: 7', '7', completion_length=50)]) == [1.0]
        assert reward([_traj('Answer: 8', '7', completion_length=50)]) == [-1.0]
        # Overlong correct answer: +1 - 0.5 = +0.5
        assert reward([_traj('Answer: 7', '7', completion_length=90)]) == [0.5]

    def test_missing_completion_length_raises(self):
        with pytest.raises(ValueError, match='completion_length'):
            self._reward()([_traj('Answer: 7', '7')])

    def test_score_tail_chars_window(self):
        # The Answer: anchor 400 chars in is OUTSIDE the default 300-char scoring tail.
        # The anchor must sit at the start of its own line (the regex anchors on ^).
        reward = self._reward()
        far = '\nAnswer: 7\n' + 'x' * 400
        assert reward([_traj(far, '7', completion_length=50)]) == [-1.0]
        near = 'x' * 100 + '\nAnswer: 7'
        assert reward([_traj(near, '7', completion_length=50)]) == [1.0]


class TestDAPOMathRewardPayload:
    """T1-28: empty-batch payload is all-zero with all four keys, not a ZeroDivisionError."""

    def test_empty_batch_payload(self):
        reward = DAPOMathReward(max_response_length=100, overlong_buffer_length=20)
        out = reward.metric_payload([], rewards=[])
        assert out == {'total_reward': 0.0, 'accuracy_reward': 0.0, 'overlong_reward': 0.0, 'overlong_ratio': 0.0}

    def test_payload_values(self):
        reward = DAPOMathReward(max_response_length=100, overlong_buffer_length=20)
        trajs = [
            _traj('Answer: 7', '7', completion_length=50),
            _traj('Answer: 8', '7', completion_length=90),  # wrong (-1) + overlong (-0.5)
        ]
        out = reward.metric_payload(trajs, rewards=reward(trajs))
        assert out['accuracy_reward'] == 0.5
        assert out['overlong_ratio'] == 0.5
        assert out['total_reward'] == (1.0 + -1.5) / 2
