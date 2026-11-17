"""GAEAdvantage.token_advantages: the recursion, its guards, and the reward shaping that feeds it.

Every expected value below is hand-computed from the recursion rather than read back from the
implementation, so a wrong ``gamma`` / ``gae_lambda`` fold, an off-by-one in the backward sweep, or
padding leaking from a longer row into a shorter one each fails a specific case.
"""
import pytest
import torch

from twinkle.advantage import GAEAdvantage
from twinkle.reward.shaping import RewardShape

ATOL = 1e-6


def _rows_close(actual, expected) -> None:
    """Compare ragged rows value by value (``pytest.approx`` rejects nested sequences)."""
    assert [len(row) for row in actual] == [len(row) for row in expected], \
        f'row lengths {[len(row) for row in actual]} != {[len(row) for row in expected]}'
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert got == pytest.approx(want, abs=ATOL), f'row {index}: {got} != {want}'


def test_single_terminal_token():
    """One token: delta = r - V, with no next value to bootstrap from."""
    gae = GAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)
    advantages, returns = gae.token_advantages([[2.0]], [[0.5]])
    _rows_close(advantages, [[1.5]])
    _rows_close(returns, [[2.0]])


def test_full_lambda_credits_every_token():
    """gamma=lam=1: the terminal reward carries all the way back, so returns = advantage + value."""
    gae = GAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)
    advantages, returns = gae.token_advantages([[0.0, 0.0, 1.0]], [[0.2, 0.3, 0.4]])
    # delta = [0+0.3-0.2, 0+0.4-0.3, 1+0-0.4] = [0.1, 0.1, 0.6]; full carry accumulates backwards.
    _rows_close(advantages, [[0.8, 0.7, 0.6]])
    _rows_close(returns, [[1.0, 1.0, 1.0]])


def test_zero_lambda_is_one_step_td():
    """lam=0 collapses GAE to the one-step TD error, with no credit carried to earlier tokens."""
    gae = GAEAdvantage(gamma=1.0, gae_lambda=0.0, normalize=False)
    advantages, returns = gae.token_advantages([[0.0, 0.0, 1.0]], [[0.5, 0.5, 0.5]])
    _rows_close(advantages, [[0.0, 0.0, 0.5]])
    _rows_close(returns, [[0.5, 0.5, 1.0]])


def test_production_discount_and_decay():
    """gamma=0.9 / lam=0.95 over a non-zero critic baseline: the parameterisation PPO actually runs."""
    gae = GAEAdvantage(gamma=0.9, gae_lambda=0.95, normalize=False)
    advantages, returns = gae.token_advantages([[0.0, 0.0, 1.0]], [[0.2, 0.3, 0.4]])
    # delta2 = 1 + 0.9*0   - 0.4 = 0.60 ; A2 = 0.60
    # delta1 = 0 + 0.9*0.4 - 0.3 = 0.06 ; A1 = 0.06 + 0.855*0.60  = 0.573
    # delta0 = 0 + 0.9*0.3 - 0.2 = 0.07 ; A0 = 0.07 + 0.855*0.573 = 0.559915
    _rows_close(advantages, [[0.559915, 0.573, 0.6]])
    _rows_close(returns, [[0.759915, 0.873, 1.0]])


def test_scalar_reward_lands_on_the_last_token():
    """A bare scalar is the terminal reward: zeros on every earlier token, the scalar on the last."""
    gae = GAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)
    from_scalar = gae.token_advantages([2.0], [[0.5, 0.5, 0.5]])
    from_row = gae.token_advantages([[0.0, 0.0, 2.0]], [[0.5, 0.5, 0.5]])
    _rows_close(from_scalar[0], from_row[0])
    _rows_close(from_scalar[1], from_row[1])


def test_ragged_rows_do_not_leak_padding():
    """Rows are ragged lists padded to a rectangle internally; a short row must not see the padding.

    Row 0 is two tokens and row 1 is four, so the internal rectangle pads row 0 with two zeros. Were
    the sweep to read past row 0's own length it would bootstrap off a padded value instead of off
    zero, and the returned row would be four long instead of two.
    """
    gae = GAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)
    advantages, returns = gae.token_advantages([[0.0, 1.0], [0.0, 0.0, 0.0, 2.0]],
                                               [[0.5, 0.5], [0.1, 0.2, 0.3, 0.4]])
    # delta = [0+0.5-0.5, 1+0-0.5] = [0.0, 0.5]; full carry -> [0.5, 0.5].
    # delta = [0.1, 0.1, 0.1, 2+0-0.4] swept backwards with full carry: 1.9, 1.8, 1.7, 1.6.
    _rows_close(advantages, [[0.5, 0.5], [1.9, 1.8, 1.7, 1.6]])
    _rows_close(returns, [[1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])


def test_normalize_whitens_over_every_valid_token():
    """normalize=True is the production default: mean 0 and std 1 across the batch's response tokens."""
    gae = GAEAdvantage(gamma=0.0, gae_lambda=0.0)
    advantages, _ = gae.token_advantages([[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]])
    flat = torch.tensor([value for row in advantages for value in row])
    assert flat.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert flat.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)


def test_normalize_skips_a_single_valid_token():
    """One token has no spread to whiten by, so the guard leaves it untouched rather than blowing it up.

    ``std`` of a single element is 0, so whitening would divide by the 1e-8 epsilon and hand the loss
    an advantage of ~5e8. The ``numel() > 1`` guard is what prevents that; this pins it.
    """
    gae = GAEAdvantage(gamma=0.0, gae_lambda=0.0, normalize=True)
    advantages, _ = gae.token_advantages([[5.0]], [[0.0]])
    _rows_close(advantages, [[5.0]])


def test_row_count_mismatch_raises():
    with pytest.raises(ValueError, match='reward rows'):
        GAEAdvantage().token_advantages([[1.0], [2.0]], [[0.0]])


def test_token_count_mismatch_raises():
    with pytest.raises(ValueError, match='response tokens'):
        GAEAdvantage().token_advantages([[1.0, 2.0]], [[0.0, 0.0, 0.0]])


def test_scalar_reward_needs_a_token_to_carry_it():
    with pytest.raises(ValueError, match='no response token'):
        GAEAdvantage().token_advantages([1.0], [[]])


@pytest.mark.parametrize('knob', ['gamma', 'gae_lambda'])
def test_out_of_range_knobs_raise_on_both_paths(knob):
    """The constructor and a per-call override are the two ways to set a knob; both must be checked.

    Validating only the override would let ``gamma=1.1`` in at construction and silently produce
    advantages that grow along the sweep instead of decaying.
    """
    with pytest.raises(ValueError, match=r'must be in \[0, 1\]'):
        GAEAdvantage(**{knob: 1.1})
    with pytest.raises(ValueError, match=r'must be in \[0, 1\]'):
        GAEAdvantage().token_advantages([[1.0]], [[0.0]], **{knob: -0.1})


def test_token_reward_shaping_puts_the_score_on_the_last_token():
    """``r' = -beta*kl_t`` per token with the ORM score added on the last: the lane GAE consumes."""
    # old_logps - ref_logps = [-1.0+1.5, -2.0+1.5] = [0.5, -0.5] under the k1 estimator.
    shaped = RewardShape()([2.0], [[0.5, -0.5]], beta=0.1, profile='token')
    _rows_close(shaped, [[-0.05, 2.05]])


def test_token_profile_requires_the_kl_lane():
    with pytest.raises(ValueError, match='per-token kl lane'):
        RewardShape()([2.0], None, beta=0.1, profile='token')
