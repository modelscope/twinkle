# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for preprocessor.utils — pure logprob math helpers.

These helpers compute conditional-vs-unconditional logprob deltas for
IFD-family scoring (CherryLLM, T-SHIRT, ChR).  All functions are stateless
and accept simple list inputs.

Conventions used in this test file:
  * "lp" lists are aligned to the FULL sequence (prompt + answer).
  * ``n_prompt`` is the number of prompt tokens; assistant tokens start at
    index ``n_prompt`` in the cond list.
  * Each lp entry is a dict {token_id: logprob_float}.
"""
import math
import pytest

from twinkle_agentic.preprocessor.utils import (_chr_min_distinct, _chr_min_weighted, _extract_logprob,
                                                _ifd_family_metrics, _lp_to_jsonable, _mean_logprob_delta, _pad_batch,
                                                _to_int_list)

# ── _extract_logprob ────────────────────────────────────────────────────────


class TestExtractLogprob:

    def test_none(self):
        assert _extract_logprob(None) is None

    def test_scalar_int(self):
        assert _extract_logprob(5) == 5.0

    def test_scalar_float(self):
        assert _extract_logprob(-1.2) == -1.2

    def test_dict_with_int_token_id(self):
        lp = {7: -0.5, 8: -2.0}
        assert _extract_logprob(lp, token_id=7) == -0.5
        assert _extract_logprob(lp, token_id=8) == -2.0

    def test_dict_with_str_token_id_fallback(self):
        # vLLM may emit string keys; lookup must fall back to str(token_id).
        lp = {'7': -0.5}
        assert _extract_logprob(lp, token_id=7) == -0.5

    def test_dict_no_token_id_picks_first(self):
        # No token_id → iter-first behaviour.
        lp = {7: -0.5}
        assert _extract_logprob(lp) == -0.5

    def test_dict_token_id_missing_uses_first(self):
        # token_id not in dict → fall back to first entry.
        lp = {99: -3.0}
        assert _extract_logprob(lp, token_id=7) == -3.0

    def test_dict_with_logprob_attr_object(self):

        class Entry:

            def __init__(self, v):
                self.logprob = v

        lp = {7: Entry(-0.7)}
        assert _extract_logprob(lp, token_id=7) == -0.7

    def test_dict_with_nested_dict(self):
        lp = {7: {'logprob': -0.9, 'rank': 1}}
        assert _extract_logprob(lp, token_id=7) == -0.9

    def test_dict_with_nested_dict_none_logprob(self):
        lp = {7: {'logprob': None}}
        assert _extract_logprob(lp, token_id=7) is None

    def test_unrecognized_type(self):
        # str entries → returns None
        lp = {7: 'oops'}
        assert _extract_logprob(lp, token_id=7) is None

    def test_non_dict_non_scalar(self):
        # A list is neither scalar nor dict → None.
        assert _extract_logprob([1, 2, 3]) is None


# ── _to_int_list ────────────────────────────────────────────────────────────


class TestToIntList:

    def test_plain_list(self):
        assert _to_int_list([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert _to_int_list((1, 2, 3)) == [1, 2, 3]

    def test_with_tolist(self):

        class Tensor:

            def tolist(self):
                return [4, 5, 6]

        assert _to_int_list(Tensor()) == [4, 5, 6]

    def test_empty(self):
        assert _to_int_list([]) == []


# ── _chr_min_distinct ───────────────────────────────────────────────────────


class TestChrMinDistinct:

    def test_empty_inputs_returns_none(self):
        assert _chr_min_distinct([], [{1: -1.0}], [], [1], 0) is None
        assert _chr_min_distinct([{1: -1.0}], [], [1], [], 0) is None
        assert _chr_min_distinct([{1: -1.0}], [{1: -1.0}], [1], [], 0) is None

    def test_simple_all_positive(self):
        # cond_lp[i] - asst_lp[i] > 0 for all i → ratio = 1.0
        n_prompt = 1
        # cond covers prompt(1) + asst(2) = 3 positions
        cond_lp = [
            {
                0: -10.0
            },  # prompt position
            {
                1: -0.1
            },  # asst pos 0 — high cond logprob
            {
                2: -0.2
            }
        ]  # asst pos 1
        asst_lp = [{1: -1.0}, {2: -1.5}]
        cond_ids = [0, 1, 2]
        asst_ids = [1, 2]
        ratio = _chr_min_distinct(cond_lp, asst_lp, cond_ids, asst_ids, n_prompt)
        assert ratio == 1.0

    def test_all_negative(self):
        # delta < 0 → ratio = 0
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: -3.0}, {2: -3.0}]
        asst_lp = [{1: -0.5}, {2: -0.5}]
        ratio = _chr_min_distinct(cond_lp, asst_lp, [0, 1, 2], [1, 2], n_prompt)
        assert ratio == 0.0

    def test_distinct_token_min_aggregation(self):
        # Two occurrences of same token: one has +delta, one has -delta.
        # min(deltas) is negative → token contributes 0 to ratio.
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: -0.1}, {1: -3.0}]
        asst_lp = [{1: -1.0}, {1: -0.5}]  # delta1=+0.9, delta2=-2.5
        ratio = _chr_min_distinct(cond_lp, asst_lp, [0, 1, 1], [1, 1], n_prompt)
        assert ratio == 0.0  # min < 0

    def test_exclude_ids(self):
        # Excluded token is dropped before counting.
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: -0.1}, {2: -0.1}]
        asst_lp = [{1: -1.0}, {2: -1.0}]
        # Without exclude: 2 distinct tokens, both positive → 1.0
        ratio = _chr_min_distinct(cond_lp, asst_lp, [0, 1, 2], [1, 2], n_prompt, exclude_ids={1})
        assert ratio == 1.0  # only token 2 counted, still positive

    def test_truncation_when_cond_short(self):
        # cond_lp shorter than n_prompt + n_asst → loop breaks early.
        n_prompt = 2
        cond_lp = [{0: 0.0}, {0: 0.0}, {1: -0.1}]  # only 1 asst position
        asst_lp = [{1: -1.0}, {2: -1.0}]  # 2 asst positions requested
        ratio = _chr_min_distinct(cond_lp, asst_lp, [0, 0, 1], [1, 2], n_prompt)
        assert ratio == 1.0  # only the first delta processed


# ── _chr_min_weighted ───────────────────────────────────────────────────────


class TestChrMinWeighted:

    def test_empty_returns_none(self):
        assert _chr_min_weighted([], [{1: -1.0}], [], [1], 0) is None

    def test_all_positive_returns_one(self):
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: -0.1}, {2: -0.2}]
        asst_lp = [{1: -1.0}, {2: -1.5}]
        ratio = _chr_min_weighted(cond_lp, asst_lp, [0, 1, 2], [1, 2], n_prompt)
        assert ratio == 1.0  # all positive → pos_w == total_w

    def test_zero_total_weight_returns_none(self):
        # All deltas == 0 → total_w == 0 → None
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: -1.0}]
        asst_lp = [{1: -1.0}]
        assert _chr_min_weighted(cond_lp, asst_lp, [0, 1], [1], n_prompt) is None

    def test_weighted_mixture(self):
        # Token A: min_delta = +2.0  (weight 2)
        # Token B: min_delta = -1.0  (weight 1)
        # pos / total = 2 / 3
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: 1.0}, {2: -2.0}]  # cond: A=1.0, B=-2.0
        asst_lp = [{1: -1.0}, {2: -1.0}]  # asst: A=-1.0, B=-1.0
        # delta A = 1.0 - (-1.0) = 2.0
        # delta B = -2.0 - (-1.0) = -1.0
        ratio = _chr_min_weighted(cond_lp, asst_lp, [0, 1, 2], [1, 2], n_prompt)
        assert abs(ratio - 2 / 3) < 1e-9


# ── _ifd_family_metrics ─────────────────────────────────────────────────────


class TestIfdFamilyMetrics:

    def test_empty_returns_empty_dict(self):
        assert _ifd_family_metrics([], [{1: -1.0}], [], [1], 0) == {}

    def test_simple_uniform(self):
        # All deltas = 0.5 → mean=0.5, ifd=exp(-0.5)
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: -0.5}, {2: -0.5}]
        asst_lp = [{1: -1.0}, {2: -1.0}]
        out = _ifd_family_metrics(cond_lp, asst_lp, [0, 1, 2], [1, 2], n_prompt)
        assert out['n_tokens'] == 2
        assert abs(out['mean_delta'] - 0.5) < 1e-9
        assert abs(out['ifd'] - math.exp(-0.5)) < 1e-9
        # s_ifd_50 keeps top-1 by |delta| = 0.5; s_ifd_75 keeps top-2 (rounded up).
        assert abs(out['s_ifd_50'] - math.exp(-0.5)) < 1e-9
        assert abs(out['s_ifd_75'] - math.exp(-0.5)) < 1e-9

    def test_mixed_deltas(self):
        # deltas = [+2.0, -1.0]; mean = 0.5
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: 1.0}, {2: -2.0}]
        asst_lp = [{1: -1.0}, {2: -1.0}]
        out = _ifd_family_metrics(cond_lp, asst_lp, [0, 1, 2], [1, 2], n_prompt)
        assert out['n_tokens'] == 2
        assert abs(out['mean_delta'] - 0.5) < 1e-9
        # s_ifd_50 keeps top-1 by |delta| = 2.0 → exp(-2.0)
        assert abs(out['s_ifd_50'] - math.exp(-2.0)) < 1e-9


# ── _mean_logprob_delta ─────────────────────────────────────────────────────


class TestMeanLogprobDelta:

    def test_empty(self):
        assert _mean_logprob_delta([], [{1: -1.0}], [], [1], 0) is None

    def test_uniform_delta(self):
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: -0.5}, {2: -0.5}]
        asst_lp = [{1: -1.0}, {2: -1.0}]
        out = _mean_logprob_delta(cond_lp, asst_lp, [0, 1, 2], [1, 2], n_prompt)
        assert abs(out - 0.5) < 1e-9

    def test_mixed_average(self):
        # deltas = [+2.0, -1.0] → mean 0.5
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: 1.0}, {2: -2.0}]
        asst_lp = [{1: -1.0}, {2: -1.0}]
        out = _mean_logprob_delta(cond_lp, asst_lp, [0, 1, 2], [1, 2], n_prompt)
        assert abs(out - 0.5) < 1e-9

    def test_skips_none_logprobs(self):
        # When asst lp returns None, that position is skipped silently.
        n_prompt = 1
        cond_lp = [{0: 0.0}, {1: -0.5}, {2: -0.5}]
        asst_lp = [None, {2: -1.0}]
        out = _mean_logprob_delta(cond_lp, asst_lp, [0, 1, 2], [1, 2], n_prompt)
        assert abs(out - 0.5) < 1e-9  # only position 1 used


# ── _lp_to_jsonable ─────────────────────────────────────────────────────────


class TestLpToJsonable:

    def test_none_input(self):
        assert _lp_to_jsonable(None) == []

    def test_empty(self):
        assert _lp_to_jsonable([]) == []

    def test_none_passthrough(self):
        assert _lp_to_jsonable([None, None]) == [None, None]

    def test_scalar_to_float(self):
        assert _lp_to_jsonable([1, -2.0]) == [1.0, -2.0]

    def test_dict_with_logprob_object(self):

        class Entry:

            def __init__(self, lp, rank, decoded):
                self.logprob = lp
                self.rank = rank
                self.decoded_token = decoded

        out = _lp_to_jsonable([{7: Entry(-0.5, 1, 'hello')}])
        assert out == [{'7': {'logprob': -0.5, 'rank': 1, 'decoded': 'hello'}}]

    def test_dict_with_nested_dict(self):
        out = _lp_to_jsonable([{7: {'logprob': -0.5}}])
        assert out == [{'7': {'logprob': -0.5}}]

    def test_dict_with_repr_fallback(self):
        # Non-dict, non-Entry value falls back to repr string.
        out = _lp_to_jsonable([{7: 'plain'}])
        assert out == [{'7': repr('plain')}]

    def test_non_dict_non_scalar_repr(self):
        # An object that isn't dict/scalar gets repr-ed.
        out = _lp_to_jsonable([(1, 2)])
        assert out == [repr((1, 2))]


# ── _pad_batch ──────────────────────────────────────────────────────────────


class TestPadBatch:

    def test_empty_batch(self):
        padded, n = _pad_batch([], floor=4)
        assert padded == []
        assert n == 0

    def test_already_at_floor(self):
        batch = [[1], [2], [3], [4]]
        padded, n = _pad_batch(batch, floor=4)
        assert padded == batch
        assert n == 4

    def test_above_floor(self):
        batch = [[1], [2], [3], [4], [5]]
        padded, n = _pad_batch(batch, floor=3)
        assert padded == batch  # unchanged
        assert n == 5

    def test_below_floor_pads_with_last(self):
        batch = [[1], [2]]
        padded, n = _pad_batch(batch, floor=4)
        assert padded == [[1], [2], [2], [2]]
        assert n == 2  # original size

    def test_returns_new_list(self):
        batch = [[1], [2]]
        padded, _ = _pad_batch(batch, floor=4)
        # Mutating padded should not affect original.
        padded.append([99])
        assert batch == [[1], [2]]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
