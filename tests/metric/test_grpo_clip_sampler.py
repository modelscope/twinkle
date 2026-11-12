# Copyright (c) ModelScope Contributors. All rights reserved.
"""Clip-rate panels and sampler reconciliation for the GRPO metric family.

The clip panel (``train/clip_ratio*``) only appears when the caller passes ``advantages``,
and GSPO/CISPO differ from GRPO *only* in how that panel is computed. The sampler
reconciliation keys (``train/sampler_logp_*`` / ``train/sampler_token_delta``) only appear
when ``sampler_logps`` is passed; they are assertion-style probes for engine drift.

Every case drives real tensors through ``accumulate`` -> ``calculate`` and pins the reported
numbers against hand-computed log ratios.
"""
import math

import pytest
import torch

from twinkle.metric import CISPOMetric, GRPOMetric, GSPOMetric


def _metric(cls, **kwargs):
    """Instantiate a metric without distributed groups (gather_results returns local)."""
    return cls(device_mesh=None, process_group=None, **kwargs)


# ---------------------------------------------------------------------------
# GRPO: token-level clip rate, gated by advantage sign
# ---------------------------------------------------------------------------

class TestGRPOClipPanel:

    def test_low_and_high_clips_are_gated_by_advantage_sign(self):
        # ratios [0.5, 1.5]; token 0 is worsened by a negative advantage -> low clip,
        # token 1 is improved by a positive advantage -> high clip.
        m = _metric(GRPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, math.log(0.5), math.log(1.5)]])
        old = [[0.0, 0.0]]  # response-only form: len == number of masked positions
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=old, advantages=[[-1.0, 1.0]])

        result = m.calculate()

        assert result['train/clip_ratio_low'] == pytest.approx(0.5)
        assert result['train/clip_ratio_high'] == pytest.approx(0.5)
        assert result['train/clip_ratio'] == pytest.approx(1.0)

    def test_advantage_sign_gating_skips_the_other_side(self):
        # Same ratios, but the signs are swapped: a cheap token with a positive advantage and
        # an expensive token with a negative advantage are both left unclipped.
        m = _metric(GRPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, math.log(0.5), math.log(1.5)]])
        old = [[0.0, 0.0]]
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=old, advantages=[[1.0, -1.0]])

        result = m.calculate()

        assert result['train/clip_ratio_low'] == pytest.approx(0.0)
        assert result['train/clip_ratio_high'] == pytest.approx(0.0)
        assert result['train/clip_ratio'] == pytest.approx(0.0)

    def test_epsilon_high_extends_only_the_upper_bound(self):
        # ratio 1.3 sits between 1 + epsilon (1.2) and 1 + epsilon_high (1.5) -> not clipped;
        # ratio 1.6 exceeds the high bound -> clipped.
        m = _metric(GRPOMetric, epsilon=0.2, epsilon_high=0.5)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, math.log(1.3), math.log(1.6)]])
        old = [[0.0, 0.0]]
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=old, advantages=[[1.0, 1.0]])

        result = m.calculate()

        assert result['train/clip_ratio_low'] == pytest.approx(0.0)
        assert result['train/clip_ratio_high'] == pytest.approx(0.5)
        assert result['train/clip_ratio'] == pytest.approx(0.5)

    def test_clip_panel_is_absent_without_advantages(self):
        m = _metric(GRPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, -1.0, -2.0]])
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=[[0.0, 0.0]])

        result = m.calculate()

        assert 'train/clip_ratio' not in result


# ---------------------------------------------------------------------------
# CISPO: token-level clip rate, unconditional on advantage sign
# ---------------------------------------------------------------------------

class TestCISPOClipPanel:

    def test_clip_is_unconditional_on_advantage_sign(self):
        # The advantage signs below would suppress both clips under GRPO; CISPO still counts them.
        m = _metric(CISPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, math.log(0.5), math.log(1.5)]])
        old = [[0.0, 0.0]]
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=old, advantages=[[1.0, -1.0]])

        result = m.calculate()

        assert result['train/clip_ratio_low'] == pytest.approx(0.5)
        assert result['train/clip_ratio_high'] == pytest.approx(0.5)
        assert result['train/clip_ratio'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GSPO: clip rate over the per-sequence geometric-mean ratio
# ---------------------------------------------------------------------------

class TestGSPOClipPanel:

    def test_sequence_mean_ratio_decides_the_clip(self):
        # Sequence 0: mean log ratio +0.6 -> ratio e^0.6 > 1.2 with a positive advantage -> high.
        # Sequence 1: mean log ratio -0.5 -> ratio e^-0.5 < 0.8 with a negative advantage -> low.
        m = _metric(GSPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11], [-100, -100, 20, 21]])
        logps = torch.tensor([
            [0.0, 0.0, 0.6, 0.6],
            [0.0, 0.0, -0.5, -0.5],
        ])
        old = [[0.0, 0.0], [0.0, 0.0]]
        m.accumulate(
            {'labels': labels}, {'logps': logps}, old_logps=old, advantages=[[1.0, 1.0], [-1.0, -1.0]])

        result = m.calculate()

        # One clipped sequence out of two valid sequences on each side.
        assert result['train/clip_ratio_low'] == pytest.approx(0.5)
        assert result['train/clip_ratio_high'] == pytest.approx(0.5)
        assert result['train/clip_ratio'] == pytest.approx(1.0)

    def test_token_extremes_cancel_in_the_sequence_mean(self):
        # Per-token ratios are e^0.7 and e^-0.7 (both far outside epsilon), but the sequence-mean
        # log ratio is 0 -> GSPO clips nothing, unlike GRPO's token-level panel.
        m = _metric(GSPOMetric)
        labels = torch.tensor([[-100, 10, 11]])
        logps = torch.tensor([[0.0, 0.7, -0.7]])
        old = [[0.0, 0.0]]
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=old, advantages=[[1.0, 1.0]])

        result = m.calculate()

        assert result['train/clip_ratio_low'] == pytest.approx(0.0)
        assert result['train/clip_ratio_high'] == pytest.approx(0.0)
        assert result['train/clip_ratio'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sampler reconciliation (``sampler_logps``)
# ---------------------------------------------------------------------------

class TestSamplerReconciliation:

    def test_matching_sampler_logps_report_zero_drift(self):
        m = _metric(GRPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, -1.0, -2.0]])
        m.accumulate({'labels': labels}, {'logps': logps}, sampler_logps=[[-1.0, -2.0]])

        result = m.calculate()

        assert result['train/sampler_logp_mae'] == pytest.approx(0.0)
        assert result['train/sampler_logp_max_abs'] == pytest.approx(0.0)
        assert result['train/sampler_token_delta'] == 0

    def test_mae_and_max_track_known_drift(self):
        # |(-1.0) - (-1.1)| = 0.1 and |(-2.0) - (-2.4)| = 0.4 -> mae 0.25, max 0.4.
        m = _metric(GRPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, -1.0, -2.0]])
        m.accumulate({'labels': labels}, {'logps': logps}, sampler_logps=[[-1.1, -2.4]])

        result = m.calculate()

        assert result['train/sampler_logp_mae'] == pytest.approx(0.25)
        assert result['train/sampler_logp_max_abs'] == pytest.approx(0.4)
        assert result['train/sampler_token_delta'] == 0

    def test_token_delta_reports_missing_sampler_tokens(self):
        # Six generated tokens total; the sampler only accounts for five (row 1 is short), so the
        # delta must be 1. The short row cannot be aligned, so it contributes nothing to the MAE.
        m = _metric(GRPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11, 12, 13], [-100, -100, -100, 20, 21, -100]])
        logps = torch.zeros(2, 6)
        logps[0, 2:6] = -1.0
        m.accumulate({'labels': labels}, {'logps': logps}, sampler_logps=[[-1.0] * 4, [-1.0]])

        result = m.calculate()

        assert result['train/sampler_token_delta'] == 1
        assert result['train/sampler_logp_mae'] == pytest.approx(0.0)
        assert result['train/sampler_logp_max_abs'] == pytest.approx(0.0)

    def test_reconciliation_keys_are_absent_without_sampler_logps(self):
        m = _metric(GRPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, -1.0, -2.0]])
        m.accumulate({'labels': labels}, {'logps': logps})

        result = m.calculate()

        assert 'train/sampler_logp_mae' not in result
        assert 'train/sampler_logp_max_abs' not in result
        assert 'train/sampler_token_delta' not in result


# ---------------------------------------------------------------------------
# High-KL token probe (``top_k_kl``)
# ---------------------------------------------------------------------------

class TestHighKLRecords:

    def test_top_k_kl_emits_the_largest_positive_token(self):
        # kl = exp(d) - d - 1: d=+1 -> 0.718, d=-1 -> 0.368. top_k_kl=1 keeps the larger one.
        m = _metric(GRPOMetric, top_k_kl=1)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, 1.0, -1.0]])
        old = [[0.0, 0.0]]
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=old)

        records = m.calculate()['_high_kl_records']

        assert len(records) == 1
        record = records[0]
        assert record['gsi'] == 0
        assert record['pos'] == 2
        assert record['token_id'] == 10
        assert record['kl'] == pytest.approx(math.e - 2)
        assert record['ratio'] == pytest.approx(math.e)
        assert record['logp_new'] == pytest.approx(1.0)
        assert record['logp_old'] == pytest.approx(0.0)

    def test_top_k_kl_is_absent_when_disabled(self):
        m = _metric(GRPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, 1.0, -1.0]])
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=[[0.0, 0.0]])

        assert '_high_kl_records' not in m.calculate()

    def test_gsi_cursor_advances_across_accumulate_calls(self):
        m = _metric(GRPOMetric, top_k_kl=2)
        m.accumulate(
            {'labels': torch.tensor([[-100, 10], [-100, 11]])},
            {'logps': torch.tensor([[0.0, 1.0], [0.0, 0.5]])},
            old_logps=[[0.0], [0.0]],
        )
        m.accumulate(
            {'labels': torch.tensor([[-100, 12]])},
            {'logps': torch.tensor([[0.0, 0.9]])},
            old_logps=[[0.0]],
        )

        records = m.calculate()['_high_kl_records']

        assert [record['gsi'] for record in records] == [0, 1, 2]
        assert [record['token_id'] for record in records] == [10, 11, 12]


# ---------------------------------------------------------------------------
# Temperature rescaling and multi-micro-batch bookkeeping
# ---------------------------------------------------------------------------

class TestGRPORescalingAndBatching:

    def test_temperature_rescales_new_and_old_consistently(self):
        # new - old is [1, 0] at T=1, so K3 = ((e - 2) + 0) / 2. At temperature=2 both sides
        # double and the difference becomes [2, 0], giving ((e^2 - 3) + 0) / 2.
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0.0, 0.0, -1.0, -2.0]])
        old = [[-2.0, -2.0]]

        plain = _metric(GRPOMetric)
        plain.accumulate({'labels': labels}, {'logps': logps}, old_logps=old)
        plain_result = plain.calculate()

        warm = _metric(GRPOMetric, temperature=2.0)
        warm.accumulate({'labels': labels}, {'logps': logps}, old_logps=old)
        warm_result = warm.calculate()

        assert plain_result['train/mean_new_logp'] == pytest.approx(-1.5)
        assert warm_result['train/mean_new_logp'] == pytest.approx(-3.0)
        assert plain_result['train/logp_diff_mean'] == pytest.approx(0.5)
        assert warm_result['train/logp_diff_mean'] == pytest.approx(1.0)
        assert plain_result['train/approx_kl'] == pytest.approx((math.exp(1.0) - 2.0) / 2)
        assert warm_result['train/approx_kl'] == pytest.approx((math.exp(2.0) - 3.0) / 2)
        # The tail probe reads raw logps before rescaling, so it stays at the sampler value.
        assert warm_result['train/logp_min'] == pytest.approx(-2.0)

    def test_multiple_micro_batches_advance_the_old_logps_cursor(self):
        m = _metric(GRPOMetric)
        inputs = [
            {'labels': torch.tensor([[-100, 10, 11]])},
            {'labels': torch.tensor([[-100, 12, 13]])},
        ]
        logps = [torch.tensor([[0.0, -1.0, -2.0]]), torch.tensor([[0.0, -3.0, -4.0]])]
        m.accumulate(
            inputs,
            {'logps': logps},
            # Distinct rows: a frozen cursor would re-read [0, 0] and report
            # mean_old_logp 0.0 / logp_diff_mean -2.5 instead of the values below.
            old_logps=[[0.0, 0.0], [-2.0, -2.0]],
            advantages=[[-1.0, -1.0], [-1.0, -1.0]],
        )

        result = m.calculate()

        assert result['train/n_trainable_tokens'] == 4
        assert result['train/mean_new_logp'] == pytest.approx(-2.5)
        assert result['train/mean_old_logp'] == pytest.approx(-1.0)
        assert result['train/logp_diff_mean'] == pytest.approx(-1.5)
        # Every ratio is below 0.8 with a negative advantage -> all four tokens low-clipped.
        assert result['train/clip_ratio_low'] == pytest.approx(1.0)

    def test_single_logps_tensor_merges_parallel_micro_batches(self):
        # One (2, seq) logps tensor with two same-width mb inputs is merged into one batch.
        m = _metric(GRPOMetric)
        inputs = [
            {'labels': torch.tensor([[-100, 10, -100, -100]])},
            {'labels': torch.tensor([[-100, -100, 11, -100]])},
        ]
        logps = torch.tensor([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -2.0, 0.0]])
        m.accumulate(inputs, {'logps': logps}, old_logps=torch.zeros(2, 4))

        result = m.calculate()

        assert result['train/n_trainable_tokens'] == 2
        assert result['train/mean_new_logp'] == pytest.approx(-1.5)
        assert result['train/logp_diff_mean'] == pytest.approx(-1.5)

    def test_unmerged_mismatched_width_micro_batches_only_score_the_first(self):
        # Different seq widths cannot be merged; only mb[0] is accumulated and mb[1] is dropped.
        m = _metric(GRPOMetric)
        inputs = [
            {'labels': torch.tensor([[-100, 10, -100, -100]])},
            {'labels': torch.tensor([[-100, 11, -100, -100, -100]])},
        ]
        logps = torch.tensor([[0.0, -1.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0]])
        m.accumulate(inputs, {'logps': logps})

        result = m.calculate()

        assert result['train/n_trainable_tokens'] == 1
        assert result['train/mean_new_logp'] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Input normalisation and defensive skips
# ---------------------------------------------------------------------------

class TestGRPOInputHandling:

    def test_empty_logps_is_a_no_op(self):
        for empty in ([], torch.tensor([]), [torch.tensor([])], 'not-logps'):
            m = _metric(GRPOMetric)
            m.accumulate({'labels': torch.tensor([[-100, 10]])}, {'logps': empty})
            assert m.calculate() == {}

    def test_high_kl_probe_skips_when_no_token_has_positive_kl(self):
        # new == old on every token -> kl = 0 -> n_pos = 0 -> top_k_kl has nothing to report.
        m = _metric(GRPOMetric, top_k_kl=1)
        m.accumulate(
            {'labels': torch.tensor([[-100, 10]])}, {'logps': torch.tensor([[0.0, -1.0]])}, old_logps=[[-1.0]])
        assert '_high_kl_records' not in m.calculate()

    def test_all_masked_batch_reports_nothing(self):
        m = _metric(GRPOMetric)
        m.accumulate({'labels': torch.tensor([[-100, -100]])}, {'logps': torch.zeros(1, 2)})
        assert m.calculate() == {}

    def test_one_dimensional_labels_are_promoted(self):
        m = _metric(GRPOMetric)
        m.accumulate({'labels': torch.tensor([10, 11])}, {'logps': torch.tensor([[-1.0, -2.0]])})
        result = m.calculate()
        assert result['train/n_trainable_tokens'] == 2
        assert result['train/mean_new_logp'] == pytest.approx(-1.5)

    def test_sequence_width_mismatch_truncates_both_sides(self):
        # logps is 3 wide while labels is 5: both are cut to 3, so only the first masked token
        # (position 2, logp -3.0) is scored.
        m = _metric(GRPOMetric)
        m.accumulate(
            {'labels': torch.tensor([[-100, -100, 10, 11, 12]])},
            {'logps': torch.tensor([[-1.0, -2.0, -3.0]])},
        )
        result = m.calculate()
        assert result['train/n_trainable_tokens'] == 1
        assert result['train/mean_new_logp'] == pytest.approx(-3.0)

    def test_entropy_wider_than_labels_is_truncated(self):
        m = _metric(GRPOMetric)
        m.accumulate(
            {'labels': torch.tensor([[-100, -100, 10, 11]])},
            {'logps': torch.zeros(1, 4), 'entropies': torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])},
        )
        # Only the two masked positions of the first four columns count: (3 + 4) / 2.
        assert m.calculate()['train/entropy'] == pytest.approx(3.5)

    def test_entropy_row_mismatch_keeps_the_leading_rows(self):
        m = _metric(GRPOMetric)
        m.accumulate(
            {'labels': torch.tensor([[-100, -100, 10, 11]])},
            {'logps': torch.zeros(1, 4),
             'entropies': torch.tensor([[1.0, 2.0, 3.0, 4.0], [9.0, 9.0, 9.0, 9.0], [8.0, 8.0, 8.0, 8.0]])},
        )
        assert m.calculate()['train/entropy'] == pytest.approx(3.5)

    def test_non_dict_and_label_less_micro_batches_are_skipped(self):
        m = _metric(GRPOMetric)
        inputs = [{'labels': None}, 'not-a-dict', {'labels': torch.tensor([[-100, 10]])}]
        logps = [torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([[0.0, -1.0]])]
        m.accumulate(inputs, {'logps': logps})
        result = m.calculate()
        assert result['train/n_trainable_tokens'] == 1
        assert result['train/mean_new_logp'] == pytest.approx(-1.0)

    def test_skipped_micro_batches_still_advance_the_old_logps_cursor(self):
        # Both dropped microbatches still own a row in the flattened old_logps list, so the
        # surviving mb must read the third row instead of re-reading the first.
        m = _metric(GRPOMetric)
        inputs = [{'labels': None}, 'not-a-dict', {'labels': torch.tensor([[-100, 10]])}]
        logps = [torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([[0.0, -1.0]])]
        m.accumulate(inputs, {'logps': logps}, old_logps=[[0.0], [0.0], [-2.0]])

        result = m.calculate()

        assert result['train/n_trainable_tokens'] == 1
        assert result['train/mean_old_logp'] == pytest.approx(-2.0)
        assert result['train/logp_diff_mean'] == pytest.approx(1.0)
