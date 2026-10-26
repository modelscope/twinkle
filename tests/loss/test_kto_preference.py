# KTO / PreferenceLossBase / distance-metric tests (plan T10a-01..T10a-03, T10a-12, T10a-13).
import math

import pytest
import torch

from twinkle.loss import KTOLoss
from twinkle.loss.dpo import PreferenceLossBase
from twinkle.loss.infonce import cosine_distance, euclidean_distance, manhattan_distance


def _kto_batch():
    """Two unpaired completions; labels valid on the first two positions of each row.

    Per-token logps are chosen so the SEQUENCE log-ratios are exactly +1 and -1:
        row 0 (desirable):   policy [.5, .5] - ref [0, 0]   -> +1.0
        row 1 (undesirable): policy [-.5, -.5] - ref [0, 0] -> -1.0
    """
    labels = torch.tensor([[7, 8, -100, -100], [7, 8, -100, -100]])
    policy = torch.tensor([[0.5, 0.5, 0.0, 0.0], [-0.5, -0.5, 0.0, 0.0]])
    ref = torch.zeros_like(policy)
    return {'labels': labels}, {'logps': policy}, ref


class TestKTOLoss:
    """T10a-01/T10a-02: the KTO prospect-theory formula and the z_KL anchor, hand-computed."""

    def test_symmetric_pair_hand_computed(self):
        # beta=0.5, z_kl=0: both sides lose 1 - sigmoid(0.5) = 0.377541.
        loss_fn = KTOLoss(beta=0.5)
        inputs, outputs, ref = _kto_batch()
        out = loss_fn(inputs, outputs, label=[True, False], z_kl=0.0, ref_logps=ref)
        expected = 1.0 / (1.0 + math.exp(0.5))
        assert out['loss'].item() == pytest.approx(expected, abs=1e-6)

    def test_z_kl_shifts_both_sides(self):
        # z_kl=1: desirable 1 - sigmoid(0.5*(1-1)) = 0.5; undesirable 1 - sigmoid(0.5*(1+1)) = 0.268941.
        loss_fn = KTOLoss(beta=0.5)
        inputs, outputs, ref = _kto_batch()
        out = loss_fn(inputs, outputs, label=[True, False], z_kl=1.0, ref_logps=ref)
        des = 0.5
        undes = 1.0 / (1.0 + math.exp(1.0))
        assert out['loss'].item() == pytest.approx((des + undes) / 2, abs=1e-6)

    def test_imbalance_weights_scale_sides(self):
        # desirable_weight=2 doubles the desirable term inside the SAME cat-mean denominator.
        loss_fn = KTOLoss(beta=0.5, desirable_weight=2.0)
        inputs, outputs, ref = _kto_batch()
        out = loss_fn(inputs, outputs, label=[True, False], z_kl=0.0, ref_logps=ref)
        base = 1.0 / (1.0 + math.exp(0.5))
        assert out['loss'].item() == pytest.approx((2 * base + base) / 2, abs=1e-6)

    def test_all_desirable_batch(self):
        # An empty rejected side is legal: the cat holds only the desirable terms.
        loss_fn = KTOLoss(beta=0.5)
        inputs, outputs, ref = _kto_batch()
        out = loss_fn(inputs, outputs, label=[True, True], z_kl=0.0, ref_logps=ref)
        des = 1.0 / (1.0 + math.exp(0.5))              # logratio +1: 1 - sigmoid(0.5)
        undes = 1.0 / (1.0 + math.exp(-0.5))          # logratio -1: 1 - sigmoid(-0.5)
        assert out['loss'].item() == pytest.approx((des + undes) / 2, abs=1e-6)

    def test_missing_label_raises(self):
        loss_fn = KTOLoss()
        inputs, outputs, ref = _kto_batch()
        with pytest.raises(ValueError, match='label'):
            loss_fn(inputs, outputs, ref_logps=ref)

    def test_missing_reference_raises(self):
        # There is no reference-free KTO; a missing ref must fail loudly, not anchor on zero.
        loss_fn = KTOLoss()
        inputs, outputs, _ = _kto_batch()
        with pytest.raises(ValueError, match='reference logps'):
            loss_fn(inputs, outputs, label=[True, False])

    def test_label_count_mismatch_raises(self):
        loss_fn = KTOLoss()
        inputs, outputs, ref = _kto_batch()
        with pytest.raises(ValueError, match='does not match'):
            loss_fn(inputs, outputs, label=[True], z_kl=0.0, ref_logps=ref)

    def test_ref_outputs_used_when_ref_logps_missing(self):
        # ref_outputs carrying 'logps' is the alternate reference plumbing.
        loss_fn = KTOLoss(beta=0.5)
        inputs, outputs, ref = _kto_batch()
        out = loss_fn(inputs, outputs, label=[True, False], z_kl=0.0, ref_outputs={'logps': ref})
        expected = 1.0 / (1.0 + math.exp(0.5))
        assert out['loss'].item() == pytest.approx(expected, abs=1e-6)


class TestPreferenceLossBase:
    """T10a-12: the shared base carries ignore_index for every preference subclass."""

    def test_kto_is_preference_base(self):
        assert isinstance(KTOLoss(), PreferenceLossBase)

    def test_ignore_index_forwarded(self):
        assert KTOLoss(ignore_index=-1).ignore_index == -1


class TestDistanceMetrics:
    """T10a-13: the three Siamese distance metrics, exact hand-computed values."""

    def test_cosine_orthogonal_is_one(self):
        # [0,3] vs [4,0]: similarity 0 -> distance 1.
        x, y = torch.tensor([[0.0, 3.0]]), torch.tensor([[4.0, 0.0]])
        assert cosine_distance(x, y).item() == pytest.approx(1.0, abs=1e-6)

    def test_euclidean_manhattan_hand_computed(self):
        x, y = torch.tensor([[0.0, 3.0]]), torch.tensor([[4.0, 0.0]])
        assert euclidean_distance(x, y).item() == pytest.approx(5.0, abs=1e-6)
        assert manhattan_distance(x, y).item() == pytest.approx(7.0, abs=1e-6)

    def test_pairwise_rows_independent(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        out = euclidean_distance(x, y)
        # F.pairwise_distance adds eps=1e-6 under the sqrt, so identical rows are ~1.4e-6, not 0.
        assert out[0].item() == pytest.approx(0.0, abs=1e-5)
        assert out[1].item() == pytest.approx(math.sqrt(2.0), abs=1e-5)
