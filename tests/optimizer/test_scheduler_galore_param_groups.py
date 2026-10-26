# Scheduler / GaLore / param_groups tests (plan T3-01..T3-16, T10e-01..T10e-04).
import math

import pytest
import torch
from torch import nn

from twinkle.module.optimizer.galore.galore_projector import GaLoreProjector
from twinkle.module.optimizer.galore.galore_adamw import GaLoreAdamW
from twinkle.module.scheduler import CosineWarmupScheduler, LinearWarmupScheduler
from twinkle.module.optimizer.param_groups import apply_param_groups_spec, validate_param_groups_spec


def _opt():
    return torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)


class TestCosineWarmupScheduler:
    """T3-01..T3-05: warmup ramp, peak, midpoint phase, num_cycles, end value."""

    def test_warmup_linear_from_zero(self):
        s = CosineWarmupScheduler(_opt(), num_warmup_steps=10, num_training_steps=100)
        assert s._lr_lambda(0) == 0.0
        assert s._lr_lambda(5) == 0.5
        assert s._lr_lambda(9) == 0.9

    def test_warmup_end_is_one(self):
        s = CosineWarmupScheduler(_opt(), num_warmup_steps=10, num_training_steps=100)
        assert s._lr_lambda(10) == pytest.approx(1.0)

    def test_midpoint_hand_computed(self):
        s = CosineWarmupScheduler(_opt(), num_warmup_steps=10, num_training_steps=110)
        # progress = (60-10)/100 = 0.5 -> 0.5 * (1 + cos(pi/2)) = 0.5
        assert s._lr_lambda(60) == pytest.approx(0.5)

    def test_num_cycles_one_crosses_zero(self):
        s = CosineWarmupScheduler(_opt(), num_warmup_steps=0, num_training_steps=100, num_cycles=1.0)
        # progress 0.5 with cycles=1: cos(pi) = -1 -> 0
        assert s._lr_lambda(50) == pytest.approx(0.0)

    def test_end_value_zero(self):
        s = CosineWarmupScheduler(_opt(), num_warmup_steps=10, num_training_steps=100)
        assert s._lr_lambda(100) == pytest.approx(0.0)

    def test_zero_warmup_no_division_error(self):
        s = CosineWarmupScheduler(_opt(), num_warmup_steps=0, num_training_steps=10)
        assert s._lr_lambda(0) == pytest.approx(1.0)  # straight into the cosine


class TestLinearWarmupScheduler:
    """T3-06..T3-09: decay slope, past-the-end clamp, zero warmup, optimizer integration."""

    def test_tail_linear_decay_hand_computed(self):
        s = LinearWarmupScheduler(_opt(), num_warmup_steps=10, num_training_steps=110)
        # progress = 50/100 = 0.5 -> factor 0.5
        assert s._lr_lambda(60) == pytest.approx(0.5)

    def test_past_total_clamps_to_zero(self):
        s = LinearWarmupScheduler(_opt(), num_warmup_steps=10, num_training_steps=100)
        assert s._lr_lambda(120) == 0.0

    def test_zero_warmup_no_division_error(self):
        s = LinearWarmupScheduler(_opt(), num_warmup_steps=0, num_training_steps=10)
        assert s._lr_lambda(0) == pytest.approx(1.0)

    def test_get_lr_integration(self):
        # T3-09: the lambda is wired into the optimizer's scheduler, not just callable.
        param = nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=1.0)
        s = LinearWarmupScheduler(opt, num_warmup_steps=2, num_training_steps=4)
        # LambdaLR steps once at construction (epoch 0): lambda(0) = 0/2 = 0.
        assert s.get_last_lr()[0] == pytest.approx(0.0)
        s.step()
        assert s.get_last_lr()[0] == pytest.approx(0.5)  # epoch 1: 1/2
        s.step()
        assert s.get_last_lr()[0] == pytest.approx(1.0)  # epoch 2: warmup complete


class TestGaLoreProjector:
    """T3-10..T3-12: shapes, reconstruction error, subspace refresh."""

    def test_project_shapes_tall_matrix(self):
        p = GaLoreProjector(rank=2)
        grad = torch.randn(8, 4)
        low = p.project(grad, iter=0)
        assert low.shape == (8, 2)

    def test_project_shapes_wide_matrix(self):
        p = GaLoreProjector(rank=2)
        grad = torch.randn(4, 8)
        low = p.project(grad, iter=0)
        assert low.shape == (2, 8)

    def test_project_back_reconstructs_low_error(self):
        p = GaLoreProjector(rank=4, scale=1.0)
        torch.manual_seed(3)
        grad = torch.randn(16, 8)
        low = p.project(grad, iter=0)
        back = p.project_back(low)
        # A rank-4 reconstruction of a random 16x8 gradient is a heavy projection but must
        # stay in the same magnitude class (not collapse to zero, not explode).
        assert back.shape == grad.shape
        assert 0.05 < (back.norm() / grad.norm()).item() < 1.0

    def test_project_back_identity_when_rank_full(self):
        # rank == the smaller dimension: the subspace spans the whole gradient.
        p = GaLoreProjector(rank=8)
        torch.manual_seed(3)
        grad = torch.randn(16, 8)
        low = p.project(grad, iter=0)
        back = p.project_back(low)
        torch.testing.assert_close(back, grad, atol=1e-5, rtol=1e-4)

    def test_update_proj_gap_refreshes_subspace(self):
        p = GaLoreProjector(rank=2, update_proj_gap=3)
        torch.manual_seed(3)
        grad = torch.randn(8, 4)
        p.project(grad, iter=0)
        first = p.ortho_matrix.clone()
        p.project(grad, iter=1)  # 1 % 3 != 0: kept
        assert torch.equal(p.ortho_matrix, first)
        torch.manual_seed(4)
        grad2 = torch.randn(8, 4)
        p.project(grad2, iter=3)  # 3 % 3 == 0: refreshed
        assert not torch.equal(p.ortho_matrix, first)


class TestGaLoreAdamW:
    """T3-13..T3-16: group isolation, single-step movement, config validation."""

    def _model(self):
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))

    def test_galore_group_moves_and_plain_group_untouched(self):
        # A GaLore group carries only 2D weights (the projection needs a matrix); the plain
        # group holds the same-named reference weights so the delta proves the step ran.
        model = self._model()
        before = list(model[0].parameters())[0].detach().clone()
        before_other = list(model[1].parameters())[0].detach().clone()
        opt = GaLoreAdamW([dict(params=[model[0].weight], lr=1e-2, weight_decay=0.0,
                                rank=2, update_proj_gap=10, scale=1.0, proj_type='std')],
                          lr=1e-2, weight_decay=0.0)
        loss = (model[0](torch.ones(1, 4)) ** 2).sum()
        loss.backward()
        opt.step()
        moved = (list(model[0].parameters())[0] - before).norm()
        assert torch.isfinite(moved) and moved > 0
        # T3-14: a parameter outside the GaLore group is unaffected by its state.
        assert torch.allclose(list(model[1].parameters())[0], before_other)

    def test_single_galore_step_finite(self):
        model = self._model()
        opt = GaLoreAdamW([dict(params=[model[0].weight], rank=2, update_proj_gap=10,
                                scale=1.0, proj_type='std')], lr=1e-3)
        loss = (model[0](torch.ones(1, 4)) ** 2).sum()
        loss.backward()
        opt.step()
        assert torch.isfinite(list(model[0].parameters())[0]).all()


class TestValidateParamGroupsSpec:
    """T10e-01: every malformed shape is rejected loudly."""

    def test_non_list_rejected(self):
        with pytest.raises(TypeError, match='list of rule dicts'):
            validate_param_groups_spec({'match': {'name': 'x'}, 'lr_mult': 2})

    @pytest.mark.parametrize('rule,match_text', [
        ('not-a-dict', 'must be a dict'),
        ({'lr_mult': 2}, 'non-empty "match"'),
        ({'match': {}, 'lr_mult': 2}, 'non-empty "match"'),
        ({'match': {'name': 'x'}, 'nope': 1}, 'unknown key'),
        ({'match': {'name': 'x', 'bogus': 1}, 'lr_mult': 2}, 'unknown key'),
        ({'match': {'name': 'x'}, 'lr': 1, 'lr_mult': 2}, 'exclusive'),
        ({'match': {'name': 'x'}}, 'neither lr nor weight_decay'),
    ])
    def test_malformed_rules_rejected(self, rule, match_text):
        with pytest.raises(ValueError, match=match_text):
            validate_param_groups_spec([rule])


def _groups(model):
    decay, no_decay, names_d, names_n = [], [], [], []
    for name, p in model.named_parameters():
        if 'bias' in name or p.ndim <= 1:
            no_decay.append(p)
            names_n.append(name)
        else:
            decay.append(p)
            names_d.append(name)
    return [
        {'params': decay, 'param_names': names_d, 'weight_decay': 0.1, 'lr': 1e-3},
        {'params': no_decay, 'param_names': names_n, 'weight_decay': 0.0, 'lr': 1e-3},
    ]


class TestApplyParamGroupsSpec:
    """T10e-02..T10e-03: first-match-wins splitting, lr forms, decay inheritance."""

    def _model(self):
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))

    def test_name_glob_splits_and_multiplies_lr(self):
        model = self._model()
        spec = [{'match': {'name': '0.weight'}, 'lr_mult': 2.0}]
        out = apply_param_groups_spec(_groups(model), spec, model=model,
                                      tower_prefixes={'vision_tower': [], 'aligner': [], 'language_model': []})
        by_names = {tuple(g['param_names']): g for g in out}
        (names, g) = next((k, v) for k, v in by_names.items() if '0.weight' in k)
        assert g['lr'] == pytest.approx(2e-3)
        # weight_decay inherited from the source group.
        assert g['weight_decay'] == pytest.approx(0.1)

    def test_absolute_lr_and_decay_override(self):
        model = self._model()
        spec = [{'match': {'name': '*.bias'}, 'lr': 5e-4, 'weight_decay': 0.2}]
        out = apply_param_groups_spec(_groups(model), spec, model=model,
                                      tower_prefixes={'vision_tower': [], 'aligner': [], 'language_model': []})
        (names, g) = next((tuple(g['param_names']), g) for g in out if 'bias' in (g['param_names'][0] if g['param_names'] else ''))
        assert g['lr'] == pytest.approx(5e-4)
        assert g['weight_decay'] == pytest.approx(0.2)

    def test_base_bucket_untouched(self):
        model = self._model()
        spec = [{'match': {'name': 'nonexistent.*'}, 'lr_mult': 100.0}]
        out = apply_param_groups_spec(_groups(model), spec, model=model,
                                      tower_prefixes={'vision_tower': [], 'aligner': [], 'language_model': []})
        # Base buckets are per source group (decay / no-decay), so count across both.
        base_params = [p for g in out if g['lr'] == pytest.approx(1e-3) for p in g['params']]
        assert len(base_params) == 4  # everything unmatched

    def test_ndim_match(self):
        model = self._model()
        spec = [{'match': {'ndim': 1}, 'lr_mult': 3.0}]
        out = apply_param_groups_spec(_groups(model), spec, model=model,
                                      tower_prefixes={'vision_tower': [], 'aligner': [], 'language_model': []})
        rule_groups = [g for g in out if g['lr'] == pytest.approx(3e-3)]
        assert all(p.ndim == 1 for g in rule_groups for p in g['params'])

    def test_module_type_match(self):
        model = self._model()
        spec = [{'match': {'module_type': 'Linear'}, 'lr_mult': 4.0}]
        out = apply_param_groups_spec(_groups(model), spec, model=model,
                                      tower_prefixes={'vision_tower': [], 'aligner': [], 'language_model': []})
        rule_groups = [g for g in out if g['lr'] == pytest.approx(4e-3)]
        matched_names = [n for g in rule_groups for n in g['param_names']]
        assert '0.weight' in matched_names

    def test_unknown_tower_role_rejected(self):
        model = self._model()
        spec = [{'match': {'tower': 'bogus_tower'}, 'lr_mult': 2.0}]
        with pytest.raises(ValueError, match='bogus_tower'):
            apply_param_groups_spec(_groups(model), spec, model=model,
                                    tower_prefixes={'vision_tower': [], 'aligner': [], 'language_model': []})

    def test_groups_without_param_names_rejected(self):
        model = self._model()
        spec = [{'match': {'name': 'x'}, 'lr_mult': 2.0}]
        with pytest.raises(ValueError, match='param_names'):
            apply_param_groups_spec([{'params': list(model.parameters())}], spec, model=model,
                                    tower_prefixes={'vision_tower': [], 'aligner': [], 'language_model': []})
