# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the Muon/QK-Clip optimizer: param grouping, orthogonalisation, and the clip itself."""
import math
import pytest
import torch
import torch.nn as nn

from twinkle.module.optimizer import MaxLogitsTracker, MuonClip, MuonConfig, create_muon_param_groups


class Tiny(nn.Module):
    """A model with the parameter names the grouping keys off: q/k/v projections, embedding, LM head."""

    def __init__(self, d=16, vocab=32, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.embed_tokens = nn.Embedding(vocab, d)
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids, use_sdpa=False):
        h = self.embed_tokens(ids)
        b, length, d = h.shape

        def heads(x):
            return x.view(b, length, self.n_heads, d // self.n_heads).transpose(1, 2)

        q, k, v = heads(self.q_proj(h)), heads(self.k_proj(h)), heads(self.v_proj(h))
        if use_sdpa:
            ctx = nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            # [B, H, Lq, Lk]: the shape the tracker accepts as attention scores.
            ctx = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.size(-1)), dim=-1) @ v
        return self.lm_head(self.norm(self.mlp(ctx.transpose(1, 2).reshape(b, length, d))))


def param_groups(model):
    """Decay / no-decay groups, the way `TwinkleModel._create_param_group` builds them."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if ('bias' in name or 'norm' in name) else decay).append((name, param))
    groups = []
    for named, weight_decay in ((decay, 0.1), (no_decay, 0.0)):
        if named:
            groups.append({
                'params': [p for _, p in named],
                'param_names': [n for n, _ in named],
                'lr': 1e-3,
                'weight_decay': weight_decay,
            })
    return groups


def frozen_optimizer(model, config):
    """An optimizer that cannot move a weight except by clipping it, so the clip is observable alone."""
    optimizer = MuonClip(create_muon_param_groups(param_groups(model), config), lr=0.0, weight_decay=0.0)
    for group in optimizer.param_groups:
        group['lr'], group['weight_decay'] = 0.0, 0.0
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    return optimizer


class TestParamGrouping:

    def test_splits_by_role(self):
        model = Tiny()
        groups = create_muon_param_groups(param_groups(model), MuonConfig())
        roles = {}
        for group in groups:
            role = 'qk' if group['is_qk'] else ('muon' if group['apply_muon'] else 'rest')
            for name in group['param_names']:
                roles[name] = role
        assert roles['q_proj.weight'] == 'qk'
        assert roles['k_proj.weight'] == 'qk'
        assert roles['v_proj.weight'] == 'muon'
        assert roles['mlp.weight'] == 'muon'
        # 2-D but excluded by name, since orthogonalising across vocabulary rows mixes unrelated tokens.
        assert roles['embed_tokens.weight'] == 'rest'
        assert roles['lm_head.weight'] == 'rest'
        # 1-D, so there is nothing to orthogonalise.
        assert roles['q_proj.bias'] == 'rest'
        assert roles['norm.weight'] == 'rest'

    def test_keeps_every_parameter_exactly_once(self):
        model = Tiny()
        source = param_groups(model)
        groups = create_muon_param_groups(source, MuonConfig())
        before = sum(len(g['params']) for g in source)
        after = sum(len(g['params']) for g in groups)
        assert before == after
        assert all(len(g['param_names']) == len(g['params']) for g in groups)

    def test_preserves_the_incoming_group_settings(self):
        """Splitting, not rebuilding: a no-decay group must not come back with the decay group's value."""
        model = Tiny()
        groups = create_muon_param_groups(param_groups(model), MuonConfig())
        by_name = {name: group for group in groups for name in group['param_names']}
        assert by_name['q_proj.bias']['weight_decay'] == 0.0
        assert by_name['norm.weight']['weight_decay'] == 0.0
        assert by_name['q_proj.weight']['weight_decay'] == 0.1
        assert by_name['mlp.weight']['weight_decay'] == 0.1

    def test_exclude_keys_are_configurable(self):
        model = Tiny()
        config = MuonConfig(exclude_keys=['mlp'])
        groups = create_muon_param_groups(param_groups(model), config)
        muon_names = [n for g in groups if g['apply_muon'] for n in g['param_names']]
        assert 'mlp.weight' not in muon_names
        assert 'embed_tokens.weight' in muon_names


class TestNewtonSchulz:

    @pytest.mark.parametrize('shape', [(16, 16), (8, 32), (32, 8)])
    def test_flattens_the_spectrum(self, shape):
        """The point of the iteration: every singular value ends up near 1, whatever the input scale."""
        torch.manual_seed(0)
        matrix = torch.randn(*shape) * 5
        out = MuonClip.newton_schulz(matrix, steps=5)
        assert tuple(out.shape) == shape
        values = torch.linalg.svdvals(out.float())
        # Five steps of the polynomial land in a band around 1 rather than exactly on it, which is the
        # trade the Muon coefficients make; the input spectrum here spans an order of magnitude.
        assert values.min() > 0.5 and values.max() < 1.5
        assert torch.isfinite(out).all()

    def test_scale_invariant(self):
        """Input scale must not reach the output: the iteration normalises before it orthogonalises.

        Asserted on the spectrum rather than element-wise, because the normalisation runs in bfloat16 --
        eight mantissa bits, amplified by five rounds of matmuls, leave the elements agreeing only to a
        few percent even though the spectrum they describe is the same.
        """
        torch.manual_seed(0)
        matrix = torch.randn(16, 16)
        spectra = []
        for scale in (1e-3, 1.0, 1e3, 1e6):
            out = MuonClip.newton_schulz(matrix * scale, steps=5)
            values = torch.linalg.svdvals(out.float())
            assert values.min() > 0.5 and values.max() < 1.5
            spectra.append(values)
        for values in spectra[1:]:
            assert torch.allclose(spectra[0], values, atol=0.1)


class TestQKClip:

    def test_scales_only_the_qk_weights(self):
        torch.manual_seed(0)
        model = Tiny()
        optimizer = frozen_optimizer(model, MuonConfig(qk_clip_tau=100.0))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer.step(max_logits=400.0)

        expected = math.sqrt(100.0 / 400.0)
        assert torch.allclose(model.q_proj.weight, before['q_proj.weight'] * expected)
        assert torch.allclose(model.k_proj.weight, before['k_proj.weight'] * expected)
        assert torch.allclose(model.v_proj.weight, before['v_proj.weight'])
        assert torch.allclose(model.mlp.weight, before['mlp.weight'])

    def test_no_clip_below_tau(self):
        torch.manual_seed(0)
        model = Tiny()
        optimizer = frozen_optimizer(model, MuonConfig(qk_clip_tau=100.0))
        before = model.q_proj.weight.detach().clone()
        optimizer.step(max_logits=50.0)
        assert torch.allclose(model.q_proj.weight, before)

    def test_no_clip_without_a_measurement(self):
        torch.manual_seed(0)
        model = Tiny()
        optimizer = frozen_optimizer(model, MuonConfig(qk_clip_tau=100.0))
        MaxLogitsTracker.consume()
        before = model.q_proj.weight.detach().clone()
        optimizer.step()
        assert torch.allclose(model.q_proj.weight, before)

    def test_disabled_clip_leaves_attention_untouched(self, monkeypatch):
        """`install` patches globally and permanently, so assert on the decision, not on the patch state."""
        installs = []
        monkeypatch.setattr(MaxLogitsTracker, 'install', classmethod(lambda cls: installs.append(1)))
        model = Tiny()
        MuonClip(create_muon_param_groups(param_groups(model), MuonConfig(qk_clip_enabled=False)), lr=1e-3)
        assert installs == []
        MuonClip(create_muon_param_groups(param_groups(model), MuonConfig(qk_clip_enabled=True)), lr=1e-3)
        assert installs == [1]


class TestMaxLogitsTracker:

    def test_records_the_eager_softmax_input(self):
        MaxLogitsTracker.install()
        model = Tiny()
        ids = torch.randint(0, 32, (2, 4))
        MaxLogitsTracker.consume()
        model(ids)
        recorded = MaxLogitsTracker.consume()
        assert recorded is not None
        assert torch.isfinite(recorded)

    def test_records_an_upper_bound_for_sdpa(self):
        MaxLogitsTracker.install()
        model = Tiny()
        ids = torch.randint(0, 32, (2, 4))
        MaxLogitsTracker.consume()
        model(ids, use_sdpa=True)
        bound = MaxLogitsTracker.consume()
        assert bound is not None and torch.isfinite(bound)

    def test_consume_resets(self):
        MaxLogitsTracker.install()
        model = Tiny()
        model(torch.randint(0, 32, (2, 4)))
        assert MaxLogitsTracker.consume() is not None
        assert MaxLogitsTracker.consume() is None

    def test_keeps_the_maximum_not_the_last(self):
        MaxLogitsTracker.consume()
        MaxLogitsTracker._update(torch.tensor(3.0))
        MaxLogitsTracker._update(torch.tensor(9.0))
        MaxLogitsTracker._update(torch.tensor(1.0))
        assert float(MaxLogitsTracker.consume()) == 9.0


class TestTraining:

    def test_loss_decreases(self):
        torch.manual_seed(0)
        model = Tiny()
        groups = create_muon_param_groups(param_groups(model), MuonConfig(qk_clip_enabled=False))
        optimizer = MuonClip(groups, lr=1e-2, weight_decay=0.0)
        ids = torch.randint(0, 32, (2, 4))
        target = torch.randint(0, 32, (2, 4))
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(ids).view(-1, 32), target.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0]
        assert all(torch.isfinite(p).all() for p in model.parameters())

    def test_muon_and_plain_groups_take_different_updates(self):
        """A 2-D Muon weight moves by the orthogonalised buffer; everything else takes an AdamW step."""
        torch.manual_seed(0)
        model = Tiny()
        groups = create_muon_param_groups(param_groups(model), MuonConfig(qk_clip_enabled=False))
        optimizer = MuonClip(groups, lr=1e-2, weight_decay=0.0)
        nn.functional.cross_entropy(
            model(torch.randint(0, 32, (2, 4))).view(-1, 32), torch.randint(0, 32, (8, ))).backward()
        bias_grad = model.q_proj.bias.grad.detach().clone()
        bias_before = model.q_proj.bias.detach().clone()
        weight_grad = model.mlp.weight.grad.detach().clone()
        weight_before = model.mlp.weight.detach().clone()
        # The group carries its own lr, which takes precedence over the constructor's.
        lr = next(g['lr'] for g in optimizer.param_groups if 'q_proj.bias' in g['param_names'])
        optimizer.step()

        # AdamW's first step is lr * grad / (|grad| + eps) once both moments are bias-corrected, which is
        # lr * sign(grad) up to eps -- the size of the step carries no information about the gradient's
        # magnitude, which is the whole point of the adaptive denominator.
        bias_step = (bias_before - model.q_proj.bias.detach()) / lr
        assert torch.allclose(bias_step, bias_grad.sign(), atol=1e-4)
        # The Muon group instead moves by the orthogonalised buffer, which is not a multiple of the
        # gradient -- and is scaled up by sqrt(max(rows, cols)) * rms_scale_factor.
        weight_step = (weight_before - model.mlp.weight) / lr
        assert not torch.allclose(weight_step, weight_grad, atol=1e-3)
        expected_norm = math.sqrt(max(weight_before.shape)) * 0.2
        assert torch.linalg.svdvals(weight_step.float()).max() < expected_norm * 1.5

    def test_state_carries_only_what_each_rule_needs(self):
        """The AdamW groups must not also pay for a Muon momentum buffer, nor the reverse."""
        torch.manual_seed(0)
        model = Tiny()
        groups = create_muon_param_groups(param_groups(model), MuonConfig(qk_clip_enabled=False))
        optimizer = MuonClip(groups, lr=1e-2, weight_decay=0.0)
        nn.functional.cross_entropy(
            model(torch.randint(0, 32, (2, 4))).view(-1, 32), torch.randint(0, 32, (8, ))).backward()
        optimizer.step()

        by_param = {id(p): name for name, p in model.named_parameters()}
        seen = {}
        for group in optimizer.param_groups:
            for param in group['params']:
                seen[by_param[id(param)]] = (group['apply_muon'], set(optimizer.state[param]))
        muon_keys = {'step', 'momentum_buffer'}
        adamw_keys = {'step', 'exp_avg', 'exp_avg_sq'}
        assert seen['mlp.weight'] == (True, muon_keys)
        assert seen['q_proj.weight'] == (True, muon_keys)
        assert seen['embed_tokens.weight'] == (False, adamw_keys)
        assert seen['q_proj.bias'] == (False, adamw_keys)
        assert seen['norm.weight'] == (False, adamw_keys)

    def test_adamw_groups_match_torch_adamw(self):
        """The non-Muon path must be AdamW proper, so check it against `torch.optim.AdamW` step for step."""
        torch.manual_seed(0)
        reference = Tiny()
        ours = Tiny()
        ours.load_state_dict(reference.state_dict())
        # Only the parameters Muon does not apply to, so both optimizers govern the same update rule.
        names = ['q_proj.bias', 'norm.weight', 'norm.bias', 'embed_tokens.weight']
        ref_params = [dict(reference.named_parameters())[n] for n in names]
        our_params = [dict(ours.named_parameters())[n] for n in names]

        ref_opt = torch.optim.AdamW(ref_params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
        our_opt = MuonClip([{
            'params': our_params,
            'param_names': names,
            'lr': 1e-2,
            'weight_decay': 0.1,
            'apply_muon': False,
            'is_qk': False,
        }],
                           qk_clip_enabled=False)

        ids = torch.randint(0, 32, (2, 4))
        target = torch.randint(0, 32, (2, 4))
        for _ in range(5):
            for model, optimizer in ((reference, ref_opt), (ours, our_opt)):
                optimizer.zero_grad()
                nn.functional.cross_entropy(model(ids).view(-1, 32), target.view(-1)).backward()
                optimizer.step()
        for name in names:
            assert torch.allclose(
                dict(ours.named_parameters())[name], dict(reference.named_parameters())[name],
                atol=1e-6), f'{name} diverged from torch.optim.AdamW'

    def test_resolves_by_name(self):
        import torch.optim

        import twinkle.module.optimizer as optimizer_module
        from twinkle.utils import construct_class
        model = Tiny()
        groups = create_muon_param_groups(param_groups(model), MuonConfig())
        optimizer = construct_class(
            'MuonClip', torch.optim.Optimizer, [torch.optim, optimizer_module], params=groups, lr=1e-3)
        assert isinstance(optimizer, MuonClip)
