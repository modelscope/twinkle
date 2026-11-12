# Copyright (c) ModelScope Contributors. All rights reserved.
"""Both backend entry points share one full-batch preflight implementation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from twinkle.loss.grpo import GRPOLoss
from twinkle.server.model.backends import common
from twinkle.server.model.backends.common import TwinkleCompatModelBase
from twinkle.server.model.backends.megatron_model import (TwinkleCompatFullMegatronModel, TwinkleCompatMegatronModel)
from twinkle.server.model.backends.transformers_model import (TwinkleCompatFullTransformersModel,
                                                              TwinkleCompatTransformersModel)


class _FakeBackend(TwinkleCompatModelBase):
    """Just the state preflight touches; no Ray actor, no device."""

    def __init__(self, loss=GRPOLoss, with_loss=True):
        self.optimizer_group = {
            'adapter': SimpleNamespace(loss_instance=loss(epsilon=0.2) if with_loss and loss else None),
        }
        self.set_loss_calls = []

    def _get_default_group(self):
        return 'adapter'

    def set_loss(self, loss_cls, **kwargs):
        kwargs.pop('adapter_name', None)
        self.set_loss_calls.append((loss_cls, kwargs))
        self.optimizer_group['adapter'].loss_instance = GRPOLoss(**{
            key: value
            for key, value in kwargs.items() if key in ('epsilon', 'beta')
        })

    def get_template(self, adapter_name):
        return None


_BAD_SAMPLE = {
    'labels': [-100, -100, 0, 0],
}
_BAD_OLD_LOGPS = [[0.1, 0.2, 0.3]]


@pytest.mark.parametrize('backend', [
    TwinkleCompatTransformersModel,
    TwinkleCompatFullTransformersModel,
    TwinkleCompatMegatronModel,
    TwinkleCompatFullMegatronModel,
])
def test_every_backend_inherits_the_preflight_entry_point(backend):
    """A backend without it silently skips the check and can still deadlock."""
    assert hasattr(backend, 'preflight_loss_inputs')


def test_preflight_rejects_a_ragged_batch_on_the_twinkle_route():
    backend = _FakeBackend()
    with pytest.raises(AssertionError, match='mismatch'):
        TwinkleCompatModelBase.preflight_loss_inputs(
            backend,
            inputs=[_BAD_SAMPLE],
            adapter_name='adapter',
            old_logps=_BAD_OLD_LOGPS,
        )
    assert backend.set_loss_calls == []


def test_preflight_accepts_a_response_only_row():
    backend = _FakeBackend()
    TwinkleCompatModelBase.preflight_loss_inputs(
        backend,
        inputs=[_BAD_SAMPLE],
        adapter_name='adapter',
        old_logps=[[0.1, 0.2]],
    )


def test_preflight_tinker_route_sets_up_the_loss_before_validating(monkeypatch):
    """The tinker payload hides its RL features in the datums, so they must be extracted."""
    monkeypatch.setattr(common, 'datum_to_input_feature', lambda inputs, template: [_BAD_SAMPLE])
    monkeypatch.setattr(common, 'extract_rl_features_for_loss', lambda inputs: {'old_logps': _BAD_OLD_LOGPS, 'ref_outputs': {'logps': []}})
    backend = _FakeBackend(with_loss=False)

    with pytest.raises(AssertionError, match='mismatch'):
        TwinkleCompatModelBase.preflight_loss_inputs(
            backend,
            inputs=[SimpleNamespace(loss_fn_inputs={})],
            adapter_name='adapter',
            loss_fn='importance_sampling',
            epsilon=0.3,
        )

    assert backend.set_loss_calls == [('GRPOLoss', {
        'epsilon': 0.3,
        'beta': 0.0
    })]


def test_preflight_is_a_no_op_without_a_loss_instance():
    backend = _FakeBackend(with_loss=False)
    TwinkleCompatModelBase.preflight_loss_inputs(backend, inputs=[_BAD_SAMPLE], adapter_name='adapter', old_logps=_BAD_OLD_LOGPS)
