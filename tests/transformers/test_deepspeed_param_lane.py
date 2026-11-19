# Copyright (c) ModelScope Contributors. All rights reserved.
"""The step-delta probe's param lane must be labeled by what the config actually does.

The bundled ZeRO presets keep ``offload_param`` / ``offload_optimizer`` present with
``device: 'none'`` when offload is off. A truthiness check on those keys labels every plain
zero0-zero3 run ``stale_view`` and sends the Δθ judge to the master lane even where the param
lane is authoritative — measured live: zero2 and zero3 probe runs both reported ``stale_view``.
Only ``device == 'cpu'`` moves buffers to the CPU (the same reading swift's
``deepspeed_cpu_offload`` uses); stage 3 without offload partitions the param digest instead.
"""
from types import SimpleNamespace

import pytest

from twinkle.model.transformers.strategy.base import TransformerStrategyBase
from twinkle.model.transformers.strategy.deepspeed import DeepSpeedStrategy

_NONE = {'device': 'none', 'pin_memory': True}
_CPU = {'device': 'cpu', 'pin_memory': True}


def _lane(zero, *, as_object=False):
    cfg = {} if zero is None else {'zero_optimization': zero}
    if as_object:
        cfg = SimpleNamespace(config=cfg)
    plugin = SimpleNamespace(deepspeed_config=cfg)
    return DeepSpeedStrategy.param_lane_status(SimpleNamespace(_deepspeed_plugin=plugin))


@pytest.mark.parametrize(
    'zero, expected',
    [
        # real preset shapes: plain tiers carry device-'none' offload dicts
        ({'stage': 2, 'offload_optimizer': _NONE}, 'authoritative'),
        ({'stage': 3, 'offload_param': _NONE, 'offload_optimizer': _NONE}, 'partitioned'),
        ({'stage': 2, 'offload_optimizer': _CPU}, 'stale_view'),
        ({'stage': 3, 'offload_param': _CPU, 'offload_optimizer': _CPU}, 'stale_view'),
        ({'stage': 3, 'offload_param': _CPU, 'offload_optimizer': _NONE}, 'stale_view'),
        ({'stage': 2}, 'authoritative'),
        ({'stage': 0}, 'authoritative'),
        ({'stage': 1}, 'authoritative'),
        ({'stage': 2, 'offload_optimizer': None}, 'authoritative'),
        ({'stage': 3, 'offload_param': None, 'offload_optimizer': None}, 'partitioned'),
        (None, 'authoritative'),
    ],
    ids=[
        'zero2_preset', 'zero3_preset', 'zero2_offload_preset', 'zero3_offload_preset',
        'zero3_param_only_offload', 'zero2_no_offload_keys', 'zero0', 'zero1',
        'offload_none_explicit', 'zero3_offload_none_explicit', 'no_zero_optimization',
    ])
def test_param_lane_status(zero, expected):
    assert _lane(zero) == expected


def test_param_lane_status_reads_the_config_object_carrier():
    zero = {'stage': 3, 'offload_param': _NONE, 'offload_optimizer': _NONE}
    assert _lane(zero, as_object=True) == 'partitioned'


def test_base_strategy_param_lane_is_authoritative():
    assert TransformerStrategyBase.param_lane_status(SimpleNamespace()) == 'authoritative'