# Copyright (c) Alibaba, Inc. and its affiliates.
"""defect84 gate: the local launcher's first process group must honor TWINKLE_DIST_TIMEOUT_SECONDS.

On the local torchrun path ``ensure_default_process_group`` creates the PG BEFORE accelerate, and
accelerate skips an already-initialized PG -- its InitProcessGroupKwargs timeout never applies.
Without the knob the torch default 600s watchdog SIGABRTs ranks whose checkpoint-load I/O skew
exceeds ten minutes on slow storage.

PLAN (case | input | expected | failure it kills):
1 default timeout | env unset, world 2 | init kwargs timeout == 7200s | torch default 600s
  watchdog killing slow-storage rank skew (the defect84 old red)
2 knob override | TWINKLE_DIST_TIMEOUT_SECONDS=1234 | timeout == 1234s | knob ignored /
  hardcoded default
3 already-initialized short-circuit | dist.is_initialized() True | returns False, no init call |
  double-init regression
Reverse-verify: deleting the timeout entry from init_kwargs turns 1/2 RED (missing kwarg),
3 stays GREEN.
"""
from datetime import timedelta
from unittest.mock import patch

from twinkle.utils.torch_utils import ensure_default_process_group


def _run(monkeypatch, env_value):
    import torch.distributed as dist
    if env_value is None:
        monkeypatch.delenv('TWINKLE_DIST_TIMEOUT_SECONDS', raising=False)
    else:
        monkeypatch.setenv('TWINKLE_DIST_TIMEOUT_SECONDS', env_value)
    captured = {}
    monkeypatch.setattr(dist, 'is_initialized', lambda: False)
    monkeypatch.setattr(dist, 'init_process_group', lambda **kw: captured.update(kw))
    from twinkle.utils import torch_utils
    from twinkle.utils.platforms import Platform
    monkeypatch.setattr(Platform, 'get_world_size', lambda: 2)
    monkeypatch.setattr(Platform, 'get_rank', lambda: 0)
    monkeypatch.setattr(Platform, 'device_backend', lambda: 'gloo')
    monkeypatch.setattr(torch_utils.Torch, 'set_device', lambda: None)
    platform = Platform.get_platform()
    monkeypatch.setattr(platform, 'ensures_socket_env', lambda: False)
    monkeypatch.setattr(platform, 'binds_device_id_to_process_group', lambda: False)
    monkeypatch.setattr(platform, 'requires_unbound_default_pg', lambda: False)
    assert ensure_default_process_group(force=True) is True
    return captured


def test_default_timeout_is_7200s(monkeypatch):
    captured = _run(monkeypatch, None)
    assert captured['timeout'] == timedelta(seconds=7200)


def test_knob_override(monkeypatch):
    captured = _run(monkeypatch, '1234')
    assert captured['timeout'] == timedelta(seconds=1234)


def test_already_initialized_short_circuits(monkeypatch):
    import torch.distributed as dist
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    with patch.object(dist, 'init_process_group', side_effect=AssertionError('must not re-init')):
        assert ensure_default_process_group(force=True) is False
