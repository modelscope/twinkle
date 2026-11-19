# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect93: TWINKLE_CHECKPOINT_BUCKET_MB knob for the full-sync transport bucket.

The NCCL/HCCL/XCCL engines assert a single weight fits one bucket; the 3 GB class default is smaller
than a 9B fp32 embedding (~4 GB), and the assertion tells the user to "Increase bucket_size" with no
way to do so. ``CheckpointEngineMixin._resolve_bucket_size`` reads the env knob; the IPC branch keeps
its deliberately smaller 512 MB default unless the class default was overridden or the knob is set.

PLAN (case | input | expected | failure it kills):
  resolve_unset       | env absent              | 3072<<20 (class default)      | knob ignored / wrong default
  resolve_empty       | env=""                  | 3072<<20                      | empty string parsed as 0 / crash
  resolve_valid       | env="4608"              | 4608*1024*1024 (hand oracle)  | MB->bytes shift wrong / off-by-unit
  resolve_zero        | env="0"                 | ValueError                    | non-positive bucket accepted
  resolve_negative    | env="-5"                | ValueError                    | negative bucket accepted
  resolve_garbage     | env="abc"               | ValueError                    | silent fallback on bad input
  nccl_unset          | backend=nccl, env absent| engine.bucket_size=3072<<20   | nccl branch bypasses resolver
  nccl_env            | backend=nccl, env=4608  | engine.bucket_size=4608<<20   | nccl branch ignores knob (the 9B bug)
  ipc_default         | backend=ipc, env absent | engine.bucket_size=512<<20    | IPC loses its small default (None not passed)
  ipc_env             | backend=ipc, env=4608   | engine.bucket_size=4608<<20   | IPC override gate ignores env
  ipc_subclass        | subclass _bucket_size=1024<<20, env absent | 1024<<20   | class-override detection broken

Reverse-verify: reverting the resolver to ``return self._bucket_size`` (knob removed) turns
resolve_valid / nccl_env / ipc_env RED; the *_unset / ipc_default / ipc_subclass cases stay GREEN
(they are the flag-off control proving the knob is load-bearing, not a always-on rewrite).
All engines constructed for real on CPU (their __init__ only assigns attrs; CUDA/shm live in prepare()).
"""

import pytest

from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin

MB = 1024 * 1024
DEFAULT = 3072 << 20
IPC_DEFAULT = 512 << 20
ENV = 'TWINKLE_CHECKPOINT_BUCKET_MB'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def _mixin(backend):
    m = CheckpointEngineMixin()
    m._checkpoint_engine_backend = backend
    m._checkpoint_engine_key = (backend, 'default')
    return m


@pytest.mark.parametrize('value, expected', [
    (None, DEFAULT),
    ('', DEFAULT),
    ('4608', 4608 * MB),
    ('1', MB),
])
def test_resolve_bucket_size(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert CheckpointEngineMixin()._resolve_bucket_size() == expected


@pytest.mark.parametrize('value', ['0', '-5', 'abc'])
def test_resolve_bucket_size_rejects_bad(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError):
        CheckpointEngineMixin()._resolve_bucket_size()


@pytest.mark.parametrize('value, expected', [(None, DEFAULT), ('4608', 4608 * MB)])
def test_nccl_branch_uses_resolver(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    engine = _mixin('nccl')._get_or_create_checkpoint_engine()
    assert engine.bucket_size == expected


@pytest.mark.parametrize('value, expected', [(None, IPC_DEFAULT), ('4608', 4608 * MB)])
def test_ipc_branch_override_gate(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    engine = _mixin('ipc')._get_or_create_checkpoint_engine()
    assert engine.bucket_size == expected


def test_ipc_branch_honors_subclass_override():
    class Sub(CheckpointEngineMixin):
        _bucket_size = 1024 << 20

    m = Sub()
    m._checkpoint_engine_backend = 'ipc'
    m._checkpoint_engine_key = ('ipc', 'default')
    engine = m._get_or_create_checkpoint_engine()
    assert engine.bucket_size == 1024 * MB