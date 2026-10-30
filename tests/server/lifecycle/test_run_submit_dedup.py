# Copyright (c) ModelScope Contributors. All rights reserved.
"""run_submit seq_id dedup: the release-on-failure decision must be driven by whether
a future record exists (i.e. whether the task was enqueued), NOT by exception type.

The load-bearing case: if submit_and_peek raises *after* the task was enqueued (e.g. a
transient state error inside the inline peek), the seq claim must be KEPT -- releasing
it would let a retry enqueue a duplicate, the exact double-apply the dedup prevents.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from twinkle.server.lifecycle.submit import run_submit
from twinkle.protocol.types.model import ForwardBackwardTaskRequest


class _FakeState:
    def __init__(self, record_after_claim):
        self._record_after_claim = record_after_claim
        self.claimed = {}
        self.claim_ttls = []
        self.released = []

    async def claim_training_seq(self, training_id, stream, seq_id, request_id, retention):
        key = (training_id, stream, seq_id)
        prior = self.claimed.get(key)
        self.claimed[key] = prior or request_id
        self.claim_ttls.append(retention)
        return prior

    async def get_future(self, request_id):
        # Simulates whether a PENDING/QUEUED record was written (task enqueued).
        return self._record_after_claim

    async def release_training_seq(self, training_id, stream, seq_id, request_id):
        self.released.append((training_id, stream, seq_id))


class _FakeManagement:
    def __init__(self, record_after_claim):
        self.state = _FakeState(record_after_claim)
        self.sequence_retention_s = 3600
        # A real deployment declares its backend; preflight reads it from here.
        self.backend = 'transformers'
        self.data_world_size = 1

    async def _on_request_start(self, request):
        return 'token'

    async def submit_and_peek(self, *args, **kwargs):
        # Fail *after* the (simulated) enqueue -- e.g. a state blip during the peek.
        raise RuntimeError('transient state error during peek')


def _request():
    return SimpleNamespace(state=SimpleNamespace(session_id='sess-1', request_id='rq-1'))


def _body(adapter_name: str = 'ad', seq_id: int = 7) -> ForwardBackwardTaskRequest:
    """A real request model, not a stand-in.

    ``run_submit`` now reads field roles off the body to build the backend kwargs and to
    run preflight, so a ``SimpleNamespace`` would exercise a shape production never sees.
    """
    return ForwardBackwardTaskRequest(inputs=[{'input_ids': [1, 2]}], adapter_name=adapter_name, seq_id=seq_id, training_id='training-' + adapter_name)


async def _call(self, body, adapter_name, token):  # pragma: no cover - never invoked
    return {'ok': True}


@pytest.mark.asyncio
async def test_release_kept_when_task_already_enqueued():
    # A record exists (task enqueued) -> peek error must NOT release the claim.
    mgmt = _FakeManagement(record_after_claim={'status': 'queued'})
    with pytest.raises(RuntimeError):
        await run_submit(mgmt, _request(), _body(), task_type='forward_backward', backend_call=_call)
    assert mgmt.state.released == [], 'claim wrongly released for an already-enqueued task'
    assert ('training-ad', 'native::sess-1::forward_backward', 7) in mgmt.state.claimed


@pytest.mark.asyncio
async def test_release_when_never_enqueued():
    # No record (e.g. preflight rejected before any write) -> release so a retry can re-enqueue.
    mgmt = _FakeManagement(record_after_claim=None)
    with pytest.raises(RuntimeError):
        await run_submit(mgmt, _request(), _body(), task_type='forward_backward', backend_call=_call)
    assert mgmt.state.released == [('training-ad', 'native::sess-1::forward_backward', 7)], 'claim should be released when nothing enqueued'


@pytest.mark.asyncio
async def test_dedup_uses_training_sequence_retention():
    """The claim must outlive queue wait plus execution, not just execution."""
    mgmt = _FakeManagement(record_after_claim={'status': 'queued'})
    with pytest.raises(RuntimeError):
        await run_submit(mgmt, _request(), _body(), task_type='forward_backward', backend_call=_call)
    assert mgmt.state.claim_ttls == [mgmt.sequence_retention_s]


@pytest.mark.asyncio
async def test_dedup_key_is_scoped_per_adapter():
    """Two adapters in ONE session must not collide on the same seq_id.

    Every client model object owns its own seq counter starting at 1 while ``session_id``
    is process-global, so multi-LoRA training from one process issues seq_id=1 twice. If
    the adapter were missing from the key, the second adapter's forward_backward would be
    swallowed as a duplicate and handed the first adapter's loss -- a silent wrong result.
    """
    mgmt = _FakeManagement(record_after_claim={'status': 'queued'})
    for adapter in ('lora-A', 'lora-B'):
        with pytest.raises(RuntimeError):
            await run_submit(
                mgmt,
                _request(),
                _body(adapter_name=adapter, seq_id=1),
                task_type='forward_backward',
                backend_call=_call)

    claimed = set(mgmt.state.claimed)
    assert claimed == {('training-lora-A', 'native::sess-1::forward_backward', 1), ('training-lora-B', 'native::sess-1::forward_backward', 1)}, (
        f'adapters collided on one dedup key: {claimed}')
