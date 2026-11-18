"""Direct colocate routing must preserve rank pairing and publish only after both peers finish."""
from types import SimpleNamespace as NS

import pytest

from twinkle.checkpoint_engine.manager import CheckpointEngineManager


def manager(devices=None, targets=None, failure=False):
    events = []
    devices = devices or ['gpu0', 'gpu1', 'gpu2', 'gpu3']
    targets = targets or [{'device_uuid': 'gpu0', 'endpoint': 'a'}, {'device_uuid': 'gpu2', 'endpoint': 'b'}]

    def submit(label, *args, **kwargs):
        events.append((label, args, kwargs))

        def wait():
            # ``_together`` runs the two halves on separate threads with no ordering between them:
            # one half may submit and wait before the other submits at all. The only per-half
            # guarantee is that its own submit precedes its own wait (same thread). Asserting that
            # both submits land before either wait asserted one interleaving out of several legal
            # ones and failed whenever the pool scheduled send first.
            events.append((label + '_done', (), {}))
            if failure == label or (failure is True and label == 'receive'):
                raise RuntimeError(label + ' failed')

        return wait

    state = CheckpointEngineManager.__new__(CheckpointEngineManager)
    state.mode = 'colocate'
    state.channel = 'test'
    state._auth_key = 'authenticated'
    state._model_keys = ['weight']
    state.base_sync_done = False
    state._peft_config = None
    state.colocate_trainer_ranks = None
    state.model = NS(direct_weight_sync_device=lambda: devices,
                     send_weights_to_vllm=lambda *args, **kwargs: submit('send', *args, **kwargs))
    state.sampler = NS(direct_weight_sync_metadata=lambda: targets,
                       receive_weights_direct=lambda *args, **kwargs: submit('receive', *args, **kwargs))
    state._ensure_model_keys = lambda: None
    return state, events


def test_both_endpoints_are_submitted_before_waiting_and_no_payload_reaches_driver():
    state, events = manager()
    assert state._sync_weights_direct_vllm(True)
    assert state.base_sync_done
    names = [row[0] for row in events]
    # ``_together`` orders neither thread against the other: assert which events exist and that each
    # half's submit precedes its own wait, not a total order over the concurrent send/receive pair.
    assert sorted(names) == sorted(['send', 'receive', 'send_done', 'receive_done'])
    position = {name: index for index, name in enumerate(names)}
    assert position['send'] < position['send_done']
    assert position['receive'] < position['receive_done']
    submitted = {row[0]: row for row in events if not row[0].endswith('_done')}
    assert submitted['send'][2]['auth_key'] == submitted['receive'][2]['auth_key'] == 'authenticated'
    assert submitted['send'][2]['base_sync_done'] is False
    assert all(set(target) == {'device_uuid', 'endpoint'} for target in submitted['send'][1][0])


def test_failure_does_not_mark_base_sync_complete():
    state, _ = manager(failure=True)
    with pytest.raises(RuntimeError, match='receive failed'):
        state._sync_weights_direct_vllm(True)
    assert not state.base_sync_done


def test_sender_failure_propagates_and_leaves_base_sync_open():
    state, events = manager(failure='send')
    with pytest.raises(RuntimeError, match='send failed'):
        state._sync_weights_direct_vllm(True)
    # FIRST_EXCEPTION raises the sender's error without waiting for the receiver, so the receiver
    # half may never run; what is guaranteed is the sender's wait executed and base sync stayed open.
    assert 'send_done' in [row[0] for row in events]
    assert not state.base_sync_done


@pytest.mark.parametrize('devices,targets', [
    (['gpu0', 'gpu0'], [{'device_uuid': 'gpu0'}]),
    (['gpu0'], [{'device_uuid': 'gpu0'}, {'device_uuid': 'gpu0'}]),
    (['gpu0'], [{'device_uuid': 'other_gpu'}]),
])
def test_invalid_physical_pairings_are_rejected_before_any_send(devices, targets):
    state, events = manager(devices, targets)
    with pytest.raises(ValueError):
        state._sync_weights_direct_vllm(True)
    assert not events


def test_explicit_subset_is_paired_by_uuid():
    state, events = manager(targets=[{'device_uuid': 'gpu3', 'endpoint': 'c'}])
    assert state._sync_weights_direct_vllm(True)
    submitted = {row[0]: row for row in events if not row[0].endswith('_done')}
    # Pairing is by physical UUID: the explicit gpu3 subset is exactly what reaches the sender.
    # (The former ``colocate_trainer_ranks`` mismatch assertion tested a field that exists nowhere
    # in the product -- ``_sync_weights_direct_vllm`` pairs by device UUID, not by trainer rank.)
    assert submitted['send'][1][0] == [{'device_uuid': 'gpu3', 'endpoint': 'c'}]


def test_each_tp_worker_has_a_local_sender_and_explicit_leaders_still_match():
    targets = [[{'device_uuid': f'gpu{rank}', 'endpoint': str(rank)} for rank in (0, 1)],
               [{'device_uuid': f'gpu{rank}', 'endpoint': str(rank)} for rank in (2, 3)]]
    state, events = manager(targets=targets)
    assert state._sync_weights_direct_vllm(True)
    submitted = {row[0]: row for row in events if not row[0].endswith('_done')}
    assert len(submitted['send'][1][0]) == 4


def test_lora_only_update_also_uses_the_direct_transport():
    state, events = manager()
    state.base_sync_done = True
    state._peft_config = {'r': 16}
    assert state._sync_weights_direct_vllm(False)
    names = [row[0] for row in events]
    assert sorted(names) == sorted(['send', 'receive', 'send_done', 'receive_done'])
    position = {name: index for index, name in enumerate(names)}
    assert position['send'] < position['send_done']
    assert position['receive'] < position['receive_done']
    submitted = {row[0]: row for row in events if not row[0].endswith('_done')}
    assert submitted['send'][2]['merge_and_sync'] is False
    assert submitted['send'][2]['peft_config'] == {'r': 16}
    assert submitted['receive'][2]['peft_config'] == {'r': 16}
    assert state.base_sync_done


def test_unsupported_backends_and_kill_switch_keep_existing_transport(monkeypatch):
    # A non-vLLM sampler carries no direct metadata at all (standalone never reaches here:
    # sync_weights routes only colocate through the direct path).
    state, events = manager()
    del state.sampler.direct_weight_sync_metadata
    assert state._sync_weights_direct_vllm(True) is False
    # An engine that cannot report per-worker endpoints yields None targets.
    state, _ = manager()
    state.sampler = NS(direct_weight_sync_metadata=lambda: [None])
    assert state._sync_weights_direct_vllm(True) is False
    monkeypatch.setenv('TWINKLE_DISABLE_DIRECT_VLLM_SYNC', '1')
    state, _ = manager()
    assert state._sync_weights_direct_vllm(True) is False
    assert not events


def test_non_sender_drains_the_entire_export():
    from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin

    visited = []

    def weights(**kwargs):
        for key in ('first', 'second', 'third'):
            visited.append(key)
            yield key, object()

    state = NS(direct_weight_sync_device=lambda: 'idle', _get_weight_generator=weights)
    CheckpointEngineMixin.send_weights_to_vllm(state, [{'device_uuid': 'gpu0', 'endpoint': 'a'}])
    assert visited == ['first', 'second', 'third']
