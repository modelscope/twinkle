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
            assert [row[0] for row in events][:2] == ['send', 'receive']
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
    state.colocate_trainer_ranks = None
    state.model = NS(direct_weight_sync_device=lambda: devices,
                     send_weights_to_vllm=lambda *args, **kwargs: submit('send', *args, **kwargs))
    state.sampler = NS(direct_weight_sync_metadata=lambda: targets)
    state._ensure_model_keys = lambda: None
    state._sampler_call = lambda method, *args, **kwargs: (
        targets if method == 'direct_weight_sync_metadata' else submit('receive', *args, **kwargs))
    return state, events


def test_both_endpoints_are_submitted_before_waiting_and_no_payload_reaches_driver():
    state, events = manager()
    assert state._sync_weights_direct_vllm(True)
    assert state.base_sync_done
    assert [row[0] for row in events] == ['send', 'receive', 'send_done', 'receive_done']
    assert events[0][2]['auth_key'] == events[1][2]['auth_key'] == 'authenticated'
    assert events[0][2]['base_sync_done'] is False
    assert all(set(target) == {'device_uuid', 'endpoint'} for target in events[0][1][0])


def test_failure_does_not_mark_base_sync_complete():
    state, _ = manager(failure=True)
    with pytest.raises(RuntimeError, match='receive failed'):
        state._sync_weights_direct_vllm(True)
    assert not state.base_sync_done


def test_sender_failure_still_collects_receiver_and_releases_its_lock():
    state, events = manager(failure='send')
    with pytest.raises(RuntimeError, match='send failed'):
        state._sync_weights_direct_vllm(True)
    assert [row[0] for row in events][-2:] == ['send_done', 'receive_done']
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
    state, _ = manager(targets=[{'device_uuid': 'gpu3', 'endpoint': 'c'}])
    state.colocate_trainer_ranks = (3,)
    assert state._sync_weights_direct_vllm(True)
    state.colocate_trainer_ranks = (0,)
    with pytest.raises(ValueError, match='colocate_trainer_ranks'):
        state._sync_weights_direct_vllm(True)


def test_each_tp_worker_has_a_local_sender_and_explicit_leaders_still_match():
    targets = [[{'device_uuid': f'gpu{rank}', 'endpoint': str(rank)} for rank in (0, 1)],
               [{'device_uuid': f'gpu{rank}', 'endpoint': str(rank)} for rank in (2, 3)]]
    state, events = manager(targets=targets)
    state.colocate_trainer_ranks = (0, 2)
    assert state._sync_weights_direct_vllm(True)
    assert len(events[0][1][0]) == 4


def test_non_cuda_lora_and_standalone_keep_existing_transport(monkeypatch):
    state, events = manager()
    assert state._sync_weights_direct_vllm(False) is False
    state.mode = 'standalone'
    assert state._sync_weights_direct_vllm(True) is False
    state.mode = 'colocate'
    state._sampler_call = lambda *args, **kwargs: [None]
    assert state._sync_weights_direct_vllm(True) is False
    monkeypatch.setenv('TWINKLE_DISABLE_DIRECT_VLLM_SYNC', '1')
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
