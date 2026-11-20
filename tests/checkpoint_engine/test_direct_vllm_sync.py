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


def test_each_tp_worker_has_a_local_sender():
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


def test_sorted_npu_tp_masks_pair_every_worker_by_physical_uuid():
    # Trainer rank order can differ from the sorted device masks of its TP replicas.
    devices = ['npu6', 'npu13', 'npu8', 'npu9', 'npu10', 'npu11', 'npu7', 'npu12']
    targets = [[{'device_uuid': f'npu{rank}', 'endpoint': str(rank)} for rank in group]
               for group in ((6, 8, 9, 13), (7, 10, 11, 12))]
    state, events = manager(devices=devices, targets=targets)
    assert state._sync_weights_direct_vllm(True)
    sent = next(row for row in events if row[0] == 'send')[1][0]
    assert {target['device_uuid'] for target in sent} == set(devices)
    assert len(sent) == 8


@pytest.mark.parametrize('direct', [False, True])
def test_colocate_releases_trainer_cache_before_waking_sampler(direct):
    """A sampler process cannot allocate from blocks still cached by the trainer."""
    state, _ = manager()
    events = []
    memory = {'free': 1, 'cached': 7}
    state.sleep_level = 1
    state.offload_kwargs = {'offload_model': True, 'offload_optimizer': True}

    def empty_cache():
        events.append('release_submitted')

        def collect():
            memory['free'] += memory.pop('cached', 0)
            events.append('release_completed')

        return collect  # The manager must wait for every rank's lazy call.

    def wake_up(*, tags):
        if tags == ['weights']:
            if memory['free'] < 4:
                raise RuntimeError('sampler OOM on wake_up')
            memory['free'] -= 4
        events.append(tuple(tags))

    state.model.empty_cache = empty_cache
    state.model.offload_to_cpu = lambda **kwargs: events.append('offload')
    state.sampler.wake_up = wake_up

    def sync(merge):
        events.append('direct' if direct else 'fallback')
        return direct

    state._sync_weights_direct_vllm = sync
    state._sync_channel = lambda merge: events.append('channel')
    state._sync_colocate(True)
    assert events == [
        'release_submitted', 'release_completed', ('weights',),
        'direct' if direct else 'fallback', *([] if direct else ['channel']),
        'offload', ('kv_cache',)]


# ── defect118: warm() colocated replicas must use IPC, not NCCL ──────────────
#
# PLAN (case | input | expected | failure it kills):
#  colocated TP2 lend | replicas=[[A,B]] dev=[A,B] | True; IPC, no NCCL
#    | defect118 NCCL "Duplicate GPU detected" actor death
#  flex TP2 growth    | replicas=[[C,D]] dev=[A,B] | False; NCCL fallback
#    | regression: IPC for flex sends nothing (no matching trainer device)
#  colocated TP1      | replicas=[[A]] dev=[A,B]   | True | TP1 lend missed
#  duplicate recv uuid| replicas=[[A,A]] dev=[A,B] | False | invalid dup-GPU topology
#  empty replicas     | replicas=[] dev=[A,B]      | False | nothing to warm
#  None target        | replicas=[None] dev=[A,B]  | False | metadata not ready
#  empty devices      | replicas=[[A]] dev=[]      | False | CPU model fallback
#  non-list replicas  | replicas=None dev=[A,B]    | False | non-vLLM sampler
#  LoRA warm two-step | colocated merge=False base_done=True | IPC base+adapter
#    | LoRA adapter never synced to new replicas
#  merged warm one-step | colocated merge=True     | IPC merged weights
#    | spurious adapter transfer on full-param
#  non-vLLM sub       | sub lacks metadata attr    | False -> NCCL
#    | regression on non-vLLM samplers


def warm_state(devices=None, replicas=None, merge=True, base_done=False):
    import threading

    devices = devices if devices is not None else ['gpu0', 'gpu1']
    replicas = replicas if replicas is not None else [[{'device_uuid': 'gpu0', 'endpoint': 'a'},
                                                       {'device_uuid': 'gpu1', 'endpoint': 'b'}]]
    state, events = manager(devices=devices, targets=replicas)
    state.mode = 'standalone'
    state.backend = 'nccl'
    state._sync_lock = threading.RLock()
    state.sync_backend = 'full'
    state._merge_and_sync = merge
    state.base_sync_done = base_done
    state._peft_config = {'r': 16}
    state._adapter_config = lambda b, m: state._peft_config
    nccl = []
    state._activate_checkpoint_channel = lambda *a, **k: nccl.append('activate') or []
    state._transfer = lambda *a, **k: nccl.append('transfer')
    state._close_checkpoint_channel = lambda *a, **k: nccl.append('close')
    return state, events, nccl


@pytest.mark.parametrize('replicas,devices,expected', [
    ([[{'device_uuid': 'gpu0'}, {'device_uuid': 'gpu1'}]], ['gpu0', 'gpu1'], True),
    ([[{'device_uuid': 'gpu2'}, {'device_uuid': 'gpu3'}]], ['gpu0', 'gpu1'], False),
    ([[{'device_uuid': 'gpu0'}]], ['gpu0', 'gpu1'], True),
    ([[{'device_uuid': 'gpu0'}, {'device_uuid': 'gpu0'}]], ['gpu0', 'gpu1'], False),
    ([], ['gpu0', 'gpu1'], False),
    ([[]], ['gpu0', 'gpu1'], False),
    ([None], ['gpu0', 'gpu1'], False),
    ([[{'device_uuid': 'gpu0'}]], [], False),
    (None, ['gpu0', 'gpu1'], False),
])
def test_is_colocated_routing(replicas, devices, expected):
    assert CheckpointEngineManager._is_colocated(replicas, devices) is expected


def test_warm_colocated_uses_ipc_and_skips_nccl():
    state, events, nccl = warm_state()
    assert state._warm_colocated_ipc(state.sampler, 'warm-chan') is True
    names = [row[0] for row in events]
    assert sorted(names) == sorted(['send', 'receive', 'send_done', 'receive_done'])
    assert not nccl  # Direct warmup does not open a checkpoint-engine channel.
    send = next(row for row in events if row[0] == 'send')
    assert send[2]['channel'] == 'warm-chan'
    assert state.base_sync_done is False


def test_warm_flex_falls_through_to_nccl():
    state, events, nccl = warm_state(replicas=[[{'device_uuid': 'gpu2', 'endpoint': 'c'},
                                                {'device_uuid': 'gpu3', 'endpoint': 'd'}]])
    assert state._warm_colocated_ipc(state.sampler, 'warm-chan') is False
    assert not events and not nccl
    state.warm(state.sampler)
    assert not events
    assert nccl == ['activate', 'transfer', 'close']


def test_warm_colocated_lora_syncs_base_then_adapter():
    state, events, nccl = warm_state(merge=False, base_done=True)
    assert state._warm_colocated_ipc(state.sampler, 'warm-chan') is True
    sends = [row for row in events if row[0] == 'send']
    assert len(sends) == 2
    assert sends[0][2]['base_sync_done'] is False
    assert sends[1][2]['base_sync_done'] is True
    assert not nccl
    assert state.base_sync_done is True


def test_warm_colocated_merged_weights_single_transfer():
    state, events, nccl = warm_state(merge=True, base_done=True)
    assert state._warm_colocated_ipc(state.sampler, 'warm-chan') is True
    sends = [row for row in events if row[0] == 'send']
    assert len(sends) == 1
    assert sends[0][2]['base_sync_done'] is True
    assert sends[0][2]['merge_and_sync'] is True
    assert state.base_sync_done is True


def test_warm_non_vllm_sub_falls_through():
    state, events, nccl = warm_state()
    del state.sampler.direct_weight_sync_metadata
    assert state._warm_colocated_ipc(state.sampler, 'warm-chan') is False
    assert not events and not nccl
    state.warm(state.sampler)
    assert not events
    assert nccl == ['activate', 'transfer', 'close']
