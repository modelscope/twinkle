# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout side: sampler coordination, GPU-local receivers, topology placement, request admission and device mapping."""
import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
import torch

from twinkle.checkpoint_engine.delta.rollout.cuda_receiver import CudaDeltaReceiver
from twinkle.checkpoint_engine.delta.rollout.mixin import VLLMDeltaSamplerMixin
from twinkle.sampler.base import Sampler
from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine
from twinkle.sampler.vllm_sampler.vllm_worker_extension import _cuda_visible_device


class LocalCudaSampler(VLLMDeltaSamplerMixin):
    def _run_in_loop(self, coroutine):
        return asyncio.run(coroutine)


class GateSampler(Sampler):
    """Concrete sampler exercising the base pause/resume gate (no engine-native pause)."""

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def apply_patch(self, *args, **kwargs):
        raise NotImplementedError


def receiver_stub(flags):
    # 58b2b979: receivers are local CudaDeltaReceiver objects whose next_flush/send_flush are sync
    # callables offloaded to threads (asyncio.to_thread), not Ray actor .remote() handles.
    return SimpleNamespace(
        next_flush=Mock(side_effect=flags),
        send_flush=Mock(),
    )


def test_zero_change_skips_native_update_but_restores_kv_and_generation():
    sampler = LocalCudaSampler()
    sampler._delta_receivers = [receiver_stub([None]), receiver_stub([None])]
    sampler._delta_config = SimpleNamespace(timeout_seconds=1, free_cache_engine=True)
    sampler.pause_generation = Mock()
    sampler.resume_generation = Mock()
    sampler.engine = SimpleNamespace(engine=SimpleNamespace(
        sleep=AsyncMock(), wake_up=AsyncMock(), reset_prefix_cache=AsyncMock(return_value=True)))
    sampler._delta_rpc = AsyncMock()
    sampler.begin_delta_sync()
    assert sampler.receive_delta_weights() == {'changed': False}
    sampler.publish_delta_sampler()
    sampler._delta_rpc.assert_not_called()
    sampler.pause_generation.assert_called_once()
    sampler.engine.engine.sleep.assert_awaited_once_with(level=1)
    assert sampler.engine.engine.wake_up.await_args_list == [
        call(tags=['weights']), call(tags=['kv_cache'])]
    sampler.resume_generation.assert_called_once()


def test_sampler_coordinates_native_update_and_gpu_local_senders():
    sampler = LocalCudaSampler()
    sampler._delta_receivers = [receiver_stub([{'is_last': True}, None]) for _ in range(2)]
    sampler._delta_config = SimpleNamespace(timeout_seconds=1, free_cache_engine=False)
    sampler._delta_weight_transfer_initialized = False
    sampler.engine = SimpleNamespace(pause_delta_generation=AsyncMock(), engine=SimpleNamespace(
        reset_prefix_cache=AsyncMock(return_value=True), reset_mm_cache=AsyncMock(), reset_encoder_cache=AsyncMock()))
    sampler._delta_rpc = AsyncMock()
    assert sampler.receive_delta_weights() == {'changed': True}
    assert [call.args[0] for call in sampler._delta_rpc.await_args_list] == [
        'init_weight_transfer_engine', 'start_weight_update', 'update_twinkle_delta_weights', 'finish_weight_update']
    for receiver in sampler._delta_receivers:
        receiver.send_flush.assert_called_once()
    sampler.engine.engine.reset_prefix_cache.assert_awaited_once_with(reset_connector=True)
    sampler.engine.engine.reset_mm_cache.assert_awaited_once()
    sampler.engine.engine.reset_encoder_cache.assert_awaited_once()


def test_receiver_stream_disagreement_fails_before_native_update():
    sampler = LocalCudaSampler()
    sampler._delta_receivers = [receiver_stub([None]), receiver_stub([{'is_last': True}])]
    sampler._delta_rpc = AsyncMock()
    with pytest.raises(ValueError, match='disagree'):
        sampler.receive_delta_weights()
    sampler._delta_rpc.assert_not_called()


def test_cache_reset_failure_does_not_resume_generation():
    sampler = LocalCudaSampler()
    sampler._delta_receivers = [receiver_stub([{'is_last': True}, None])]
    sampler._delta_weight_transfer_initialized = True
    sampler._delta_rpc = AsyncMock()
    sampler.engine = SimpleNamespace(engine=SimpleNamespace(
        reset_prefix_cache=AsyncMock(return_value=False), resume_generation=AsyncMock()))
    with pytest.raises(RuntimeError, match='prefix cache reset failed'):
        sampler.receive_delta_weights()
    sampler.engine.engine.resume_generation.assert_not_called()


def test_without_kv_release_pause_and_resume_still_run():
    sampler = LocalCudaSampler()
    sampler._delta_config = SimpleNamespace(timeout_seconds=1, free_cache_engine=False)
    sampler.pause_generation = Mock()
    sampler.resume_generation = Mock()
    sampler.engine = SimpleNamespace(engine=SimpleNamespace(
        sleep=AsyncMock(), wake_up=AsyncMock()))
    sampler.begin_delta_sync()
    sampler.publish_delta_sampler()
    sampler.pause_generation.assert_called_once()
    sampler.resume_generation.assert_called_once()
    sampler.engine.engine.sleep.assert_not_called()
    sampler.engine.engine.wake_up.assert_not_called()


def test_checkpoint_receiver_keeps_flush_on_local_gpu_until_ipc(monkeypatch):
    from twinkle.checkpoint_engine.delta.rollout import cuda_receiver as delta_rollout
    engine = Mock()
    tensors = [('__values__', torch.zeros(2))]
    engine.receive_weights.return_value = iter([(tensors, True)])
    monkeypatch.setattr(delta_rollout, 'DeltaShardedCheckpointEngine', Mock(return_value=engine))
    monkeypatch.setattr(torch.cuda, 'set_device', Mock())
    monkeypatch.setattr(torch.cuda, 'get_device_properties', Mock(return_value=SimpleNamespace(uuid='GPU-test')))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    sender = SimpleNamespace(async_send_weights=AsyncMock())
    module = ModuleType('twinkle.sampler.vllm_sampler.delta.cuda_ipc')
    module.BucketedWeightSender = Mock(return_value=sender)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = SimpleNamespace(bucket_bytes=64 << 20)
    receiver = CudaDeltaReceiver(config, {'world_size': 3, 'master_metadata': 'master'},
                                 {'rank': 2, 'device_uuid': 'GPU-test', 'cuda_visible_device': '3',
                                  'zmq_handle': 'ipc:///tmp/rank2.sock'})
    # 58b2b979: the process group is joined lazily in ready() (the rendezvous barrier), not in __init__.
    receiver.ready()
    engine.init_process_group.assert_called_once_with(2, 3, 'master')
    assert receiver.next_flush() == {'is_last': True}
    receiver.send_flush()
    assert list(sender.async_send_weights.call_args.args[0]) == tensors
    assert receiver.next_flush() is None
    assert receiver._current is None


def test_receiver_binds_gpu_before_cuda_initialization(monkeypatch):
    from twinkle.checkpoint_engine.delta.rollout import cuda_receiver as delta_rollout
    import os

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    observed = []
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: observed.append(os.environ['CUDA_VISIBLE_DEVICES']))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', Mock(return_value=SimpleNamespace(uuid='GPU-test')))
    monkeypatch.setattr(delta_rollout, 'DeltaShardedCheckpointEngine', Mock())
    CudaDeltaReceiver(SimpleNamespace(bucket_bytes=64), dict(world_size=2, master_metadata='master'),
                      dict(rank=1, device_uuid='GPU-test', cuda_visible_device='3', zmq_handle='ipc:///test'))
    assert observed == ['3']


def test_receiver_rejects_gpu_mismatch_before_joining_group(monkeypatch):
    from twinkle.checkpoint_engine.delta.rollout import cuda_receiver as delta_rollout

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    monkeypatch.setattr(torch.cuda, 'set_device', Mock())
    monkeypatch.setattr(torch.cuda, 'get_device_properties', Mock(return_value=SimpleNamespace(uuid='GPU-wrong')))
    engine = Mock()
    monkeypatch.setattr(delta_rollout, 'DeltaShardedCheckpointEngine', engine)
    with pytest.raises(ValueError, match='GPU mismatch'):
        CudaDeltaReceiver(SimpleNamespace(bucket_bytes=64), dict(world_size=2, master_metadata='master'),
                          dict(rank=1, device_uuid='GPU-test', cuda_visible_device='3', zmq_handle='ipc:///test'))
    engine.assert_not_called()


@pytest.mark.parametrize('use_worker_selector', [False, True])
def test_topology_places_one_receiver_on_each_vllm_gpu(monkeypatch, use_worker_selector):
    # 58b2b979: receivers are local CudaDeltaReceiver participants bound to a DeviceGroup (node_addresses
    # + visible_devices), not Ray actor handles scheduled by node_id. Assert the placement each receiver
    # is constructed with: one per vLLM GPU, global rank, and the worker selector overriding the UUID.
    from twinkle.checkpoint_engine.delta.rollout import mixin as rollout_mixin
    metadata = [dict(node_ip='10.0.0.1', device_uuid=f'GPU-{rank}', zmq_handle=f'ipc:///tmp/rank{rank}.sock')
                for rank in range(2)]
    if use_worker_selector:
        for rank, worker in enumerate(metadata):
            worker['cuda_visible_device'] = str(rank + 2)
    created = []
    groups = []

    class FakeGroup:
        def __init__(self, **kwargs):
            groups.append(kwargs)

    class FakeReceiver:
        def __init__(self, config, endpoint, placement, remote_group=None, instance_id=None, call_timeout=None):
            created.append((placement['rank'], placement['zmq_handle'], remote_group))

        def ready(self):
            return 0

    monkeypatch.setattr(rollout_mixin, 'DeviceGroup', FakeGroup)
    monkeypatch.setattr(rollout_mixin, 'CudaDeltaReceiver', FakeReceiver)
    sampler = LocalCudaSampler()
    sampler._delta_rpc = AsyncMock(return_value=metadata)
    sampler._delta_config = SimpleNamespace(timeout_seconds=1)
    sampler.init_delta_sampler_transport({'world_size': 3, 'master_metadata': 'master',
                                              'rollout_workers': metadata})
    assert [item[:2] for item in created] == [(1, 'ipc:///tmp/rank0.sock'), (2, 'ipc:///tmp/rank1.sock')]
    expected_selectors = ['2', '3'] if use_worker_selector else ['GPU-0', 'GPU-1']
    assert [g['visible_devices'] for g in groups] == [[s] for s in expected_selectors]
    assert all(g['node_addresses'] == ['10.0.0.1'] for g in groups)
    assert all(g['device_type'] == 'GPU' and g['ranks'] == 1 for g in groups)


def test_replica_receivers_use_global_ranks_across_nodes(monkeypatch):
    workers = [dict(node_ip=f'10.0.0.{rank // 2 + 1}', device_uuid=f'GPU-{rank}',
                    zmq_handle=f'ipc:///rank{rank}') for rank in range(4)]
    endpoint = dict(world_size=5, master_metadata='master', rollout_workers=workers)
    sampler = LocalCudaSampler()
    placements = [sampler._delta_receiver_placements(workers[:2], endpoint),
                  sampler._delta_receiver_placements(workers[2:], endpoint)]
    assert [[item['rank'] for item in replica] for replica in placements] == [[1, 2], [3, 4]]
    # 58b2b979: placements carry node_ip (DeviceGroup node_addresses), not a resolved Ray node_id.
    assert [[item['node_ip'] for item in replica] for replica in placements] == [
        ['10.0.0.1', '10.0.0.1'], ['10.0.0.2', '10.0.0.2']]


def test_sampler_dp_allows_independent_rollout_replicas():
    from twinkle.checkpoint_engine.delta import DeltaSyncConfig
    sampler = LocalCudaSampler()
    sampler.device_mesh = SimpleNamespace(data_world_size=2)
    sampler.engine = SimpleNamespace(enable_sleep_mode=True)
    sampler._delta_rpc = AsyncMock(return_value=[dict(tp=2, device='cuda')] * 2)
    assert sampler.validate_delta_sampler(DeltaSyncConfig()) == dict(tp=2, device='cuda')


def test_pause_waits_for_inflight_request_then_blocks_new_until_resume():
    # The submission barrier now lives in the base _GenerationGate: pause must wait for a request
    # already past the gate (in-flight reader) and block new reads until resume.
    sampler = GateSampler()
    gate = sampler._gate()
    entered, release, read_done = threading.Event(), threading.Event(), threading.Event()

    def reader():
        with gate.read():
            entered.set()
            release.wait()
        read_done.set()

    inflight = threading.Thread(target=reader, daemon=True)
    inflight.start()
    assert entered.wait(2)
    paused = threading.Event()

    def pauser():
        sampler.pause_generation()
        paused.set()

    pausing = threading.Thread(target=pauser, daemon=True)
    pausing.start()
    assert not paused.wait(0.2), 'pause completed while a request still held the gate'
    new_started = threading.Event()

    def new_reader():
        with gate.read():
            new_started.set()

    blocked = threading.Thread(target=new_reader, daemon=True)
    blocked.start()
    assert not new_started.wait(0.2), 'a new read slipped through while paused'
    release.set()
    inflight.join(2)
    pausing.join(2)
    assert read_done.is_set() and paused.is_set()
    sampler.resume_generation()
    blocked.join(2)
    assert new_started.is_set(), 'resume did not release the blocked read'


def test_admitted_outputs_close_source_on_early_consumer_close():
    engine = VLLMEngine.__new__(VLLMEngine)
    closed = asyncio.Event()

    async def source():
        try:
            yield 'first'
            yield 'second'  # consumer closes after the first; only aclose can end the source here
        finally:
            closed.set()

    async def run():
        source_gen = source()  # keep an external ref so only an explicit aclose can close it
        outputs = engine._delta_admitted_outputs(source_gen)
        assert await anext(outputs) == 'first'
        await outputs.aclose()
        assert closed.is_set(), 'early consumer close stranded the engine generator'
    asyncio.run(run())


def test_admitted_outputs_propagate_source_error():
    engine = VLLMEngine.__new__(VLLMEngine)

    async def source():
        yield 'first'
        raise ValueError('source failed')

    async def run():
        outputs = engine._delta_admitted_outputs(source())
        assert await anext(outputs) == 'first'
        with pytest.raises(ValueError, match='source failed'):
            await anext(outputs)
    asyncio.run(run())


def test_admitted_outputs_reject_before_consuming_when_delta_unavailable():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._delta_unavailable = True
    started = False

    async def source():
        nonlocal started
        started = True
        yield 'x'

    async def run():
        with pytest.raises(RuntimeError, match='unavailable'):
            async for _ in engine._delta_admitted_outputs(source()):
                pass
        assert not started, 'generator was consumed despite the closed availability gate'
    asyncio.run(run())


def test_admitted_outputs_pass_through_in_order_and_close_on_exhaustion():
    engine = VLLMEngine.__new__(VLLMEngine)
    closed = asyncio.Event()

    async def source():
        try:
            yield 'a'
            yield 'b'
        finally:
            closed.set()

    async def run():
        assert [item async for item in engine._delta_admitted_outputs(source())] == ['a', 'b']
        assert closed.is_set(), 'generator left open after normal exhaustion'
    asyncio.run(run())


def test_stream_close_closes_admitted_engine_generator(monkeypatch):
    inputs = ModuleType('vllm.inputs')
    inputs.TextPrompt = lambda **fields: fields
    inputs.TokensPrompt = lambda **fields: fields
    sampling = ModuleType('vllm.sampling_params')
    sampling.RequestOutputKind = SimpleNamespace(DELTA=object())
    monkeypatch.setitem(sys.modules, inputs.__name__, inputs)
    monkeypatch.setitem(sys.modules, sampling.__name__, sampling)

    async def run():
        closed = asyncio.Event()
        async def source(**kwargs):
            try:
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1], finish_reason=None)])
                yield SimpleNamespace(outputs=[])
            finally:
                closed.set()
        engine = VLLMEngine.__new__(VLLMEngine)
        engine.enable_lora = False
        engine._synced_lora_request = None
        engine.engine = SimpleNamespace(generate=source)
        engine.get_tokenizer = AsyncMock(return_value=SimpleNamespace(decode=lambda *args, **kwargs: 'token'))
        params = SimpleNamespace(to_vllm=lambda **kwargs: SimpleNamespace())
        stream = engine.generate_stream([1], params)
        assert await anext(stream) == ('token', None)
        await stream.aclose()
        assert closed.is_set()
    asyncio.run(run())


@pytest.mark.parametrize(('visible_devices', 'expected'), [
    ('2,3', '3'),
    ('3,2', '2'),
    ('GPU-first,GPU-second', 'GPU-second'),
    ('MIG-first,MIG-second', 'MIG-second'),
    (' 2, 3 ', '3'),
])
def test_selector_uses_worker_local_device_mapping(monkeypatch, visible_devices, expected):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible_devices)
    assert _cuda_visible_device(SimpleNamespace(index=1)) == expected


def test_single_card_worker_preserves_assigned_selector(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    assert _cuda_visible_device(SimpleNamespace(index=0)) == '3'


def test_unrestricted_worker_uses_device_ordinal(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert _cuda_visible_device(SimpleNamespace(index=2)) == '2'


@pytest.mark.parametrize('visible_devices', ['', '-1', '2,'])
def test_invalid_worker_visibility_is_rejected(monkeypatch, visible_devices):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible_devices)
    device = SimpleNamespace(index=1 if visible_devices == '2,' else 0)
    with pytest.raises(ValueError, match='Cannot map CUDA device'):
        _cuda_visible_device(device)


def test_device_ordinal_outside_visible_list_is_rejected(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    with pytest.raises(ValueError, match='Cannot map CUDA device'):
        _cuda_visible_device(SimpleNamespace(index=1))
