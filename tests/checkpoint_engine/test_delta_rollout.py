# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout side: sampler coordination, GPU-local receivers, topology placement, request admission and device mapping."""
import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
import torch

from twinkle.checkpoint_engine.delta.rollout.cuda_receiver import CudaDeltaReceiver
from twinkle.checkpoint_engine.delta.rollout.mixin import DeltaSamplerMixin
from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine
from twinkle.sampler.vllm_sampler.vllm_worker_extension import _cuda_visible_device


class LocalCudaSampler(DeltaSamplerMixin):
    def _run_in_loop(self, coroutine):
        return asyncio.run(coroutine)


def receiver_stub(flags):
    return SimpleNamespace(
        next_flush=SimpleNamespace(remote=AsyncMock(side_effect=flags)),
        send_flush=SimpleNamespace(remote=AsyncMock()),
    )


def test_zero_change_skips_native_update_but_restores_kv_and_generation():
    sampler = LocalCudaSampler()
    sampler._delta_receivers = [receiver_stub([None]), receiver_stub([None])]
    sampler._delta_config = SimpleNamespace(timeout_seconds=1, free_cache_engine=True)
    sampler.engine = SimpleNamespace(pause_delta_generation=AsyncMock(), engine=SimpleNamespace(
        sleep=AsyncMock(), wake_up=AsyncMock(), reset_prefix_cache=AsyncMock(return_value=True),
        resume_generation=AsyncMock()))
    sampler._delta_rpc = AsyncMock()
    sampler.begin_delta_sync()
    assert sampler.receive_delta_weights() == {'changed': False}
    sampler.publish_delta_sampler()
    sampler._delta_rpc.assert_not_called()
    sampler.engine.pause_delta_generation.assert_awaited_once_with(1)
    sampler.engine.engine.sleep.assert_awaited_once_with(level=1)
    assert sampler.engine.engine.wake_up.await_args_list == [
        call(tags=['weights']), call(tags=['kv_cache'])]
    sampler.engine.engine.resume_generation.assert_awaited_once()


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
        receiver.send_flush.remote.assert_awaited_once()
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
    sampler.engine = SimpleNamespace(pause_delta_generation=AsyncMock(), engine=SimpleNamespace(
        sleep=AsyncMock(), wake_up=AsyncMock(), resume_generation=AsyncMock()))
    sampler.begin_delta_sync()
    sampler.publish_delta_sampler()
    sampler.engine.pause_delta_generation.assert_awaited_once()
    sampler.engine.engine.resume_generation.assert_awaited_once()
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
    import ray
    metadata = [dict(node_ip='10.0.0.1', device_uuid=f'GPU-{rank}', zmq_handle=f'ipc:///tmp/rank{rank}.sock')
                for rank in range(2)]
    if use_worker_selector:
        for rank, worker in enumerate(metadata):
            worker['cuda_visible_device'] = str(rank + 2)
    created = []

    class Actor:
        def options(self, **options):
            self.options_for_rank = options
            return self

        def remote(self, config, endpoint, placement):
            created.append((self.options_for_rank, placement['rank'], placement['zmq_handle']))
            return SimpleNamespace(ready=SimpleNamespace(remote=lambda: placement['rank']))

    remote = Mock(return_value=lambda cls: Actor())
    monkeypatch.setattr(ray, 'remote', remote)
    monkeypatch.setattr(ray, 'nodes', lambda: [dict(Alive=True, NodeManagerAddress='10.0.0.1', NodeID='ab' * 28)])
    monkeypatch.setattr(ray, 'get', lambda refs, **kwargs: refs)
    sampler = LocalCudaSampler()
    sampler._delta_rpc = AsyncMock(return_value=metadata)
    sampler._delta_config = SimpleNamespace(timeout_seconds=1)
    sampler.init_delta_sampler_transport({'world_size': 3, 'master_metadata': 'master',
                                              'rollout_workers': metadata})
    assert [item[1:] for item in created] == [(1, 'ipc:///tmp/rank0.sock'), (2, 'ipc:///tmp/rank1.sock')]
    expected_selectors = ['2', '3'] if use_worker_selector else ['GPU-0', 'GPU-1']
    assert [item[0]['runtime_env']['env_vars']['CUDA_VISIBLE_DEVICES'] for item in created] == expected_selectors
    assert all(item[0]['scheduling_strategy'].node_id == 'ab' * 28 for item in created)
    assert remote.call_args.kwargs == {'num_cpus': 0, 'num_gpus': 0}


def test_replica_receivers_use_global_ranks_across_nodes(monkeypatch):
    import ray
    workers = [dict(node_ip=f'10.0.0.{rank // 2 + 1}', device_uuid=f'GPU-{rank}',
                    zmq_handle=f'ipc:///rank{rank}') for rank in range(4)]
    monkeypatch.setattr(ray, 'nodes', lambda: [
        dict(Alive=True, NodeManagerAddress=f'10.0.0.{rank + 1}', NodeID=str(rank)) for rank in range(2)])
    endpoint = dict(world_size=5, master_metadata='master', rollout_workers=workers)
    sampler = LocalCudaSampler()
    placements = [sampler._delta_receiver_placements(workers[:2], endpoint),
                  sampler._delta_receiver_placements(workers[2:], endpoint)]
    assert [[item['rank'] for item in replica] for replica in placements] == [[1, 2], [3, 4]]
    assert [[item['node_id'] for item in replica] for replica in placements] == [['0', '0'], ['1', '1']]


def test_sampler_dp_allows_independent_rollout_replicas():
    from twinkle.checkpoint_engine.delta import DeltaSyncConfig
    sampler = LocalCudaSampler()
    sampler.device_mesh = SimpleNamespace(data_world_size=2)
    sampler.engine = SimpleNamespace(enable_sleep_mode=True)
    sampler._delta_rpc = AsyncMock(return_value=[dict(tp=2, device='cuda')] * 2)
    assert sampler.validate_delta_sampler(DeltaSyncConfig()) == dict(tp=2, device='cuda')


def test_submission_barrier_waits_for_request_already_past_gate():
    async def run():
        engine = VLLMEngine.__new__(VLLMEngine)
        admitted = asyncio.Event()
        release = asyncio.Event()
        paused = asyncio.Event()
        engine.engine = SimpleNamespace(pause_generation=AsyncMock(side_effect=lambda **_: paused.set()))

        async def source():
            admitted.set()
            await release.wait()
            yield 'first'

        async def consume():
            return [output async for output in engine._delta_admitted_outputs(source())]

        request = asyncio.create_task(consume())
        await admitted.wait()
        update = asyncio.create_task(engine.pause_delta_generation(1))
        await asyncio.sleep(0)
        assert engine._delta_unavailable
        assert not paused.is_set()
        with pytest.raises(RuntimeError, match='unavailable'):
            await consume()
        release.set()
        assert await request == ['first']
        await update
        assert paused.is_set()
        assert engine._delta_admitting == 0
    asyncio.run(run())


def test_admission_error_does_not_strand_barrier():
    async def run():
        engine = VLLMEngine.__new__(VLLMEngine)
        engine.engine = SimpleNamespace(pause_generation=AsyncMock())
        async def source():
            raise ValueError('admission failed')
            yield
        with pytest.raises(ValueError, match='admission failed'):
            async for _ in engine._delta_admitted_outputs(source()):
                pass
        await engine.pause_delta_generation(1)
        engine.engine.pause_generation.assert_awaited_once()
        assert engine._delta_admitting == 0
    asyncio.run(run())


def test_admission_timeout_keeps_gate_closed_and_does_not_pause_or_write():
    async def run():
        engine = VLLMEngine.__new__(VLLMEngine)
        engine._delta_admitting = 1
        engine.engine = SimpleNamespace(pause_generation=AsyncMock())
        with pytest.raises(TimeoutError):
            await engine.pause_delta_generation(.01)
        engine.engine.pause_generation.assert_not_awaited()
        assert engine._delta_unavailable
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
        assert engine._delta_admitting == 0
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
