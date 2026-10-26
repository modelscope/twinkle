# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout-side NPU (HCCL) orchestration: manager rendezvous and DeltaSamplerMixin RPC sequence, all mocked."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from twinkle import Platform
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.checkpoint_engine.delta import DeltaSyncConfig
from twinkle.checkpoint_engine.delta.rollout.mixin import DeltaSamplerMixin
from twinkle.checkpoint_engine.hccl_checkpoint_engine import MasterMetadata

MASTER = MasterMetadata(zmq_ip='10.0.0.1', zmq_port=6000, dist_ip='10.0.0.1', dist_port=5000)


@pytest.fixture(autouse=True)
def fake_npu_platform(monkeypatch):
    """No torch_npu here: resolve the platform names without importing the NPU backend."""
    classes = {'NPU': type('NPU', (), {}), 'GPU': type('GPU', (), {})}
    monkeypatch.setattr(Platform, 'get_platform', staticmethod(lambda platform=None: classes[platform or 'GPU']))


# --------------------------------------------------------------------------- manager
def npu_pair(tps=(2, ), trainer_device='npu', worker_device='npu'):
    model = Mock(_actors=[1, 2], device_mesh=SimpleNamespace(world_size=2))
    sampler = Mock(_actors=list(range(len(tps))), device_mesh=SimpleNamespace(data_world_size=len(tps)))
    model.prepare_delta_sync.return_value = dict(device=trainer_device, master_metadata=MASTER)
    caps = [dict(tp=tp, device=worker_device, replica_rank=rank) for rank, tp in enumerate(tps)]
    sampler.validate_delta_sampler.return_value = caps if len(caps) > 1 else caps[0]
    model.send_delta_weights.return_value = lambda: {'payload_bytes': 6}
    sampler.receive_delta_weights.return_value = lambda: [{'changed': True}] * len(tps)
    if len(tps) == 1:
        sampler.receive_delta_weights.return_value = lambda: {'changed': True}
    manager = CheckpointEngineManager(model, sampler, platform='NPU', sync_backend='delta_sharded')
    return manager, model, sampler


def test_npu_preflight_order_and_no_worker_metadata():
    manager, model, sampler = npu_pair((2, 2))
    order = []
    sampler.validate_delta_sampler.side_effect = lambda cfg: (order.append('validate'),
                                                              [dict(tp=2, device='npu', replica_rank=0),
                                                               dict(tp=2, device='npu', replica_rank=1)])[1]
    model.prepare_delta_sync.side_effect = lambda cfg: (
        order.append('prepare'), dict(device='npu', master_metadata=MASTER))[1]
    manager.sync_weights()
    assert order == ['validate', 'prepare']
    sampler.delta_sampler_metadata.assert_not_called()


def test_npu_endpoint_world_size_and_no_rollout_workers():
    manager, model, sampler = npu_pair((2, 2))
    manager.sync_weights()
    manager.sync_weights()
    endpoint = model.init_delta_transport.call_args.args[0]
    assert endpoint['world_size'] == 5 and 'rollout_workers' not in endpoint
    assert endpoint['master_metadata'] is MASTER
    assert sampler.init_delta_sampler_transport.call_args.args[0] == endpoint


def test_npu_world_size_sums_tp_of_each_replica():
    manager, model, _ = npu_pair((1, ))
    manager.sync_weights()
    assert model.init_delta_transport.call_args.args[0]['world_size'] == 2


def test_npu_begin_before_init_and_init_only_on_seed():
    events = []
    manager, model, sampler = npu_pair((2, ))
    sampler.begin_delta_sync.side_effect = lambda: events.append('begin')
    model.init_delta_transport.side_effect = lambda ep: (events.append('trainer-init'), lambda: None)[1]
    sampler.init_delta_sampler_transport.side_effect = lambda ep: (events.append('sampler-init'), lambda: None)[1]
    manager.sync_weights()
    assert events == ['begin', 'trainer-init', 'sampler-init']
    manager.sync_weights()
    assert events == ['begin', 'trainer-init', 'sampler-init', 'begin']
    assert manager.last_delta_metrics['seed'] is False
    assert manager._delta_replica_count == 1


def test_npu_device_mismatch_names_both_values():
    manager, model, sampler = npu_pair((2, ), trainer_device='npu', worker_device='cuda')
    with pytest.raises(ValueError, match="'npu'.*'cuda'"):
        manager.sync_weights()
    sampler.begin_delta_sync.assert_not_called()
    model.init_delta_transport.assert_not_called()


@pytest.mark.parametrize('ranks', [[0, 0], [1, 2], [0, 2], [None, 1], [0, None]])
def test_npu_bad_replica_ranks_rejected(ranks):
    manager, model, sampler = npu_pair((2, 2))
    sampler.validate_delta_sampler.return_value = [
        dict(tp=2, device='npu', **({} if rank is None else dict(replica_rank=rank))) for rank in ranks]
    with pytest.raises(ValueError, match='delta replica ranks'):
        manager.sync_weights()
    sampler.begin_delta_sync.assert_not_called()
    model.init_delta_transport.assert_not_called()
    sampler.fail_delta_sampler.assert_not_called()


def test_npu_single_replica_must_be_rank_zero():
    manager, _, sampler = npu_pair((2, ))
    sampler.validate_delta_sampler.return_value = dict(tp=2, device='npu', replica_rank=1)
    with pytest.raises(ValueError, match='delta replica ranks'):
        manager.sync_weights()


def test_npu_replica_order_does_not_matter():
    manager, model, sampler = npu_pair((2, 2))
    sampler.validate_delta_sampler.return_value = [dict(tp=2, device='npu', replica_rank=1),
                                                   dict(tp=2, device='npu', replica_rank=0)]
    manager.sync_weights()
    assert manager._delta_replica_count == 2


def test_npu_replica_disagreement_keeps_rollout_paused():
    manager, model, sampler = npu_pair((2, 2))
    manager.sync_weights()
    sampler.publish_delta_sampler.reset_mock()
    sampler.receive_delta_weights.return_value = lambda: [{'changed': True}, {'changed': False}]
    with pytest.raises(ValueError, match='replicas disagree'):
        manager.sync_weights()
    sampler.publish_delta_sampler.assert_not_called()
    sampler.fail_delta_sampler.assert_called_once()


def test_npu_replica_count_mismatch_fails():
    manager, _, sampler = npu_pair((2, 2))
    sampler.receive_delta_weights.return_value = lambda: [{'changed': True}]
    with pytest.raises(ValueError, match='replicas disagree'):
        manager.sync_weights()
    sampler.fail_delta_sampler.assert_called_once()


@pytest.mark.parametrize('stage', ['send_delta_weights', 'receive_delta_weights', 'init_delta_sampler_transport'])
def test_npu_failure_marks_failed_and_fails_sampler(stage):
    manager, model, sampler = npu_pair((2, ))
    getattr(model if stage == 'send_delta_weights' else sampler, stage).side_effect = TimeoutError('injected')
    with pytest.raises(TimeoutError):
        manager.sync_weights()
    assert manager._delta_failed and manager._delta_round == 0
    sampler.fail_delta_sampler.assert_called_once()
    sampler.publish_delta_sampler.assert_not_called()
    with pytest.raises(RuntimeError, match='restart the entire'):
        manager.sync_weights()


def test_npu_preflight_failure_is_retryable():
    manager, model, sampler = npu_pair((2, ))
    model.prepare_delta_sync.side_effect = ValueError('bad config')
    with pytest.raises(ValueError, match='bad config'):
        manager.sync_weights()
    sampler.begin_delta_sync.assert_not_called()
    model.prepare_delta_sync.side_effect = None
    manager.sync_weights()
    assert manager.last_delta_metrics['seed'] is True


def test_npu_close_then_new_seed():
    manager, model, sampler = npu_pair((2, ))
    manager.sync_weights()
    manager.close_sync_session()
    sampler.close_delta_sampler.assert_called_once()
    manager.sync_weights()
    assert model.init_delta_transport.call_count == 2
    assert sampler.validate_delta_sampler.call_count == 2


def test_gpu_manager_still_uses_worker_metadata():
    model = Mock(_actors=[1], device_mesh=SimpleNamespace(world_size=1))
    sampler = Mock(_actors=[2], device_mesh=SimpleNamespace(data_world_size=1))
    model.prepare_delta_sync.return_value = dict(device='cuda', master_metadata=object())
    sampler.validate_delta_sampler.return_value = dict(device='cuda', tp=1)
    sampler.delta_sampler_metadata.return_value = [dict(node_ip='n', device_uuid='g0', zmq_handle='h')]
    sampler.receive_delta_weights.return_value = lambda: {'changed': True}
    manager = CheckpointEngineManager(model, sampler, sync_backend='delta_sharded')
    manager.sync_weights()
    sampler.delta_sampler_metadata.assert_called_once()
    assert 'rollout_workers' in model.init_delta_transport.call_args.args[0]


# --------------------------------------------------------------------------- sampler
class LocalSampler(DeltaSamplerMixin):
    def _run_in_loop(self, coroutine):
        return asyncio.run(coroutine)


def make_sampler(monkeypatch, *, rank=0, world=1, tp=2, device='npu', sleep=True):
    if rank is None:
        monkeypatch.delenv('RANK', raising=False)
    else:
        monkeypatch.setenv('RANK', str(rank))
    monkeypatch.setenv('WORLD_SIZE', str(world))
    sampler = LocalSampler()
    sampler.engine = SimpleNamespace(
        enable_sleep_mode=sleep, pause_delta_generation=AsyncMock(),
        engine=SimpleNamespace(reset_prefix_cache=AsyncMock(return_value=True), reset_mm_cache=AsyncMock(),
                               reset_encoder_cache=AsyncMock()))
    sampler._delta_rpc = AsyncMock(side_effect=lambda method, kwargs=None: (
        [dict(tp=tp, device=device)] * tp if method == 'validate_delta_worker' else []))
    return sampler


def rpc_calls(sampler):
    return [(c.args[0], c.args[1] if len(c.args) > 1 else None) for c in sampler._delta_rpc.await_args_list]


def validated_npu_sampler(monkeypatch, *, rank=0, world=1, tp=2):
    sampler = make_sampler(monkeypatch, rank=rank, world=world, tp=tp)
    caps = sampler.validate_delta_sampler(DeltaSyncConfig(timeout_seconds=7))
    sampler.init_delta_sampler_transport({'world_size': 1 + world * tp, 'master_metadata': MASTER})
    sampler._delta_rpc.reset_mock()
    return sampler, caps


def test_validate_returns_replica_rank_on_npu(monkeypatch):
    sampler = make_sampler(monkeypatch, rank=1, world=2, tp=2)
    assert sampler.validate_delta_sampler(DeltaSyncConfig()) == dict(tp=2, device='npu', replica_rank=1)


def test_validate_keeps_exact_legacy_dict_on_cuda(monkeypatch):
    sampler = make_sampler(monkeypatch, rank=1, world=2, tp=2, device='cuda')
    assert sampler.validate_delta_sampler(DeltaSyncConfig()) == dict(tp=2, device='cuda')


@pytest.mark.parametrize('rank, world', [(None, 1), (-1, 1), (2, 2), (3, 2)])
def test_validate_npu_undeterminable_replica_rank_is_a_clear_error(monkeypatch, rank, world):
    sampler = make_sampler(monkeypatch, rank=rank, world=world)
    with pytest.raises(RuntimeError, match='cannot determine the delta replica rank'):
        sampler.validate_delta_sampler(DeltaSyncConfig())


def test_validate_npu_sleep_mode_message_covers_npu(monkeypatch):
    sampler = make_sampler(monkeypatch, sleep=False)
    with pytest.raises(ValueError, match='enable_sleep_mode=True or free_cache_engine=False'):
        sampler.validate_delta_sampler(DeltaSyncConfig())
    sampler.validate_delta_sampler(DeltaSyncConfig(free_cache_engine=False))


def test_npu_init_transport_spawns_no_actors_and_skips_receiver_metadata(monkeypatch):
    import ray
    remote = Mock()
    monkeypatch.setattr(ray, 'remote', remote)
    monkeypatch.setattr(ray, 'nodes', Mock(side_effect=AssertionError('must not touch ray')))
    sampler = make_sampler(monkeypatch)
    sampler.validate_delta_sampler(DeltaSyncConfig())
    sampler._delta_rpc.reset_mock()
    sampler._delta_weight_transfer_initialized = True
    sampler.init_delta_sampler_transport({'world_size': 3, 'master_metadata': MASTER})
    remote.assert_not_called()
    sampler._delta_rpc.assert_not_called()
    assert not sampler._delta_weight_transfer_initialized
    assert not getattr(sampler, '_delta_receivers', [])


def test_npu_round_rpc_sequence_and_kwargs(monkeypatch):
    sampler, _ = validated_npu_sampler(monkeypatch, rank=0, world=1, tp=2)
    assert sampler.receive_delta_weights() == {'changed': True}
    assert rpc_calls(sampler) == [
        ('init_weight_transfer_engine', {'init_info': dict(
            dist_ip='10.0.0.1', dist_port=5000, zmq_ip='10.0.0.1', zmq_port=6000,
            topic='bucket_metadata', world_size=3)}),
        ('start_weight_update', None),
        ('update_twinkle_delta_weights', {'update_info': [{'group_rank': 1}, {'group_rank': 2}]}),
        ('finish_weight_update', None),
    ]
    engine = sampler.engine.engine
    engine.reset_prefix_cache.assert_awaited_once_with(reset_connector=True)
    engine.reset_mm_cache.assert_awaited_once()
    engine.reset_encoder_cache.assert_awaited_once()


def test_npu_init_info_forwards_trainer_hccl_env(monkeypatch):
    sampler = make_sampler(monkeypatch, rank=0, world=1, tp=1)
    sampler.validate_delta_sampler(DeltaSyncConfig())
    env = {'HCCL_IF_BASE_PORT': '30000'}
    sampler.init_delta_sampler_transport({'world_size': 2, 'master_metadata': MASTER, 'hccl_env': env})
    sampler._delta_rpc.reset_mock()
    sampler.receive_delta_weights()
    method, kwargs = rpc_calls(sampler)[0]
    assert method == 'init_weight_transfer_engine' and kwargs['init_info']['hccl_env'] == env


def test_npu_endpoint_carries_trainer_hccl_env():
    manager, model, sampler = npu_pair((2, ))
    env = {'HCCL_IF_BASE_PORT': '30000'}
    model.prepare_delta_sync.return_value = dict(device='npu', master_metadata=MASTER, hccl_env=env)
    manager.sync_weights()
    assert sampler.init_delta_sampler_transport.call_args.args[0]['hccl_env'] == env


@pytest.mark.parametrize('rank, world, tp, expected', [
    (0, 2, 2, [1, 2]), (1, 2, 2, [3, 4]), (0, 1, 1, [1]), (2, 3, 1, [3]), (1, 2, 4, [5, 6, 7, 8])])
def test_npu_group_ranks_replica_major_tp_minor(monkeypatch, rank, world, tp, expected):
    sampler, _ = validated_npu_sampler(monkeypatch, rank=rank, world=world, tp=tp)
    sampler.receive_delta_weights()
    update = [kw for name, kw in rpc_calls(sampler) if name == 'update_twinkle_delta_weights'][0]
    assert [item['group_rank'] for item in update['update_info']] == expected


def test_npu_init_rpc_only_first_round_and_again_after_close(monkeypatch):
    sampler, _ = validated_npu_sampler(monkeypatch)
    sampler.receive_delta_weights()
    sampler.receive_delta_weights()
    names = [name for name, _ in rpc_calls(sampler)]
    assert names.count('init_weight_transfer_engine') == 1 and names.count('update_twinkle_delta_weights') == 2
    sampler.close_delta_sampler()
    sampler.validate_delta_sampler(DeltaSyncConfig())
    sampler.init_delta_sampler_transport({'world_size': 3, 'master_metadata': MASTER})
    sampler.receive_delta_weights()
    names = [name for name, _ in rpc_calls(sampler)]
    assert names.count('init_weight_transfer_engine') == 2


def test_npu_close_shuts_workers_down_and_resets(monkeypatch):
    sampler, _ = validated_npu_sampler(monkeypatch)
    sampler.receive_delta_weights()
    sampler._delta_rpc.reset_mock()
    sampler.close_delta_sampler()
    assert rpc_calls(sampler) == [('shutdown_delta_weight_transfer', None)]
    sampler.engine.pause_delta_generation.assert_awaited_once_with(7)
    assert sampler.engine._delta_unavailable is True
    assert not sampler._delta_weight_transfer_initialized


def test_npu_close_resets_flag_even_when_worker_shutdown_fails(monkeypatch):
    sampler, _ = validated_npu_sampler(monkeypatch)
    sampler.receive_delta_weights()
    sampler._delta_rpc.side_effect = RuntimeError('rpc down')
    with pytest.raises(RuntimeError, match='rpc down'):
        sampler.close_delta_sampler()
    assert not sampler._delta_weight_transfer_initialized


@pytest.mark.parametrize('failing', ['init_weight_transfer_engine', 'update_twinkle_delta_weights'])
def test_npu_round_failure_propagates_without_cache_clear(monkeypatch, failing):
    sampler, _ = validated_npu_sampler(monkeypatch)

    async def rpc(method, kwargs=None):
        if method == failing:
            raise TimeoutError('injected')
    sampler._delta_rpc = AsyncMock(side_effect=rpc)
    with pytest.raises(TimeoutError):
        sampler.receive_delta_weights()
    sampler.engine.engine.reset_prefix_cache.assert_not_called()
    if failing == 'init_weight_transfer_engine':
        assert not sampler._delta_weight_transfer_initialized


def test_npu_receive_without_transport_is_a_clear_error(monkeypatch):
    sampler = make_sampler(monkeypatch)
    sampler.validate_delta_sampler(DeltaSyncConfig())
    with pytest.raises(RuntimeError, match='not initialized'):
        sampler.receive_delta_weights()


# --------------------------------------------------------------------------- vllm_engine backend wiring
@pytest.fixture
def engine_env(monkeypatch):
    """Stub vLLM and the adapter module so VLLMEngine._create_engine runs up to AsyncEngineArgs."""
    import sys
    from types import ModuleType

    captured = {}

    def args_cls(model=None, enable_sleep_mode=False, weight_transfer_config=None, **_):
        captured['args'] = dict(enable_sleep_mode=enable_sleep_mode, weight_transfer_config=weight_transfer_config)
        return SimpleNamespace(create_engine_config=lambda usage_context: 'vllm-config')

    modules = {
        'vllm.engine.arg_utils': SimpleNamespace(AsyncEngineArgs=args_cls),
        'vllm.usage.usage_lib': SimpleNamespace(UsageContext=SimpleNamespace(OPENAI_API_SERVER='api')),
        'vllm.v1.engine.async_llm': SimpleNamespace(AsyncLLM=SimpleNamespace(from_vllm_config=lambda **_: 'engine')),
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(vars(attrs))
        monkeypatch.setitem(sys.modules, name, module)
    adapter = SimpleNamespace(require_vllm_delta_support=Mock(), register_twinkle_delta_weight_transfer_engine=Mock(),
                              register_twinkle_delta_hccl_weight_transfer_engine=Mock())
    monkeypatch.setitem(sys.modules, 'twinkle.sampler.vllm_sampler.delta.transfer_engine', adapter)
    return SimpleNamespace(adapter=adapter, captured=captured)


def _bare_vllm_engine(**engine_kwargs):
    from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine
    engine = VLLMEngine.__new__(VLLMEngine)
    engine.__dict__.update(
        model_id='m', tensor_parallel_size=1, gpu_memory_utilization=.5, max_num_seqs=1, trust_remote_code=False,
        enforce_eager=True, dtype='bfloat16', load_format='auto', max_model_len=None, quantization=None,
        enable_prefix_caching=False, enable_sleep_mode=False, runner=None, logprobs_mode=None, enable_lora=False,
        enable_sampling_replay=False, engine_kwargs=engine_kwargs)
    return engine


def test_vllm_engine_hccl_backend_registers_engine_without_forcing_sleep_mode(engine_env):
    engine = _bare_vllm_engine(weight_transfer_config={'backend': 'twinkle_delta_hccl'})
    assert engine._create_engine() == 'engine'
    engine_env.adapter.require_vllm_delta_support.assert_called_once()
    engine_env.adapter.register_twinkle_delta_hccl_weight_transfer_engine.assert_called_once()
    engine_env.adapter.register_twinkle_delta_weight_transfer_engine.assert_not_called()
    assert engine.enable_sleep_mode is False and engine_env.captured['args']['enable_sleep_mode'] is False
    assert engine_env.captured['args']['weight_transfer_config'] == {'backend': 'twinkle_delta_hccl'}


def test_vllm_engine_ipc_backend_still_enables_sleep_mode(engine_env):
    engine = _bare_vllm_engine(weight_transfer_config={'backend': 'twinkle_delta_ipc'})
    engine._create_engine()
    engine_env.adapter.register_twinkle_delta_weight_transfer_engine.assert_called_once()
    engine_env.adapter.register_twinkle_delta_hccl_weight_transfer_engine.assert_not_called()
    assert engine.enable_sleep_mode is True


def test_vllm_engine_hccl_backend_requires_weight_transfer_config_support(engine_env, monkeypatch):
    import sys

    def old_args_cls(model=None, **_):
        raise AssertionError('must fail before constructing args')

    sys.modules['vllm.engine.arg_utils'].AsyncEngineArgs = old_args_cls
    engine = _bare_vllm_engine(weight_transfer_config={'backend': 'twinkle_delta_hccl'})
    with pytest.raises(RuntimeError, match='weight_transfer_config'):
        engine._create_engine()
