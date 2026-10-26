# Copyright (c) ModelScope Contributors. All rights reserved.
"""Worker-side HCCL delta receiver: decode, pump, vLLM engine, worker extension (CPU, fakes)."""
from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from twinkle.checkpoint_engine.delta.rollout import hccl_receiver
from twinkle.checkpoint_engine.delta.protocol.decode import DeltaPatch, decode_delta_wire
from twinkle.checkpoint_engine.delta.protocol.wire import checksum

from .test_delta_vllm import adapter, named_flush  # noqa: F401  (adapter is a fixture)


# --------------------------------------------------------------------------- helpers
def _spec(named):
    return json.loads(bytes(dict(named)['__delta_spec__'].numpy().tobytes()).decode())


def _replace_spec(named, **changes):
    spec = _spec(named)
    spec.update(changes)
    tensor = torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8)
    return [('__delta_spec__', tensor)] + [(k, v) for k, v in named if k != '__delta_spec__']


def _mutate_param(named, **changes):
    spec = _spec(named)
    spec['params'][0].update(changes)
    return _replace_spec(named, params=spec['params'])


def _manifest(named, is_last=True):
    """Build the trainer-side manifest + the byte payloads for one flush."""
    t = dict(named)
    spec = _spec(named)
    pos = t.get('__positions__') if spec['encoding'] != 'dense' else None
    val = t['__values__']
    meta = dict(is_full=spec['encoding'] == 'dense', encoding=spec['encoding'], is_last=is_last,
                terminal_empty=False, val_numel=val.numel(), val_dtype=str(val.dtype).split('.')[-1], spec=spec)
    if spec['encoding'] != 'dense':
        meta['pos_numel'] = pos.numel()
    return meta, (pos.clone() if pos is not None else None), val.view(torch.uint8).clone()


class FakeSub:

    def __init__(self, metas):
        self.items = [('bucket_metadata', m) for m in metas]
        self.recv_count = 0
        self.closed = False
        self.close_kwargs = None
        self.context = Mock()

    def recv_string(self):
        self.recv_count += 1
        return self.items[0][0]

    def recv_pyobj(self):
        return self.items.pop(0)[1]

    def close(self, **kwargs):
        self.closed = True
        self.close_kwargs = kwargs


class FakeComm:
    """Fills broadcast targets from queued payloads, in call order."""

    def __init__(self, payloads=()):
        self.payloads = list(payloads)
        self.calls = []
        self.comm = object()
        self.destroyed = []

    def broadcast(self, tensor, src=0):
        assert src == 0
        data = self.payloads.pop(0)
        assert data.numel() == tensor.numel()
        tensor.copy_(data)
        self.calls.append(tensor.numel())

    def destroyComm(self, handle):
        self.destroyed.append(handle)


@pytest.fixture
def sync(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(torch.cpu, 'synchronize', mock)
    return mock


def _terminal():
    return dict(is_full=False, encoding='indices', is_last=True, terminal_empty=True)


def _run_pump(flushes, sync_mock=None, extra_metas=()):
    metas, payloads, applied = [], [], []
    for i, (named, is_last) in enumerate(flushes):
        meta, pos, val = _manifest(named, is_last)
        metas.append(meta)
        if pos is not None:
            payloads.append(pos)
        payloads.append(val)
    sub, comm = FakeSub(metas + list(extra_metas)), FakeComm(payloads)
    enc = hccl_receiver.pump_hccl_delta_session(sub, comm, torch.device('cpu'),
                                                on_flush=lambda n: applied.append(
                                                    [(k, v.clone()) for k, v in n]))
    return enc, sub, comm, applied


# --------------------------------------------------------------------------- (a) decode
def test_decode_indices_and_dense():
    enc, patches = decode_delta_wire(named_flush())
    assert enc == 'indices' and isinstance(patches[0], DeltaPatch)
    assert (patches[0].name, patches[0].shape, patches[0].dtype) == ('w', (4, ), torch.bfloat16)
    assert patches[0].positions.tolist() == [0, 3] and patches[0].positions.dtype == torch.int32
    enc, patches = decode_delta_wire(named_flush('dense'))
    assert enc == 'dense' and patches[0].positions is None and patches[0].values.numel() == 2


def test_decode_corrupt_checksum():
    flush = named_flush()
    flush[-1][1][0] = 42
    with pytest.raises(RuntimeError, match='checksum'):
        decode_delta_wire(flush)


def test_decode_dense_with_positions():
    flush = named_flush('dense')
    pos = torch.tensor([1], dtype=torch.int32).view(torch.uint8)
    flush = [flush[0], ('__positions__', pos), flush[2]]
    with pytest.raises(ValueError, match='must not contain positions'):
        decode_delta_wire(flush)


def test_decode_indices_without_positions():
    flush = [x for x in named_flush() if x[0] != '__positions__']
    with pytest.raises(ValueError, match='requires __positions__'):
        decode_delta_wire(flush)


def test_decode_bad_pos_width():
    with pytest.raises(ValueError, match='requires pos_width=4'):
        decode_delta_wire(_mutate_param(named_flush(), pos_width=2))


def test_decode_inconsistent_slice_lengths():
    with pytest.raises(ValueError, match='inconsistent lengths'):
        decode_delta_wire(_mutate_param(named_flush(), val_end=1))


def test_decode_unsupported_encoding():
    with pytest.raises(ValueError, match='Unsupported DeltaFlush encoding'):
        decode_delta_wire(_replace_spec(named_flush(), encoding='bogus'))


def test_decode_missing_tensors_and_bad_json():
    with pytest.raises(ValueError, match='missing __values__'):
        decode_delta_wire([x for x in named_flush() if x[0] != '__values__'])
    with pytest.raises(ValueError, match='missing __delta_spec__'):
        decode_delta_wire([x for x in named_flush() if x[0] != '__delta_spec__'])
    bad = [('__delta_spec__', torch.tensor(list(b'{nope'), dtype=torch.uint8))] + named_flush()[1:]
    with pytest.raises(ValueError, match='invalid JSON'):
        decode_delta_wire(bad)


def test_decode_payload_wrapper_returns_vllm_patches(adapter):
    encoding, patches = adapter.decode_delta_payload(named_flush())
    CheckpointWeightPatch = sys.modules['vllm.model_executor.model_loader.checkpoint_weight_patch'
                                        ].CheckpointWeightPatch
    assert encoding == 'indices' and isinstance(patches[0], CheckpointWeightPatch)
    assert patches[0].indices.tolist() == [0, 3]
    _, dense = adapter.decode_delta_payload(named_flush('dense'))
    assert dense[0].indices is None


# --------------------------------------------------------------------------- (b) pump
def test_pump_indices_flush_positions_before_values(sync):
    named = named_flush()
    enc, sub, comm, applied = _run_pump([(named, True)])
    assert enc == 'indices'
    assert comm.calls == [8, 4]  # positions (2 x int32 = 8 B) then values (2 x bf16 = 4 B)
    assert [k for k, _ in applied[0]] == ['__delta_spec__', '__positions__', '__values__']
    got = dict(applied[0])
    assert got['__values__'].dtype == torch.bfloat16
    assert torch.equal(got['__values__'].view(torch.uint8), dict(named)['__values__'].view(torch.uint8))
    assert json.loads(bytes(got['__delta_spec__'].numpy().tobytes()).decode()) == _spec(named)
    # the reassembled flush is a valid, checksummed payload
    assert decode_delta_wire(applied[0])[0] == 'indices'
    sync.assert_called()


def test_pump_dense_flush_has_no_positions_broadcast(sync):
    enc, _, comm, applied = _run_pump([(named_flush('dense'), True)])
    assert enc == 'dense' and comm.calls == [4]
    assert [k for k, _ in applied[0]] == ['__delta_spec__', '__values__']
    assert decode_delta_wire(applied[0])[0] == 'dense'


def test_pump_multi_flush_stops_after_is_last(sync):
    enc, sub, comm, applied = _run_pump([(named_flush(), False), (named_flush(), True)],
                                        extra_metas=[_terminal()])
    assert enc == 'indices' and len(applied) == 2 and comm.calls == [8, 4, 8, 4]
    assert sub.recv_count == 2 and len(sub.items) == 1  # nothing consumed after is_last


def test_pump_terminal_only_session(sync):
    enc, sub, comm, applied = _run_pump([], extra_metas=[_terminal()])
    assert enc == 'indices' and applied == [] and comm.calls == [] and sub.recv_count == 1


def test_pump_terminal_after_flushes(sync):
    enc, _, comm, applied = _run_pump([(named_flush('dense'), False)], extra_metas=[_terminal()])
    assert enc == 'dense' and len(applied) == 1


@pytest.mark.parametrize('meta', [
    'not-a-dict',
    {'is_last': True},
    {'is_last': True, 'encoding': 'indices', 'val_numel': 1, 'val_dtype': 'bfloat16', 'spec': {}},
    {'encoding': 'dense', 'val_numel': 1, 'val_dtype': 'bfloat16', 'spec': {}},
    {'is_last': True, 'encoding': 'dense', 'val_numel': 1, 'val_dtype': 'nope', 'spec': {}},
])
def test_pump_malformed_manifest_raises_value_error(sync, meta):
    sub, comm = FakeSub([meta]), FakeComm()
    with pytest.raises(ValueError, match='Malformed delta manifest'):
        hccl_receiver.pump_hccl_delta_session(sub, comm, torch.device('cpu'), on_flush=Mock())
    assert comm.calls == []


def test_pump_synchronizes_device_type(monkeypatch):
    npu = SimpleNamespace(synchronize=Mock())
    monkeypatch.setattr(torch, 'npu', npu, raising=False)
    meta, pos, val = _manifest(named_flush('dense'))
    dev = SimpleNamespace(type='npu')
    # allocate on cpu while the "device type" resolves to the fake torch.npu module
    real_empty = torch.empty
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: real_empty(*a, **{**k, 'device': 'cpu'}))
    monkeypatch.setattr(torch.Tensor, 'to', lambda self, *a, **k: self)
    hccl_receiver.pump_hccl_delta_session(FakeSub([meta]), FakeComm([val]), dev, on_flush=lambda n: None)
    npu.synchronize.assert_called()


def test_build_communicator_args(monkeypatch):
    stub = Mock(return_value='comm')
    monkeypatch.setattr(hccl_receiver, 'stateless_init_process_group', stub)
    assert hccl_receiver.build_delta_hccl_communicator('1.2.3.4', 99, 2, 3, torch.device('cpu', 1)) == 'comm'
    stub.assert_called_once_with(master_address='1.2.3.4', master_port=99, rank=2, world_size=3, device=1,
                                 backend='hccl')


def test_build_communicator_applies_trainer_hccl_env(monkeypatch):
    seen = {}
    monkeypatch.setenv('HCCL_IF_BASE_PORT', '11111')  # worker's own (mismatched) value
    monkeypatch.setattr(hccl_receiver, 'stateless_init_process_group',
                        lambda **_: seen.update(port=hccl_receiver.os.environ['HCCL_IF_BASE_PORT']) or 'comm')
    env = {'HCCL_IF_BASE_PORT': '22222'}
    assert hccl_receiver.build_delta_hccl_communicator('1.2.3.4', 99, 1, 2, torch.device('cpu', 0),
                                                       hccl_env=env) == 'comm'
    assert seen['port'] == '22222'


# --------------------------------------------------------------------------- (c) engine
@pytest.fixture
def hccl(adapter, monkeypatch, sync):
    order = []
    sub = FakeSub([])
    comm = FakeComm()
    zmq_mod = sys.modules['twinkle.checkpoint_engine.zmq_pubsub']

    def fake_connect(ip, port, topic):
        order.append(('connect', ip, port, topic))
        return sub

    def fake_build(ip, port, rank, world, device, hccl_env=None):
        order.append(('build', ip, port, rank, world))
        hccl_envs.append(hccl_env)
        return comm

    hccl_envs = []
    monkeypatch.setattr(zmq_mod, 'connect_subscriber', fake_connect)
    monkeypatch.setattr(hccl_receiver, 'build_delta_hccl_communicator', fake_build)
    return SimpleNamespace(adapter=adapter, order=order, sub=sub, comm=comm, hccl_envs=hccl_envs)


def hccl_engine(adapter, device='cpu'):
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1),
                             model_config=SimpleNamespace(dtype=torch.bfloat16))
    model = SimpleNamespace(named_modules=lambda **_: [], named_parameters=lambda: [])
    engine = adapter.TwinkleDeltaHCCLWeightTransferEngine(None, config, torch.device(device), model)
    engine._validate_configuration = Mock()  # device type is checked separately below
    return engine


def init_info(adapter, world_size=3):
    return adapter.TwinkleDeltaHCCLInitInfo(dist_ip='10.0.0.1', dist_port=5000, zmq_ip='10.0.0.2', zmq_port=6000,
                                            world_size=world_size)


def feed(hccl_ns, flushes):
    """Queue manifests on the fake SUB and payloads on the fake comm."""
    for named, is_last in flushes:
        meta, pos, val = _manifest(named, is_last)
        hccl_ns.sub.items.append(('bucket_metadata', meta))
        if pos is not None:
            hccl_ns.comm.payloads.append(pos)
        hccl_ns.comm.payloads.append(val)


def feed_terminal(hccl_ns):
    hccl_ns.sub.items.append(('bucket_metadata', _terminal()))


def test_ensure_channel_order_reuse_and_rank_change(hccl):
    engine = hccl_engine(hccl.adapter)
    with pytest.raises(RuntimeError, match='init_transfer_engine must run'):
        engine._ensure_channel(1)
    engine.init_transfer_engine(init_info(hccl.adapter))
    sub, comm = engine._ensure_channel(1)
    assert (sub, comm) == (hccl.sub, hccl.comm)
    assert hccl.order == [('connect', '10.0.0.2', 6000, 'bucket_metadata'), ('build', '10.0.0.1', 5000, 1, 3)]
    assert engine._ensure_channel(1) == (sub, comm) and len(hccl.order) == 2
    with pytest.raises(RuntimeError, match='group_rank changed'):
        engine._ensure_channel(2)
    assert hccl.hccl_envs == [{}]


def test_ensure_channel_forwards_hccl_env(hccl):
    engine = hccl_engine(hccl.adapter)
    env = {'HCCL_IF_BASE_PORT': '30000'}
    engine.init_transfer_engine(hccl.adapter.TwinkleDeltaHCCLInitInfo(
        dist_ip='10.0.0.1', dist_port=5000, zmq_ip='10.0.0.2', zmq_port=6000, world_size=3, hccl_env=env))
    engine._ensure_channel(1)
    assert hccl.hccl_envs == [env]


def test_ensure_channel_build_failure_closes_sub(hccl, monkeypatch):
    engine = hccl_engine(hccl.adapter)
    engine.init_transfer_engine(init_info(hccl.adapter))
    monkeypatch.setattr(hccl_receiver, 'build_delta_hccl_communicator', Mock(side_effect=OSError('rdzv')))
    with pytest.raises(OSError):
        engine._ensure_channel(1)
    assert hccl.sub.closed and engine._comm is None and engine._manifest_sub is None


def _update(adapter, rank=1):
    return adapter.TwinkleDeltaHCCLUpdateInfo(group_rank=rank)


def test_receive_sparse_session_prepares_sparse_loaders(hccl, monkeypatch):
    utils = sys.modules.get('twinkle.sampler.vllm_sampler.delta.weight_update_utils')
    prepare = Mock()
    drop = Mock(side_effect=lambda model, patches: patches)
    refresh = Mock(return_value=0)
    monkeypatch.setitem(sys.modules, 'twinkle.sampler.vllm_sampler.delta.weight_update_utils',
                        SimpleNamespace(prepare_sparse_weight_loaders=prepare, drop_tied_alias_patches=drop,
                                        refresh_derived_weights=refresh))
    engine = hccl_engine(hccl.adapter)
    engine.init_transfer_engine(init_info(hccl.adapter))
    feed(hccl, [(named_flush(), False), (named_flush(), True)])
    engine.start_weight_update()
    engine.receive_weights(_update(hccl.adapter))
    engine.finish_weight_update()
    loader = sys.modules['vllm.model_executor.model_loader.checkpoint_weight_patch']
    reload = sys.modules['vllm.model_executor.model_loader.reload']
    assert prepare.call_count == 2 and drop.call_count == 2
    refresh.assert_called_once_with(engine.model)  # once per session, after the last flush
    assert loader.load_checkpoint_weight_patches.call_count == 2
    assert loader.load_checkpoint_weight_patches.call_args.kwargs == {'validate_unique_indices': False}
    reload.initialize_layerwise_reload.assert_not_called()
    reload.finalize_layerwise_reload.assert_not_called()
    del utils


def test_receive_dense_session_runs_one_reload_lifecycle(hccl):
    engine = hccl_engine(hccl.adapter)
    engine.init_transfer_engine(init_info(hccl.adapter))
    feed(hccl, [(named_flush('dense'), False), (named_flush('dense'), True)])
    engine.start_weight_update()
    engine.receive_weights(_update(hccl.adapter))
    engine.finish_weight_update()
    reload = sys.modules['vllm.model_executor.model_loader.reload']
    loader = sys.modules['vllm.model_executor.model_loader.checkpoint_weight_patch']
    assert reload.initialize_layerwise_reload.call_count == 1
    assert reload.finalize_layerwise_reload.call_count == 1
    assert loader.load_checkpoint_weight_patches.call_count == 2


def test_receive_mixed_encodings_rejected_and_poisons(hccl):
    engine = hccl_engine(hccl.adapter)
    engine.init_transfer_engine(init_info(hccl.adapter))
    feed(hccl, [(named_flush('dense'), False), (named_flush(), True)])
    engine.start_weight_update()
    with pytest.raises(ValueError, match='cannot mix dense and sparse'):
        engine.receive_weights(_update(hccl.adapter))
    assert engine._update_failed


def test_terminal_only_session_is_noop(hccl):
    engine = hccl_engine(hccl.adapter)
    engine.init_transfer_engine(init_info(hccl.adapter))
    feed_terminal(hccl)
    engine.start_weight_update()
    engine.receive_weights(_update(hccl.adapter))
    engine.finish_weight_update()
    reload = sys.modules['vllm.model_executor.model_loader.reload']
    loader = sys.modules['vllm.model_executor.model_loader.checkpoint_weight_patch']
    loader.load_checkpoint_weight_patches.assert_not_called()
    reload.initialize_layerwise_reload.assert_not_called()
    reload.finalize_layerwise_reload.assert_not_called()
    assert not engine._update_failed


def test_pump_failure_sets_update_failed(hccl):
    engine = hccl_engine(hccl.adapter)
    engine.init_transfer_engine(init_info(hccl.adapter))
    hccl.sub.items.append(('bucket_metadata', {'oops': 1}))
    engine.start_weight_update()
    with pytest.raises(ValueError, match='Malformed delta manifest'):
        engine.receive_weights(_update(hccl.adapter))
    assert engine._update_failed
    with pytest.raises(RuntimeError, match='previous delta update did not complete'):
        engine.start_weight_update()


def test_channel_failure_sets_update_failed(hccl):
    engine = hccl_engine(hccl.adapter)  # init_transfer_engine never ran
    with pytest.raises(RuntimeError, match='init_transfer_engine must run'):
        engine.receive_weights(_update(hccl.adapter))
    assert engine._update_failed


def test_validate_configuration_device_types(adapter):
    def build(cls, device):
        config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1),
                                 model_config=SimpleNamespace(dtype=torch.bfloat16))
        model = SimpleNamespace()
        engine = cls(None, config, SimpleNamespace(type=device), model)
        return engine

    ok = build(adapter.TwinkleDeltaHCCLWeightTransferEngine, 'npu')
    ok._validate_configuration()
    with pytest.raises(NotImplementedError, match='NPU'):
        build(adapter.TwinkleDeltaHCCLWeightTransferEngine, 'cuda')._validate_configuration()
    build(adapter.TwinkleDeltaIPCWeightTransferEngine, 'cuda')._validate_configuration()
    with pytest.raises(NotImplementedError, match='^Twinkle delta weight transfer requires CUDA$'):
        build(adapter.TwinkleDeltaIPCWeightTransferEngine, 'npu')._validate_configuration()
    assert adapter.TwinkleDeltaIPCWeightTransferEngine.required_device_type == 'cuda'
    assert adapter.TwinkleDeltaHCCLWeightTransferEngine.required_device_type == 'npu'


def test_shutdown_destroys_once_and_is_idempotent(hccl):
    engine = hccl_engine(hccl.adapter)
    engine.init_transfer_engine(init_info(hccl.adapter))
    engine._ensure_channel(1)
    engine.shutdown()
    engine.shutdown()
    assert hccl.comm.destroyed == [hccl.comm.comm]
    assert hccl.sub.closed and hccl.sub.close_kwargs == {'linger': 0}
    hccl.sub.context.term.assert_called_once()
    assert (engine._comm, engine._manifest_sub, engine._group_rank, engine._rendezvous) == (None, None, None, None)


def test_register_hccl_engine_once(adapter, monkeypatch):
    registry = []
    factory = SimpleNamespace(register_engine=lambda name, cls: registry.append((name, cls)))
    module = type(sys)('vllm.distributed.weight_transfer')
    module.WeightTransferEngineFactory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter.register_twinkle_delta_hccl_weight_transfer_engine()
    adapter.register_twinkle_delta_hccl_weight_transfer_engine()
    assert registry == [('twinkle_delta_hccl', adapter.TwinkleDeltaHCCLWeightTransferEngine)]


# --------------------------------------------------------------------------- (d) worker extension
@pytest.fixture
def worker_cls(adapter, monkeypatch):
    from twinkle.sampler.vllm_sampler import vllm_worker_extension as ext
    monkeypatch.setitem(sys.modules, 'twinkle.sampler.vllm_sampler.delta.transfer_engine', adapter)
    monkeypatch.setattr(ext, 'set_death_signal', Mock())
    import twinkle.patch.vllm_lora_weights as lora
    monkeypatch.setattr(lora, 'VLLMLoraWeights', lambda: (lambda _: None))
    return ext.TwinkleWorkerExtension


@pytest.mark.parametrize('backend, ipc, hccl_calls', [('twinkle_delta_hccl', 0, 1), ('twinkle_delta_ipc', 1, 0),
                                                       ('other', 0, 0)])
def test_worker_new_registers_by_backend(worker_cls, adapter, monkeypatch, backend, ipc, hccl_calls):
    reg_ipc, reg_hccl = Mock(), Mock()
    monkeypatch.setattr(adapter, 'register_twinkle_delta_weight_transfer_engine', reg_ipc)
    monkeypatch.setattr(adapter, 'register_twinkle_delta_hccl_weight_transfer_engine', reg_hccl)
    vllm_config = SimpleNamespace(weight_transfer_config=SimpleNamespace(backend=backend))
    worker_cls(vllm_config=vllm_config)
    assert (reg_ipc.call_count, reg_hccl.call_count) == (ipc, hccl_calls)


def test_validate_delta_worker_reports_npu_and_names_both_backends(worker_cls, adapter):
    engine = hccl_engine(adapter, 'cpu')
    engine.device = SimpleNamespace(type='npu')
    engine.model_config.hf_config = SimpleNamespace(model_type='llama')
    worker = SimpleNamespace(weight_transfer_engine=engine,
                             model_runner=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2)))
    assert worker_cls.validate_delta_worker(worker) == {'tp': 2, 'device': 'npu'}
    with pytest.raises(RuntimeError, match='weight_transfer_config') as err:
        worker_cls.validate_delta_worker(SimpleNamespace(weight_transfer_engine=object()))
    assert 'twinkle_delta_ipc' in str(err.value) and 'twinkle_delta_hccl' in str(err.value)


def fake_delta_worker(worker_cls, rank=0, dp_index=None, world_size=None):
    """A worker stand-in whose update_weights records its argument; mirrors the bound helper like a real worker."""
    parallel = None if dp_index is None and world_size is None else SimpleNamespace(
        data_parallel_index=dp_index or 0, world_size=world_size or 1)
    worker = SimpleNamespace(update_weights=Mock(), delta_ipc_endpoint=lambda: 'ipc:///x', rank=rank,
                             vllm_config=SimpleNamespace(parallel_config=parallel))
    worker._delta_update_info_for_this_worker = lambda info: worker_cls._delta_update_info_for_this_worker(
        worker, info)
    return worker


def test_update_twinkle_delta_weights_dispatch(worker_cls):
    worker = fake_delta_worker(worker_cls)
    worker_cls.update_twinkle_delta_weights(worker)
    worker.update_weights.assert_called_once_with({'zmq_handle': 'ipc:///x'})
    worker.update_weights.reset_mock()
    info = [{'group_rank': 1}, {'group_rank': 2}]
    worker_cls.update_twinkle_delta_weights(worker, info)
    # a list is reduced to this worker's own entry and handed over as a plain dict (vllm-ascend's
    # NPUWorker.update_weights does not accept lists)
    worker.update_weights.assert_called_once_with({'group_rank': 1})


@pytest.mark.parametrize('rank,expected', [(0, 5), (1, 6), (2, 7)])
def test_update_twinkle_delta_weights_selects_entry_by_worker_rank(worker_cls, rank, expected):
    worker = fake_delta_worker(worker_cls, rank=rank, dp_index=0, world_size=3)
    worker_cls.update_twinkle_delta_weights(worker, [{'group_rank': 5}, {'group_rank': 6}, {'group_rank': 7}])
    worker.update_weights.assert_called_once_with({'group_rank': expected})


def test_update_twinkle_delta_weights_indexes_across_data_parallel_replicas(worker_cls):
    # same formula as vLLM's GPU Worker.update_weights: data_parallel_index * world_size + rank
    worker = fake_delta_worker(worker_cls, rank=1, dp_index=1, world_size=2)
    worker_cls.update_twinkle_delta_weights(worker, [{'group_rank': i} for i in range(4)])
    worker.update_weights.assert_called_once_with({'group_rank': 3})


def test_update_twinkle_delta_weights_rejects_short_list(worker_cls):
    worker = fake_delta_worker(worker_cls, rank=2, dp_index=0, world_size=3)
    with pytest.raises(ValueError, match='needs entry 2'):
        worker_cls.update_twinkle_delta_weights(worker, [{'group_rank': 1}, {'group_rank': 2}])
    worker.update_weights.assert_not_called()


# --------------------------------------------------------------------------- (e) import hygiene
def test_import_without_npu_modules():
    code = ("import sys, twinkle.checkpoint_engine.delta.rollout.hccl_receiver, twinkle.checkpoint_engine.delta.protocol.decode;"
            "print('torch_npu' in sys.modules, 'vllm_ascend' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ['False', 'False']


def test_engine_pieces_do_not_import_npu_modules(adapter):
    engine = hccl_engine(adapter)
    engine.init_transfer_engine(init_info(adapter))
    assert 'torch_npu' not in sys.modules and 'vllm_ascend' not in sys.modules
    assert checksum  # keep import used


# --------------------------------------------------------------------------- (f) re-initialization / shutdown RPC
def test_init_transfer_engine_after_shutdown_rebuilds_against_new_rendezvous(hccl):
    engine = hccl_engine(hccl.adapter)
    engine.init_transfer_engine(init_info(hccl.adapter))
    engine._ensure_channel(1)
    engine.shutdown()
    new = hccl.adapter.TwinkleDeltaHCCLInitInfo(dist_ip='10.0.0.9', dist_port=5001, zmq_ip='10.0.0.8',
                                                zmq_port=6001, world_size=5)
    engine.init_transfer_engine(new)
    engine._ensure_channel(3)
    assert hccl.order[-2:] == [('connect', '10.0.0.8', 6001, 'bucket_metadata'), ('build', '10.0.0.9', 5001, 3, 5)]
    assert hccl.comm.destroyed == [hccl.comm.comm]


@pytest.mark.parametrize('identical', [True, False])
def test_init_transfer_engine_with_live_channel_releases_it_first(hccl, identical):
    engine = hccl_engine(hccl.adapter)
    engine.init_transfer_engine(init_info(hccl.adapter))
    engine._ensure_channel(1)
    new = init_info(hccl.adapter) if identical else hccl.adapter.TwinkleDeltaHCCLInitInfo(
        dist_ip='10.0.0.9', dist_port=5001, zmq_ip='10.0.0.8', zmq_port=6001, world_size=4)
    engine.init_transfer_engine(new)
    # the old communicator and SUB were released, nothing is live until the next receive
    assert hccl.comm.destroyed == [hccl.comm.comm] and hccl.sub.closed
    assert (engine._comm, engine._manifest_sub, engine._group_rank) == (None, None, None)
    assert engine._rendezvous is new
    # a new rendezvous may also use a different group rank
    engine._ensure_channel(2)
    assert hccl.order[-1][3] == 2 and len(hccl.order) == 4


def test_shutdown_delta_weight_transfer_calls_engine_shutdown_for_delta_engines(worker_cls, adapter):
    engine = hccl_engine(adapter, 'cpu')
    engine.shutdown = Mock()
    worker_cls.shutdown_delta_weight_transfer(SimpleNamespace(weight_transfer_engine=engine))
    engine.shutdown.assert_called_once()
    ipc = adapter.TwinkleDeltaIPCWeightTransferEngine.__new__(adapter.TwinkleDeltaIPCWeightTransferEngine)
    ipc.shutdown = Mock()
    worker_cls.shutdown_delta_weight_transfer(SimpleNamespace(weight_transfer_engine=ipc))
    ipc.shutdown.assert_called_once()


def test_shutdown_delta_weight_transfer_is_noop_for_other_engines(worker_cls):
    other = Mock()
    worker_cls.shutdown_delta_weight_transfer(SimpleNamespace(weight_transfer_engine=other))
    other.shutdown.assert_not_called()
    worker_cls.shutdown_delta_weight_transfer(SimpleNamespace(weight_transfer_engine=None))
    worker_cls.shutdown_delta_weight_transfer(SimpleNamespace())
