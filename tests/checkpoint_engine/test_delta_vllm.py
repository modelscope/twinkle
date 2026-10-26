# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM weight-transfer adapter: flush decoding, native patch loading, tied weights and Mamba sparse loaders."""
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from twinkle.checkpoint_engine.delta.protocol.wire import checksum


@pytest.fixture
def adapter(monkeypatch):
    @dataclass
    class CheckpointWeightPatch:
        name: str
        shape: tuple
        dtype: torch.dtype
        values: torch.Tensor
        indices: torch.Tensor | None = None

    class WeightTransferEngine:
        def __init__(self, config, vllm_config, device, model):
            self.model = model
            self.vllm_config = vllm_config
            self.device = device
            self.parallel_config = vllm_config.parallel_config
            self.model_config = vllm_config.model_config

    modules = {
        'vllm': {'__version__': 'test'},
        'vllm.config.weight_transfer': {'WeightTransferConfig': type('WeightTransferConfig', (), {})},
        'vllm.distributed.weight_transfer.base': {
            'WeightTransferEngine': WeightTransferEngine,
            'WeightTransferInitInfo': type('WeightTransferInitInfo', (), {}),
            'WeightTransferUpdateInfo': type('WeightTransferUpdateInfo', (), {}),
        },
        'vllm.model_executor.model_loader.checkpoint_weight_patch': {
            'CheckpointWeightPatch': CheckpointWeightPatch,
            'load_checkpoint_weight_patches': Mock(),
        },
        'vllm.model_executor.layers.vocab_parallel_embedding': {
            'VocabParallelEmbedding': type('VocabParallelEmbedding', (), {}),
        },
        'vllm.model_executor.model_loader.reload': {
            'initialize_layerwise_reload': Mock(),
            'finalize_layerwise_reload': Mock(),
        },
    }
    for name, fields in modules.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / 'src/twinkle/sampler/vllm_sampler/delta/transfer_engine.py'
    spec = importlib.util.spec_from_file_location('_twinkle_delta_adapter_test', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def named_flush(encoding='indices'):
    values = torch.tensor([-0., 2.], dtype=torch.bfloat16)
    positions = torch.tensor([0, 3], dtype=torch.int32).view(torch.uint8)
    if encoding == 'dense':
        positions = torch.empty(0, dtype=torch.uint8)
    params = [dict(name='w', dtype='bfloat16', shape=[4 if encoding == 'indices' else 2],
                   pos_start=0, pos_end=positions.numel(), pos_width=4, val_start=0, val_end=2)]
    spec = dict(encoding=encoding, params=params, checksum=checksum(positions, values))
    spec_tensor = torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8)
    return [('__delta_spec__', spec_tensor), ('__positions__', positions), ('__values__', values)]


def transfer_engine(adapter):
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1),
                             model_config=SimpleNamespace(dtype=torch.bfloat16))
    model = SimpleNamespace(named_modules=lambda **_: [], named_parameters=lambda: [])
    return adapter.TwinkleDeltaIPCWeightTransferEngine(None, config, torch.device('cuda'), model)


def test_decoder_consumes_verl_flush_without_custom_envelope(adapter):
    encoding, patches = adapter.decode_delta_payload(named_flush())
    assert (encoding, patches[0].name, patches[0].shape, patches[0].indices.tolist(),
            patches[0].values.view(torch.int16).tolist()) == ('indices', 'w', (4,), [0, 3], [-32768, 16384])


def test_corrupt_flush_is_rejected_before_loading(adapter):
    flush = named_flush()
    flush[-1][1][0] = 42
    with pytest.raises(RuntimeError, match='checksum'):
        adapter.decode_delta_payload(flush)


def test_dense_update_runs_one_reload_lifecycle(adapter):
    engine = transfer_engine(adapter)
    engine._receive_payload = Mock(return_value=named_flush('dense'))
    engine.start_weight_update()
    update = adapter.TwinkleDeltaIPCUpdateInfo(zmq_handle='ipc:///tmp/test.sock')
    engine.receive_weights(update)
    engine.receive_weights(update)
    engine.finish_weight_update()
    reload = sys.modules['vllm.model_executor.model_loader.reload']
    assert (reload.initialize_layerwise_reload.call_count, reload.finalize_layerwise_reload.call_count) == (1, 1)


def test_sparse_update_uses_native_patch_loader_without_reload(adapter):
    engine = transfer_engine(adapter)
    engine._receive_payload = Mock(return_value=named_flush())
    engine.start_weight_update()
    engine.receive_weights(adapter.TwinkleDeltaIPCUpdateInfo(zmq_handle='ipc:///tmp/test.sock'))
    engine.finish_weight_update()
    loader = sys.modules['vllm.model_executor.model_loader.checkpoint_weight_patch']
    assert loader.load_checkpoint_weight_patches.call_args.kwargs == {'validate_unique_indices': False}
    reload = sys.modules['vllm.model_executor.model_loader.reload']
    reload.initialize_layerwise_reload.assert_not_called()
    reload.finalize_layerwise_reload.assert_not_called()


def ascend_gdn_model(dim=6, width=4):
    """Mimic vllm-ascend ops/gdn.py: conv1d.weight [D, 1, W] plus a packed [W, D] copy."""
    conv1d = torch.nn.Module()
    conv1d.weight = torch.nn.Parameter(torch.randn(dim, 1, width).to(torch.bfloat16), requires_grad=False)
    packed = conv1d.weight.view(dim, width).transpose(0, 1).contiguous()
    conv1d.register_parameter('ascend_conv1d_weight', torch.nn.Parameter(packed, requires_grad=False))
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].conv1d = conv1d
    return model, conv1d


def expected_packed(conv1d):
    weight = conv1d.weight
    return weight.view(weight.size(0), weight.size(2)).transpose(0, 1).contiguous()


@pytest.mark.parametrize('encoding, refreshed', [('indices', True), ('dense', False)])
def test_sparse_update_refreshes_ascend_packed_conv_weight(adapter, encoding, refreshed):
    model, conv1d = ascend_gdn_model()
    stale = conv1d.ascend_conv1d_weight.detach().clone()
    loader = sys.modules['vllm.model_executor.model_loader.checkpoint_weight_patch']

    def write_source(*_, **__):
        # The checkpoint patch API only writes the source parameter.
        flat = conv1d.weight.data.view(-1)
        flat[[1, 9]] = -flat[[1, 9]] + 1

    loader.load_checkpoint_weight_patches.side_effect = write_source
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1),
                             model_config=SimpleNamespace(dtype=torch.bfloat16))
    engine = adapter.TwinkleDeltaIPCWeightTransferEngine(None, config, torch.device('cuda'), model)
    engine._receive_payload = Mock(return_value=named_flush(encoding))
    engine.start_weight_update()
    engine.receive_weights(adapter.TwinkleDeltaIPCUpdateInfo(zmq_handle='ipc:///tmp/test.sock'))
    assert not torch.equal(stale, expected_packed(conv1d))  # the source really changed
    packed_ptr = conv1d.ascend_conv1d_weight.data_ptr()
    engine.finish_weight_update()
    assert conv1d.ascend_conv1d_weight.data_ptr() == packed_ptr  # refreshed in place
    if refreshed:
        assert torch.equal(conv1d.ascend_conv1d_weight.view(torch.int16), expected_packed(conv1d).view(torch.int16))
    else:  # dense: finalize_layerwise_reload (mocked here) owns the repack
        assert torch.equal(conv1d.ascend_conv1d_weight, stale)


def test_refresh_derived_weights_rejects_unknown_packed_layout(adapter):
    from twinkle.sampler.vllm_sampler.delta.weight_update_utils import refresh_derived_weights
    model, conv1d = ascend_gdn_model()
    conv1d.ascend_conv1d_weight = torch.nn.Parameter(torch.zeros(6, 4, dtype=torch.bfloat16), requires_grad=False)
    with pytest.raises(RuntimeError, match='unsupported packed conv layout'):
        refresh_derived_weights(model)


def test_refresh_derived_weights_is_noop_without_packed_weights(adapter):
    from twinkle.sampler.vllm_sampler.delta.weight_update_utils import refresh_derived_weights
    assert refresh_derived_weights(torch.nn.Linear(2, 2)) == 0


def test_partial_update_failure_poisoned_across_sessions(adapter):
    engine = transfer_engine(adapter)
    engine._receive_payload = Mock(return_value=named_flush())
    loader = sys.modules['vllm.model_executor.model_loader.checkpoint_weight_patch']
    loader.load_checkpoint_weight_patches.side_effect = ValueError('partial write')
    engine.start_weight_update()
    with pytest.raises(ValueError, match='partial write'):
        engine.receive_weights(adapter.TwinkleDeltaIPCUpdateInfo(zmq_handle='ipc:///tmp/test.sock'))
    with pytest.raises(RuntimeError, match='restart the full job'):
        engine.start_weight_update()


def test_worker_requires_native_delta_backend(adapter, monkeypatch):
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension
    monkeypatch.setitem(sys.modules, 'twinkle.sampler.vllm_sampler.delta.transfer_engine', adapter)
    worker = SimpleNamespace(weight_transfer_engine=None)
    with pytest.raises(RuntimeError, match='weight_transfer_config'):
        TwinkleWorkerExtension.validate_delta_worker(worker)


def test_worker_allows_dense_model_and_tp_beyond_previous_allowlist(adapter, monkeypatch):
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension
    monkeypatch.setitem(sys.modules, 'twinkle.sampler.vllm_sampler.delta.transfer_engine', adapter)
    engine = transfer_engine(adapter)
    engine.model_config.hf_config = SimpleNamespace(model_type='llama')
    worker = SimpleNamespace(weight_transfer_engine=engine,
                             model_runner=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=4)))
    assert TwinkleWorkerExtension.validate_delta_worker(worker) == {'tp': 4, 'device': 'cuda'}


class Embedding:
    def __init__(self, weight):
        self.weight = weight

    def named_parameters(self, remove_duplicate=False):
        return [('weight', self.weight)]


class CheckpointMapper:
    def apply_list(self, names):
        return [f'language_model.{name}' for name in names]


def load_source(monkeypatch, filename):
    path = Path(__file__).parents[2] / 'src/twinkle/sampler/vllm_sampler' / filename
    spec = importlib.util.spec_from_file_location(f'_delta_test_{path.stem}', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tied_helpers(monkeypatch):
    module = ModuleType('vllm.model_executor.layers.vocab_parallel_embedding')
    module.VocabParallelEmbedding = Embedding
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return load_source(monkeypatch, 'delta/weight_update_utils.py')


@pytest.fixture(params=[False, True], ids=['language-model', 'qwen-multimodal'])
def tied_model(request):
    shared_weight = [1., 2., 3.]
    prefix = 'language_model.' if request.param else ''
    modules = [(f'{prefix}model.embed_tokens', Embedding(shared_weight)),
               (f'{prefix}lm_head', Embedding(shared_weight))]
    model = SimpleNamespace(named_modules=lambda **_: modules, named_parameters=lambda: [], shared_weight=shared_weight)
    if request.param:
        model.hf_to_vllm_mapper = CheckpointMapper()
    return model


def test_separate_alias_bucket_is_dropped(tied_helpers, tied_model):
    alias = SimpleNamespace(name='lm_head.weight', indices=[1], values=[7.])
    assert tied_helpers.drop_tied_alias_patches(tied_model, [alias]) == []


def test_canonical_patch_preserves_checkpoint_name_and_payload(tied_helpers, tied_model):
    canonical = SimpleNamespace(name='model.embed_tokens.weight', indices=[1], values=[7.])
    alias = SimpleNamespace(name='lm_head.weight', indices=[1], values=[7.])
    assert tied_helpers.drop_tied_alias_patches(tied_model, [canonical, alias]) == [canonical]


def test_untied_head_is_preserved(tied_helpers):
    modules = [('model.embed_tokens', Embedding(object())), ('lm_head', Embedding(object()))]
    model = SimpleNamespace(named_modules=lambda **_: modules)
    alias = SimpleNamespace(name='lm_head.weight')
    assert tied_helpers.drop_tied_alias_patches(model, [alias]) == [alias]


def test_shared_non_embedding_parameters_are_preserved(tied_helpers):
    shared = object()
    submodule = SimpleNamespace(named_parameters=lambda **_: [('weight', shared)])
    modules = [('first', submodule), ('second', submodule)]
    model = SimpleNamespace(named_modules=lambda **_: modules)
    patch = SimpleNamespace(name='second.weight')
    assert tied_helpers.drop_tied_alias_patches(model, [patch]) == [patch]


def test_mapper_dropped_name_is_left_to_vllm(tied_helpers, tied_model):
    tied_model.hf_to_vllm_mapper = SimpleNamespace(apply_list=lambda _: [])
    patch = SimpleNamespace(name='unknown.weight')
    assert tied_helpers.drop_tied_alias_patches(tied_model, [patch]) == [patch]


@pytest.fixture
def tied_adapter(monkeypatch, tied_helpers):
    class WeightTransferEngine:
        def __init__(self, config, vllm_config, device, model):
            self.model = model
            self.model_config = vllm_config.model_config

    modules = {
        'torch': {},
        'vllm': {},
        'vllm.config.weight_transfer': {'WeightTransferConfig': object},
        'vllm.distributed.weight_transfer.base': {
            'WeightTransferEngine': WeightTransferEngine,
            'WeightTransferInitInfo': object,
            'WeightTransferUpdateInfo': object,
        },
        'vllm.model_executor.model_loader.checkpoint_weight_patch': {
            'CheckpointWeightPatch': object,
            'load_checkpoint_weight_patches': Mock(),
        },
        'vllm.model_executor.model_loader.reload': {
            'initialize_layerwise_reload': Mock(),
            'finalize_layerwise_reload': Mock(),
        },
    }
    for name, fields in modules.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, 'twinkle.sampler.vllm_sampler.delta.weight_update_utils', tied_helpers)
    return load_source(monkeypatch, 'delta/transfer_engine.py')


@pytest.mark.parametrize('encoding', ['indices', 'dense'])
def test_transfer_applies_shared_weight_across_separate_flushes(tied_adapter, tied_model, monkeypatch, encoding):
    canonical = SimpleNamespace(name='model.embed_tokens.weight', indices=[1], values=[7.])
    alias = SimpleNamespace(name='lm_head.weight', indices=[1], values=[7.])
    config = SimpleNamespace(model_config=object())
    engine = tied_adapter.TwinkleDeltaIPCWeightTransferEngine(None, config, None, tied_model)
    engine._receive_payload = Mock(side_effect=[[canonical], [alias]])
    monkeypatch.setattr(tied_adapter, 'decode_delta_payload', lambda patches: (encoding, patches))

    def load_patches(model, patches, **_):
        names = {patch.name for patch in patches}
        if 'lm_head.weight' in names and 'model.embed_tokens.weight' not in names:
            raise ValueError('tied weight is uninitialized')
        for patch in patches:
            for index, value in zip(patch.indices, patch.values):
                model.shared_weight[index] = value

    loader = sys.modules['vllm.model_executor.model_loader.checkpoint_weight_patch']
    loader.load_checkpoint_weight_patches.side_effect = load_patches
    engine.start_weight_update()
    update = tied_adapter.TwinkleDeltaIPCUpdateInfo(zmq_handle='ipc:///tmp/tied-test.sock')
    engine.receive_weights(update)
    engine.receive_weights(update)
    engine.finish_weight_update()
    assert (tied_model.shared_weight, loader.load_checkpoint_weight_patches.call_args.args[1]) == ([1., 7., 3.], [])


@pytest.mark.parametrize('encoding', ['indices', 'dense'])
def test_transfer_prepares_native_loaders_for_sparse_patches(tied_adapter, tied_model, monkeypatch, encoding):
    tied_helpers = sys.modules['twinkle.sampler.vllm_sampler.delta.weight_update_utils']
    prepare = Mock()
    monkeypatch.setattr(tied_helpers, 'prepare_sparse_weight_loaders', prepare)
    engine = tied_adapter.TwinkleDeltaIPCWeightTransferEngine(None, SimpleNamespace(model_config=object()), None, tied_model)
    engine._receive_payload = Mock(return_value=[])
    monkeypatch.setattr(tied_adapter, 'decode_delta_payload', lambda _: (encoding, []))
    loader = sys.modules['vllm.model_executor.model_loader.checkpoint_weight_patch']

    def verify_preparation(*_, **__):
        assert prepare.call_count == (1 if encoding == 'indices' else 0)

    loader.load_checkpoint_weight_patches.side_effect = verify_preparation
    engine.start_weight_update()
    engine.receive_weights(tied_adapter.TwinkleDeltaIPCUpdateInfo(zmq_handle='ipc:///tmp/tied-test.sock'))
    engine.finish_weight_update()


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    @property
    def data(self):
        return self

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def __setitem__(self, key, source):
        self.values[key] = source.values

    def copy_(self, source):
        self.values[...] = source.values


def native_mamba_loader(shard_spec, tp_size, tp_rank):
    """Reproduce the native slice assignment that bypasses Tensor.copy_."""
    def loader(param, loaded_weight):
        boundary, loaded_boundary = 0, 0
        for full_dim, extra, duplicate_groups in shard_spec:
            shard_size = full_dim // tp_size
            rank = 0 if duplicate_groups else tp_rank
            loaded_skip = rank * shard_size
            loaded_start = loaded_boundary + loaded_skip
            take = min(shard_size, full_dim - extra - loaded_skip)
            param.data[boundary:boundary + take, ...] = loaded_weight[loaded_start:loaded_start + take]
            boundary += shard_size
            loaded_boundary += full_dim - extra

    loader.__module__ = 'vllm.model_executor.layers.mamba.mamba_mixer2'
    loader.__qualname__ = 'mamba_v2_sharded_weight_loader.<locals>.loader'
    return loader


@pytest.fixture
def mamba_helpers():
    path = Path(__file__).parents[2] / 'src/twinkle/sampler/vllm_sampler/delta/weight_update_utils.py'
    spec = importlib.util.spec_from_file_location('_delta_mamba_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def masked_copy(monkeypatch):
    def copy_non_nan(destination, source):
        destination.values[...] = np.where(np.isnan(source.values), destination.values, source.values)

    monkeypatch.setattr(Tensor, 'copy_', copy_non_nan)


def model_for(parameter):
    return SimpleNamespace(named_parameters=lambda: [('conv1d.weight', parameter)])


@pytest.mark.parametrize('tp_rank', [0, 1])
@pytest.mark.parametrize('shard_spec', [
    [(4, 0, False), (4, 0, False), (4, 0, False)],
    [(4, 0, False), (4, 0, True), (4, 1, False)],
])
def test_sparse_updates_match_dense_tp_load(mamba_helpers, masked_copy, tp_rank, shard_spec):
    checkpoint_size = sum(full_dim - extra for full_dim, extra, _ in shard_spec)
    parameter_size = sum(full_dim // 2 for full_dim, _, _ in shard_spec)
    checkpoint = Tensor(np.arange(checkpoint_size * 2).reshape(checkpoint_size, 1, 2))
    native = native_mamba_loader(shard_spec, 2, tp_rank)
    parameter = Tensor(np.full((parameter_size, 1, 2), -1.))
    native(parameter, checkpoint)
    expected = Tensor(parameter.values.copy())
    changed = np.arange(checkpoint_size) % 2 == 0
    updated = Tensor(checkpoint.values.copy())
    updated.values[changed] += 100.
    native(expected, updated)
    patch = Tensor(np.full_like(checkpoint.values, np.nan))
    patch.values[changed] = updated.values[changed]
    parameter.weight_loader = native
    mamba_helpers.prepare_sparse_weight_loaders(model_for(parameter))
    parameter.weight_loader(parameter, patch)
    np.testing.assert_array_equal(parameter.values, expected.values)


def test_dense_updates_keep_native_geometry(mamba_helpers):
    native = native_mamba_loader([(4, 0, False), (4, 0, True), (4, 1, False)], 2, 1)
    checkpoint = Tensor(np.arange(22).reshape(11, 1, 2))
    expected = Tensor(np.full((6, 1, 2), -1.))
    native(expected, checkpoint)
    parameter = Tensor(np.full((6, 1, 2), -1.))
    parameter.weight_loader = native
    mamba_helpers.prepare_sparse_weight_loaders(model_for(parameter))
    parameter.weight_loader(parameter, checkpoint)
    np.testing.assert_array_equal(parameter.values, expected.values)


def test_loader_adaptation_is_idempotent(mamba_helpers):
    parameter = Tensor([0.])
    parameter.weight_loader = native_mamba_loader([(1, 0, False)], 1, 0)
    model = model_for(parameter)
    mamba_helpers.prepare_sparse_weight_loaders(model)
    adapted = parameter.weight_loader
    mamba_helpers.prepare_sparse_weight_loaders(model)
    assert parameter.weight_loader is adapted


def test_unrelated_loaders_are_unchanged(mamba_helpers):
    original = native_mamba_loader([(1, 0, False)], 1, 0)
    original.__module__ = 'other_backend'
    parameter = Tensor([0.])
    parameter.weight_loader = original
    mamba_helpers.prepare_sparse_weight_loaders(model_for(parameter))
    assert parameter.weight_loader is original


def test_unknown_native_geometry_is_rejected(mamba_helpers):
    def loader(param, loaded_weight):
        pass

    loader.__module__ = 'vllm.model_executor.layers.mamba.mamba_mixer2'
    loader.__qualname__ = 'mamba_v2_sharded_weight_loader.<locals>.loader'
    parameter = Tensor([0.])
    parameter.weight_loader = loader
    with pytest.raises(RuntimeError, match='cannot safely apply sparse'):
        mamba_helpers.prepare_sparse_weight_loaders(model_for(parameter))


def test_sparse_conv_patch_preserves_torch_runtime_values(mamba_helpers, monkeypatch):
    torch = pytest.importorskip('torch')
    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(6, 1, 2))
    parameter.weight_loader = native_mamba_loader([(2, 0, False)] * 3, 1, 0)
    patch = torch.full_like(parameter, float('nan'))
    patch[2, 0, 1] = 99.
    expected = parameter.detach().clone()
    expected[2, 0, 1] = 99.

    def copy_non_nan(destination, source):
        return torch.where(torch.isnan(source), destination, source, out=destination)

    monkeypatch.setattr(torch.Tensor, 'copy_', copy_non_nan)
    mamba_helpers.prepare_sparse_weight_loaders(model_for(parameter))
    parameter.weight_loader(parameter, patch)
    assert torch.equal(parameter, expected)
