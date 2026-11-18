# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ascend expert runtime tensors have a different layout from checkpoint tensors."""
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


class UnquantizedFusedMoEMethod:
    pass


class Experts(torch.nn.Module):
    def __init__(self, tp_rank):
        super().__init__()
        self.tp_rank = tp_rank
        self.quant_method = UnquantizedFusedMoEMethod()
        # Qwen3.5-35B-A3B: H=2048, I=512, TP=4. Ascend transposes both matrices.
        self.w13_weight = torch.nn.Parameter(torch.zeros(1, 2048, 256), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.zeros(1, 128, 2048), requires_grad=False)

    def load_weight(self, name, weight):
        pending = getattr(self, '_pending', None)
        if pending is not None:
            pending.append((name, weight))
            return
        if name == 'gate_up_proj':
            for index, shard in enumerate(weight.chunk(2, dim=1)):
                destination = self.w13_weight.data[0]
                half = destination.shape[0] // 2
                destination = destination.narrow(0, index * half, half)
                source = shard[0].narrow(0, self.tp_rank * 128, 128)
                destination[:source.shape[0]].copy_(source)
        else:
            source = weight[:, :, self.tp_rank * 128:(self.tp_rank + 1) * 128]
            self.w2_weight.data.copy_(source)


class Model(torch.nn.Module):
    def __init__(self, tp_rank):
        super().__init__()
        self.experts = Experts(tp_rank)

    def load_weights(self, weights):
        for name, weight in weights:
            self.experts.load_weight(name, weight)


@pytest.fixture
def reload_api(monkeypatch):
    """Model the native reload contract with CPU tensors and deferred bucket reads."""
    from twinkle.sampler.vllm_sampler import vllm_worker_extension as extension

    def initialize(layer):
        layer._live = (layer.w13_weight, layer.w2_weight)
        layer.w13_weight = torch.nn.Parameter(torch.empty(1, 256, 2048), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.empty(1, 2048, 128), requires_grad=False)
        layer._pending = []

    def finalize(layer, model_config):
        pending = layer._pending
        del layer._pending
        for name, tensor in pending:
            layer.load_weight(name, tensor)
        w13, w2 = layer._live
        w13.data.copy_(layer.w13_weight.transpose(1, 2))
        w2.data.copy_(layer.w2_weight.transpose(1, 2))
        layer.w13_weight, layer.w2_weight = w13, w2

    api = ModuleType('vllm.model_executor.model_loader.reload')
    api.initialize_layerwise_reload = Mock(side_effect=initialize)
    api.finalize_layerwise_reload = Mock(side_effect=finalize)
    method = ModuleType('vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method')
    method.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod
    config = ModuleType('vllm.config')
    config.set_current_vllm_config = lambda *args: nullcontext()
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setitem(sys.modules, method.__name__, method)
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.setattr(extension.Torch, 'synchronize', lambda *args: None)
    monkeypatch.setattr(extension.Torch, 'empty_cache', lambda: None)
    monkeypatch.setattr(extension.Torch, 'ipc_collect', lambda: None)
    monkeypatch.setattr('twinkle.patch.vllm_weight_loader.patch_lora_weight_loader', lambda: None)
    return extension, api


def worker_for(extension, tp_rank=0):
    worker = object.__new__(extension.TwinkleWorkerExtension)
    worker.device = NS(type='npu', index=0)
    worker.rank = worker.local_rank = 0
    worker.model_runner = NS(model=Model(tp_rank), model_config=NS(), vllm_config=NS(),
                             parallel_config=NS(tensor_parallel_size=1))
    return worker


def assert_loaded(worker, weights, live):
    experts = worker.model_runner.model.experts
    gate, up = weights['gate_up_proj'].chunk(2, dim=1)
    start = experts.tp_rank * 128
    w13 = torch.cat((gate[:, start:start + 128], up[:, start:start + 128]), dim=1).transpose(1, 2)
    w2 = weights['down_proj'][:, :, start:start + 128].transpose(1, 2)
    assert torch.equal(experts.w13_weight, w13)
    assert torch.equal(experts.w2_weight, w2)
    assert experts.w13_weight is live[0] and experts.w2_weight is live[1]
    assert not worker._moe_reload_active


@pytest.mark.parametrize('tp_rank', range(4))
def test_moe_sync_restores_checkpoint_layout_and_preserves_runtime_storage(reload_api, tp_rank):
    extension, api = reload_api
    worker = worker_for(extension, tp_rank)
    experts = worker.model_runner.model.experts
    live = (experts.w13_weight, experts.w2_weight)
    for step in range(2):
        weights = {'gate_up_proj': torch.randn(1, 1024, 2048), 'down_proj': torch.randn(1, 2048, 512)}
        # The unbracketed loader reproduces the reported [128, 256] vs [128, 2048] mismatch.
        with pytest.raises(RuntimeError, match='must match'):
            experts.load_weight('gate_up_proj', weights['gate_up_proj'])
        worker.load_synced_weights(weights)
        assert_loaded(worker, weights, live)
        assert api.initialize_layerwise_reload.call_count == step + 1
        assert api.finalize_layerwise_reload.call_count == step + 1


@pytest.mark.parametrize('direct_handle', [False, True])
def test_streamed_reload_retains_weights_across_reused_buckets(reload_api, monkeypatch, direct_handle):
    extension, api = reload_api
    worker = worker_for(extension)
    experts = worker.model_runner.model.experts
    live = (experts.w13_weight, experts.w2_weight)
    weights = {'gate_up_proj': torch.randn(1, 1024, 2048), 'down_proj': torch.randn(1, 2048, 512)}
    buffer = torch.empty(weights['gate_up_proj'].numel() * 4, dtype=torch.uint8)
    metadata = [None]
    for index, (name, tensor) in enumerate(weights.items()):
        entry = dict(name=name, dtype=tensor.dtype, shape=tensor.shape, offset=0)
        if direct_handle:
            entry['handle'] = name
        metadata.append(dict(bucket_meta=[entry], is_last=index == len(weights) - 1))
    messages = iter(metadata)

    def receive(*args):
        message = next(messages)
        if message is None:
            return 'buffer'
        name = message['bucket_meta'][0]['name']
        tensor = weights[name]
        buffer[:tensor.numel() * 4].copy_(tensor.view(torch.uint8).flatten())
        return message

    def rebuild(handle, device):
        if handle == 'buffer':
            return buffer
        tensor = weights[handle]
        return buffer[:tensor.numel() * 4].view(tensor.dtype).view(tensor.shape)

    socket = NS(connect=lambda *args: None, close=lambda: None,
                send=lambda *args: buffer.fill_(0))
    worker._zmq_ctx = NS(socket=lambda *args: socket)
    monkeypatch.setattr(extension, '_rebuild_ipc', rebuild)
    monkeypatch.setattr(extension, 'configure_zmq_socket', lambda *args, **kwargs: None)
    monkeypatch.setattr(extension.Platform, 'get_platform', lambda: NS(ensure_ipc_reducer=lambda: None))
    monkeypatch.setattr('twinkle.checkpoint_engine.zmq_auth.recv_auth_pyobj', receive)
    worker.update_weights_from_ipc(zmq_handle='ipc:///unused', independent=True)
    assert_loaded(worker, weights, live)
    assert api.initialize_layerwise_reload.call_count == api.finalize_layerwise_reload.call_count == 1


@pytest.mark.parametrize('mode', ['cuda', 'cpu', 'adapter', 'quantized', 'dense'])
def test_reload_leaves_existing_non_moe_and_adapter_paths_unchanged(reload_api, mode):
    extension, api = reload_api
    worker = worker_for(extension)
    worker._load_weights = Mock()
    if mode in ('cuda', 'cpu'):
        worker.device.type = mode
    elif mode == 'quantized':
        worker._quant_reload_state = object()
    elif mode == 'dense':
        worker.model_runner.model = torch.nn.Linear(2, 2)
    kwargs = dict(peft_config={'r': 8}, base_sync_done=True) if mode == 'adapter' else {}
    worker.load_synced_weights({}, **kwargs)
    api.initialize_layerwise_reload.assert_not_called()
    api.finalize_layerwise_reload.assert_not_called()


def test_failed_receive_preserves_original_error(reload_api):
    extension, api = reload_api
    worker = worker_for(extension)
    worker._receive_weights_from_ipc = Mock(side_effect=ValueError('receive failed'))
    with pytest.raises(ValueError, match='receive failed'):
        worker.update_weights_from_ipc()
    assert not worker._moe_reload_active
    api.finalize_layerwise_reload.assert_not_called()
