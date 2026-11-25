"""A runtime weight update must refresh parameters the backend derives from source weights.

vllm-ascend keeps its GDN convolutions in a kernel layout (``conv1d.ascend_conv1d_weight``) and only
refreshes that copy from ``conv1d.weight`` through ``_pack_conv_weights``, which it wires to
``process_weights_after_loading`` -- i.e. model build time. Training-side weight sync only writes the
source tensor, so without an explicit repack every hybrid layer keeps convolving with the random
weights it was built with while the rest of the model matches the trainer exactly.
"""
import pytest
import torch
from torch import nn

pytest.importorskip('vllm')

from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension  # noqa: E402


class PackedConv(nn.Module):
    """Stand-in for vllm-ascend's conv1d, which owns a packed copy of ``weight``."""

    def __init__(self, channels=3, width=4):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, 1, width))
        self.register_parameter(
            'ascend_conv1d_weight',
            nn.Parameter(torch.full((width, channels), float('nan')), requires_grad=False))

    def pack(self):
        with torch.no_grad():
            self.ascend_conv1d_weight.copy_(self.weight.view(self.weight.size(0), self.weight.size(2)).T)


class ConvolutionalLayer(nn.Module):
    """Stand-in for ``AscendGatedDeltaNetAttention``: the hook lives on the attention module."""

    def __init__(self):
        super().__init__()
        self.conv1d = PackedConv()

    def _pack_conv_weights(self):
        self.conv1d.pack()


class HybridModel(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.model = nn.ModuleDict({'layers': nn.ModuleList([ConvolutionalLayer() for _ in range(layers)])})
        self.plain = nn.Linear(2, 2)

    def load_weights(self, weights):
        by_name = dict(weights)
        for index, layer in enumerate(self.model['layers']):
            layer.conv1d.weight.data.copy_(by_name[f'conv{index}'])
        self.plain.weight.data.copy_(by_name['plain'])
        self.loaded = len(weights)
        return set(by_name)


def test_repack_follows_a_source_weight_update():
    model = HybridModel()
    convs = [layer.conv1d for layer in model.model['layers']]
    assert all(conv.ascend_conv1d_weight.isnan().all() for conv in convs)

    weights = [(f'conv{index}', torch.arange(12, dtype=torch.float32).reshape(3, 1, 4) + index)
               for index in range(3)]
    weights.append(('plain', torch.ones(2, 2)))
    model.load_weights(weights)
    assert TwinkleWorkerExtension._repack_derived_weights(model) == 3
    for index, conv in enumerate(convs):
        assert torch.equal(conv.ascend_conv1d_weight, conv.weight.view(3, 4).T)
        assert not conv.ascend_conv1d_weight.isnan().any()


def test_repack_is_a_noop_for_backends_without_the_hook():
    model = nn.Sequential(nn.Linear(4, 2), nn.LayerNorm(2))
    assert TwinkleWorkerExtension._repack_derived_weights(model) == 0


def test_a_failing_hook_fails_the_repack():
    """A hook that raises must fail the sync: a stale packed copy emits garbage tokens."""
    class Broken(nn.Module):
        def _pack_conv_weights(self):
            raise RuntimeError('kernel layout changed')

    model = nn.Sequential(Broken(), ConvolutionalLayer())
    layer = model[1]
    layer.conv1d.weight.data.fill_(1.0)

    with pytest.raises(RuntimeError, match='kernel layout changed'):
        TwinkleWorkerExtension._repack_derived_weights(model)


class HooklessLayer(nn.Module):
    """A layer with a packed copy but no private hook: the parameter itself is the contract."""

    def __init__(self):
        super().__init__()
        self.conv1d = PackedConv()


def test_repack_uses_the_packed_parameter_when_the_backend_has_no_hook():
    """A renamed or dropped ``_pack_conv_weights`` must be repaired, not skipped silently."""
    model = nn.Sequential(HooklessLayer(), HooklessLayer())
    for layer in model:
        layer.conv1d.weight.data.fill_(2.0)

    assert TwinkleWorkerExtension._repack_derived_weights(model) == 2
    for layer in model:
        assert torch.equal(layer.conv1d.ascend_conv1d_weight, torch.full((4, 3), 2.0))


def test_repack_rejects_a_packed_layout_it_does_not_understand():
    """An unrecognised packed shape must fail the sync rather than leave the kernel on stale data."""
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.zeros(3, 1, 4))
    layer.register_parameter('ascend_conv1d_weight', nn.Parameter(torch.zeros(7, 5)))

    with pytest.raises(RuntimeError, match='unsupported packed conv layout'):
        TwinkleWorkerExtension._repack_derived_weights(layer)


def _worker(model, order):
    worker = object.__new__(TwinkleWorkerExtension)
    worker.rank = worker.local_rank = 0
    worker.device = torch.device('cpu')
    worker.model_runner = nn.Module()
    worker.model_runner.model = model
    for layer in model.model['layers']:
        pack = layer._pack_conv_weights
        layer._pack_conv_weights = lambda pack=pack: (order.append('pack'), pack())[1]
    return worker


def test_load_weights_repacks_after_the_source_tensors_land():
    model = HybridModel()
    order = []
    original = model.load_weights
    model.load_weights = lambda weights: (order.append('load'), original(weights))[1]
    worker = _worker(model, order)
    weights = [(f'conv{index}', torch.ones(3, 1, 4)) for index in range(3)] + [('plain', torch.ones(2, 2))]

    TwinkleWorkerExtension._load_weights(worker, weights, peft_config=None, base_sync_done=False)

    assert order == ['load', 'pack', 'pack', 'pack']
    for layer in model.model['layers']:
        assert not layer.conv1d.ascend_conv1d_weight.isnan().any()
