"""Compare the patched LoRA forward with a dense oracle using real Gloo collectives."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


class CopyToParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad)
        return grad


class ReduceFromParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        value = value.clone()
        dist.all_reduce(value)
        return value

    @staticmethod
    def backward(ctx, grad):
        return grad


class ParallelLinear(nn.Module):
    def __init__(self, weight, mode):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.parallel_mode = mode

    def forward(self, inputs, *args, **kwargs):
        if self.parallel_mode == 'column':
            inputs = CopyToParallel.apply(inputs)
        outputs = nn.functional.linear(inputs, self.weight)
        if self.parallel_mode == 'row':
            outputs = ReduceFromParallel.apply(outputs)
        return outputs, None


class BaseLinear(ParallelLinear):
    def forward(self, inputs, *args, **kwargs):
        return inputs.new_zeros(inputs.shape[0], self.weight.shape[0]), None


class LoraParallelLinear(nn.Module):
    def __init__(self, rank, mode, a, b):
        super().__init__()
        self.sequence_parallel = self.disable_adapters = self.merged = False
        self.config = SimpleNamespace(fp8_param=False)
        self.active_adapters = ['slot']
        shard = slice(rank * 2, rank * 2 + 2)
        self.base_layer = BaseLinear(torch.zeros(4 if mode == 'row' else 2,
                                                 2 if mode == 'row' else 4), mode)
        self.lora_A = nn.ModuleDict({'slot': ParallelLinear(a[:, shard] if mode == 'row' else a,
                                                           'row' if mode == 'row' else None)})
        self.lora_B = nn.ModuleDict({'slot': ParallelLinear(b if mode == 'row' else b[shard],
                                                           None if mode == 'row' else 'column')})
        self.lora_dropout = nn.ModuleDict({'slot': nn.Identity()})


def _worker(rank, rendezvous, mode):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        from twinkle.model.multi_lora import MultiLora

        # Only transport-specific TE classes are replaced; the production patch and
        # CPU collective/autograd operations below run normally in two processes.
        bridge = ModuleType('mcore_bridge')
        bridge.LoraParallelLinear = LoraParallelLinear
        sys.modules['mcore_bridge'] = bridge
        te = ModuleType('megatron.core.extensions.transformer_engine')
        te.TELinear = ParallelLinear
        te.TEGroupedLinear = type('Grouped', (nn.Module,), {})
        te.TELayerNormColumnParallelLinear = type('LayerNormColumn', (nn.Module,), {})
        sys.modules[te.__name__] = te
        tp = ModuleType('megatron.core.tensor_parallel')
        tp.gather_from_sequence_parallel_region = tp.scatter_to_sequence_parallel_region = lambda value: value
        sys.modules[tp.__name__] = tp
        router = ModuleType('megatron.core.transformer.moe.router')
        router.TopKRouter = type('Router', (nn.Module,), {})
        sys.modules[router.__name__] = router
        torch.manual_seed(42)
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        reference_a, reference_b = nn.Parameter(a[:2].clone()), nn.Parameter(b[:, :2].clone())
        reference_optim = torch.optim.Adam([reference_a, reference_b], lr=0.01)
        layer = LoraParallelLinear(rank, mode, a, b)
        config = SimpleNamespace(r=2, lora_alpha=2, use_rslora=False, target_modules='all-linear')
        tenant = SimpleNamespace(tenant_config=config, adapter_name='slot')
        manager = SimpleNamespace(max_r=4, find_lora=lambda name: tenant, match_target_modules=lambda *args: True)
        MultiLora._patch_lora_forward(manager, 'linear', layer)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        inputs, target = torch.randn(2, 4), torch.randn(2, 4)
        shard = slice(rank * 2, rank * 2 + 2)
        for _ in range(3):
            expected = nn.functional.linear(nn.functional.linear(inputs, reference_a), reference_b)
            actual, _ = layer(inputs[:, shard] if mode == 'row' else inputs)
            torch.testing.assert_close(actual, expected if mode == 'row' else expected[:, shard])
            (expected - target).square().sum().backward()
            (actual - (target if mode == 'row' else target[:, shard])).square().sum().backward()
            expected_a = reference_a.grad[:, shard] if mode == 'row' else reference_a.grad
            expected_b = reference_b.grad if mode == 'row' else reference_b.grad[shard]
            torch.testing.assert_close(layer.lora_A['slot'].weight.grad[:2], expected_a)
            torch.testing.assert_close(layer.lora_B['slot'].weight.grad[:, :2], expected_b)
            assert not torch.count_nonzero(layer.lora_A['slot'].weight.grad[2:])
            assert not torch.count_nonzero(layer.lora_B['slot'].weight.grad[:, 2:])
            reference_optim.step()
            optimizer.step()
            reference_optim.zero_grad()
            optimizer.zero_grad()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['row', 'column'])
def test_multi_lora_preserves_tensor_parallel_forward_and_gradients(tmp_path, mode):
    rendezvous = (tmp_path / 'gloo').as_uri()
    mp.spawn(_worker, args=(rendezvous, mode), nprocs=2, join=True)
