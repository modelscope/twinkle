"""Offloading a Megatron model off the device, and getting it back unchanged.

Colocation runs a sampler on the trainer's GPU, so between steps the trainer has to hand the memory
back. What makes this worth testing is where Megatron actually keeps things. With a distributed
optimizer the parameters are pooled into a flat buffer and the module only holds views, so an
implementation that moves ``param.data`` reports success and frees nothing. Without one, Megatron
pools gradients but leaves ``param_data`` as None and the parameters stay on the module -- so code
that treats "has buffers" as "buffers hold everything" silently offloads gradients only. Both
arrangements are covered here, and the assertions are on measured device memory for that reason.

Requires a GPU: freeing buffer storage and refilling it from pinned host memory has no meaningful
CPU equivalent.
"""
from types import SimpleNamespace
import os

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='offload frees device memory')

PARAM_NUMEL = 1 << 22  # ~4M bf16 parameters -> 8 MiB, big enough to see against allocator noise


class Tiny(nn.Module):
    """A stand-in for a model chunk: one trainable parameter and one frozen one.

    The frozen one matters -- Megatron's DDP keeps parameters that never need a gradient out of its
    buckets, so they only move if the offload walks the module too.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trainable = nn.Parameter(torch.randn(PARAM_NUMEL, dtype=torch.bfloat16))
        self.frozen = nn.Parameter(torch.randn(PARAM_NUMEL // 4, dtype=torch.bfloat16), requires_grad=False)


@pytest.fixture(scope='module')
def megatron_single_rank():
    """A one-rank Megatron world, which is all these buffer mechanics need."""
    import torch.distributed as dist
    from megatron.core import parallel_state

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29713')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
    yield
    if created:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def transformer_config():
    from megatron.core.transformer.transformer_config import TransformerConfig
    return TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1, bf16=True)


@pytest.fixture
def make_chunk(megatron_single_rank):
    """Wrap a :class:`Tiny` in Megatron's DDP, with or without a distributed optimizer."""
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed import DistributedDataParallelConfig

    chunks = []

    def _make(distributed_optimizer: bool):
        config = transformer_config()
        chunk = MegatronDDP(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                use_distributed_optimizer=distributed_optimizer,
            ),
            module=Tiny(config).cuda(),
        )
        chunks.append(chunk)
        return chunk

    yield _make
    chunks.clear()
    torch.cuda.empty_cache()


@pytest.fixture
def strategy():
    """Only the offload methods are exercised, so skip the constructor and its parallel config."""
    from twinkle.model.megatron.strategy.megatron import MegatronStrategy
    return object.__new__(MegatronStrategy)


def allocated_mib():
    return torch.cuda.memory_allocated() / 2**20


@pytest.mark.parametrize('distributed_optimizer', [True, False])
def test_where_megatron_keeps_the_parameters(strategy, make_chunk, distributed_optimizer):
    """Pins down the premise the rest of this file rests on, in both arrangements.

    Without this, a change to Megatron's buffer layout would make the tests below pass vacuously
    instead of failing.
    """
    buffers = strategy._flat_buffers(make_chunk(distributed_optimizer))
    assert buffers, 'MegatronDDP exposed no flat buffers at all'
    pooled = [b.param_data is not None for b in buffers]
    assert all(pooled) if distributed_optimizer else not any(pooled)


@pytest.mark.parametrize('distributed_optimizer', [True, False])
def test_offload_actually_returns_device_memory(strategy, make_chunk, distributed_optimizer):
    """The whole point. Relocating views instead of freeing storage fails here."""
    chunk = make_chunk(distributed_optimizer)
    before = allocated_mib()
    strategy.offload_to_cpu([chunk])
    freed = before - allocated_mib()
    # The trainable parameter alone is 8 MiB. Require most of it rather than an exact figure: the
    # buffers are padded and the allocator reports in blocks.
    assert freed > 6, f'offload freed only {freed:.2f} MiB of a >8 MiB model'


@pytest.mark.parametrize('distributed_optimizer', [True, False])
def test_round_trip_is_bit_exact(strategy, make_chunk, distributed_optimizer):
    """Offload is a detour, not a transformation: bf16 leaves a sloppy copy nowhere to hide."""
    chunk = make_chunk(distributed_optimizer)
    trainable = chunk.module.trainable.detach().clone()
    frozen = chunk.module.frozen.detach().clone()

    strategy.offload_to_cpu([chunk])
    strategy.reload_to_gpu([chunk])

    assert torch.equal(trainable, chunk.module.trainable.detach())
    assert torch.equal(frozen, chunk.module.frozen.detach())
    assert chunk.module.trainable.device.type == 'cuda'
    assert chunk.module.frozen.device.type == 'cuda'


def test_the_frozen_parameter_leaves_the_device_too(strategy, make_chunk):
    """It sits outside the buckets, so it only moves if the offload walks the module as well."""
    chunk = make_chunk(distributed_optimizer=True)
    strategy.offload_to_cpu([chunk])
    assert chunk.module.frozen.device.type == 'cpu'
    strategy.reload_to_gpu([chunk])
    assert chunk.module.frozen.device.type == 'cuda'


def test_a_gradient_only_bucket_does_not_pass_for_full_coverage(strategy, make_chunk):
    """Without a distributed optimizer the buckets hold gradients and nothing else, so the trainable
    parameter is still on the module -- and has to be moved by hand. Reading "buffers exist" as
    "buffers hold the parameters" leaves 8 MiB resident and still reports success."""
    chunk = make_chunk(distributed_optimizer=False)
    strategy.offload_to_cpu([chunk])
    assert chunk.module.trainable.device.type == 'cpu'
    strategy.reload_to_gpu([chunk])
    assert chunk.module.trainable.device.type == 'cuda'


@pytest.mark.parametrize('distributed_optimizer', [True, False])
def test_both_directions_are_idempotent(strategy, make_chunk, distributed_optimizer):
    """Callers bracket a training step with these, so a repeated call must not corrupt or realloc.

    A second host stash would quietly reintroduce the double copy that offloading exists to avoid,
    and would not surface as a wrong answer.
    """
    chunk = make_chunk(distributed_optimizer)
    reference = chunk.module.trainable.detach().clone()

    strategy.offload_to_cpu([chunk])
    once = allocated_mib()
    stashes = [b.param_data_cpu for b in chunk.buffers]
    strategy.offload_to_cpu([chunk])
    assert allocated_mib() == pytest.approx(once, abs=0.01)
    assert [b.param_data_cpu for b in chunk.buffers] == stashes, 'a host stash was reallocated'

    strategy.reload_to_gpu([chunk])
    twice = allocated_mib()
    strategy.reload_to_gpu([chunk])
    assert allocated_mib() == pytest.approx(twice, abs=0.01)
    assert torch.equal(reference, chunk.module.trainable.detach())


def test_an_unwrapped_chunk_is_offloaded_by_walking_it(strategy, megatron_single_rank):
    """Twinkle skips the DDP wrap on a single rank and reference models are never wrapped, so a
    chunk with no buckets still has to move -- all of it, not just the frozen parts."""
    chunk = Tiny(transformer_config()).cuda()
    assert strategy._flat_buffers(chunk) == []

    strategy.offload_to_cpu([chunk])
    assert chunk.trainable.device.type == 'cpu' and chunk.frozen.device.type == 'cpu'
    strategy.reload_to_gpu([chunk])
    assert chunk.trainable.device.type == 'cuda' and chunk.frozen.device.type == 'cuda'


def test_optimizer_state_follows_the_model(strategy, make_chunk):
    """State is walked by tensor rather than by well-known key, so a plain torch optimizer suffices
    to show momentum buffers making the trip -- and covers the bare-optimizer fallback."""
    chunk = make_chunk(distributed_optimizer=False)
    optimizer = torch.optim.SGD([chunk.module.trainable], lr=0.1, momentum=0.9)
    chunk.module.trainable.grad = torch.ones_like(chunk.module.trainable)
    optimizer.step()  # materialises momentum_buffer on the device

    def state_devices():
        return {v.device.type for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)}

    assert state_devices() == {'cuda'}
    strategy._move_optimizer_state(optimizer, 'cpu')
    assert state_devices() == {'cpu'}
    strategy._move_optimizer_state(optimizer, 'cuda:0')
    assert state_devices() == {'cuda'}


def test_megatron_fsdp_is_refused_rather_than_half_offloaded(strategy):
    """FSDP keeps parameters in a different structure. Freeing only the parts this code understands
    would leave most of the memory resident while reporting success, so it must refuse instead."""
    from megatron.core.distributed import FullyShardedDataParallel as MegatronFSDP

    # isinstance is all _flat_buffers looks at; a real FSDP wrap would need a sharded world.
    with pytest.raises(NotImplementedError, match='Megatron-FSDP'):
        strategy._flat_buffers(object.__new__(MegatronFSDP))


def _recording_model(optimizer_group):
    """A MegatronModel reduced to the parts offloading touches, with the strategy replaced by a spy.

    The real constructor builds a parallel layout and a Megatron model; what needs checking here is
    only that the driver-facing methods pass on both halves of the state.
    """
    from twinkle.model.megatron.megatron import MegatronModel

    class RecordingStrategy:

        def __init__(self):
            self.calls = []

        def offload_to_cpu(self, model, optimizer=None):
            self.calls.append(('offload', model, optimizer))

        def reload_to_gpu(self, model, optimizer=None):
            self.calls.append(('reload', model, optimizer))

    model = object.__new__(MegatronModel)
    model.strategy = RecordingStrategy()
    model.model = [nn.Linear(2, 2)]
    model.optimizer_group = optimizer_group
    model._get_default_group = lambda: 'default'
    return model


def test_the_model_passes_both_its_chunks_and_its_optimizer_on():
    """The driver reaches offloading through the model, and the optimizer state is most of the cost."""
    optimizer = object()
    model = _recording_model({'default': SimpleNamespace(optimizer=optimizer)})

    model.offload_to_cpu()
    model.reload_to_gpu()

    assert model.strategy.calls == [
        ('offload', model.model, optimizer),
        ('reload', model.model, optimizer),
    ]


def test_offloading_a_model_that_has_no_optimizer_yet():
    """A reference model, or a rollout-only phase: there is no optimizer to move, which is not an error."""
    model = _recording_model({})

    model.offload_to_cpu()

    assert model.strategy.calls == [('offload', model.model, None)]
