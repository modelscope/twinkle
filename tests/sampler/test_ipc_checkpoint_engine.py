"""Handing weights to a peer that shares the GPU, with two real processes on one device.

This is the case the NCCL engine cannot cover at all: two ranks on one device fail to build a
communicator (``Duplicate GPU detected``). So the transport under test here is not an alternative
path, it is the only one, and it has to be exercised with genuinely separate processes -- a
same-process test would map memory the sender already owns and prove nothing about IPC.

The receiver copies each tensor to host as it arrives, before the acknowledgement that lets the
sender refill the buffer. That is the protocol's real constraint: what the receiver yields are views
into the sender's memory, not copies, so anything held past the acknowledgement is a dangling read.
The multi-bucket cases below are what make that constraint bite.
"""
import asyncio
import io
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA IPC needs a GPU')

MB = 1 << 20


def _receiver(result_queue, bucket_size):
    """Run in a separate process: map the sender's buffer and copy what it points at to host."""
    sys.path.insert(0, os.environ['TWINKLE_SRC'])
    from twinkle.checkpoint_engine import IPCCheckpointEngine

    torch.cuda.set_device(0)
    engine = IPCCheckpointEngine(bucket_size=bucket_size)
    engine.init_process_group(rank=1)
    received = []
    mappings = set()

    async def drain():
        async for name, tensor in engine.receive_weights():
            # Host copy now, while the mapping is still valid; also records which mapping it came from
            # so the test can tell whether the receiver re-mapped per bucket.
            mappings.add(tensor.untyped_storage().data_ptr())
            received.append((name, tensor.detach().to('cpu', copy=True)))

    try:
        asyncio.run(drain())
        # Serialised rather than handed over as tensors: Torch shares CPU tensors across processes via
        # shared-memory file descriptors that stay tied to this process, and this one is about to exit.
        blob = io.BytesIO()
        torch.save(received, blob)
        result_queue.put(('ok', blob.getvalue(), len(mappings)))
    except Exception as e:  # noqa: BLE001
        import traceback
        result_queue.put(('error', f'{type(e).__name__}: {e}', traceback.format_exc()))
    finally:
        engine.finalize()


def _round_trip(tensors, bucket_size=8 * MB):
    """Send ``tensors`` from this process to a child on the same GPU; return what the child saw."""
    import torch.multiprocessing as mp

    from twinkle.checkpoint_engine import IPCCheckpointEngine

    torch.cuda.set_device(0)
    engine = IPCCheckpointEngine(bucket_size=bucket_size)
    # Bind before the child connects, and before it can trip over a socket file from an earlier run.
    engine.init_process_group(rank=0)

    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    os.environ['TWINKLE_SRC'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    child = ctx.Process(target=_receiver, args=(queue, bucket_size))
    child.start()
    try:
        asyncio.run(engine.send_weights(iter(tensors)))
        try:
            outcome = queue.get(timeout=300)
        except Exception as e:  # noqa: BLE001
            child.join(timeout=30)
            # A negative exit code is a signal: the receiver died rather than reporting anything, which
            # a bare queue error would otherwise hide.
            raise AssertionError(f'receiver produced no result ({type(e).__name__}); '
                                 f'exit code {child.exitcode}') from e
    finally:
        child.join(timeout=60)
        engine.finalize()

    if outcome[0] == 'error':
        raise AssertionError(f'receiver failed: {outcome[1]}\n{outcome[2]}')
    return torch.load(io.BytesIO(outcome[1]), weights_only=True), outcome[2]


def test_weights_arrive_bit_exact_across_processes():
    """Mixed dtypes and ranks, including a source that is not contiguous."""
    source = [
        ('embed.weight', torch.randn(512, 64, dtype=torch.bfloat16, device='cuda')),
        ('layer.0.bias', torch.randn(64, dtype=torch.float32, device='cuda')),
        # Transposed: the sender has to make this contiguous before reinterpreting it as bytes.
        ('layer.0.weight', torch.randn(64, 128, device='cuda').t()),
        ('head.weight', torch.randn(37, 5, dtype=torch.float16, device='cuda')),
    ]
    received, _ = _round_trip(source)

    assert [name for name, _ in received] == [name for name, _ in source]
    for (_, expected), (name, got) in zip(source, received, strict=True):
        assert got.dtype == expected.dtype, name
        assert got.shape == expected.shape, name
        assert torch.equal(got, expected.cpu()), name


def test_a_model_larger_than_the_bucket_is_streamed():
    """Several buckets, so the sender reuses the buffer while the receiver keeps one mapping.

    Re-mapping per bucket is what makes device memory look like it is growing during a sync, so the
    mapping count is asserted, not just the payload.
    """
    bucket_size = 4 * MB
    # 12 x 1 MiB against a 4 MiB bucket: three full flushes plus the closing one.
    source = [(f'layer.{i}.weight', torch.randn(512, 512, dtype=torch.float32, device='cuda')) for i in range(12)]
    total_bytes = sum(t.numel() * t.element_size() for _, t in source)
    assert total_bytes > 2 * bucket_size, 'this test is pointless unless it spans several buckets'

    received, mapping_count = _round_trip(source, bucket_size=bucket_size)

    assert len(received) == len(source)
    for (_, expected), (name, got) in zip(source, received, strict=True):
        assert torch.equal(got, expected.cpu()), name
    assert mapping_count == 1, f'receiver re-mapped {mapping_count} times instead of reusing one buffer'


def test_a_single_tensor_wider_than_the_bucket_still_fits():
    """The buffer grows for a tensor that cannot be split, rather than silently truncating it."""
    bucket_size = 1 * MB
    big = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')  # 4 MiB, four times the bucket
    assert big.numel() * big.element_size() > bucket_size

    received, _ = _round_trip([('big.weight', big), ('small.bias', torch.randn(8, device='cuda'))],
                              bucket_size=bucket_size)

    assert [name for name, _ in received] == ['big.weight', 'small.bias']
    assert torch.equal(received[0][1], big.cpu())


def test_sending_nothing_still_ends_the_receiver():
    """The closing message carries no buffer; the receiver must not try to map one."""
    received, _ = _round_trip([])
    assert received == []


def test_the_endpoint_is_per_physical_gpu():
    """Both peers derive the rendezvous from the GPU rather than exchanging it.

    Device index would not do: under Ray each role sees its own GPU as index 0, so the paths would
    collide between ranks that are not colocated at all.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs two GPUs to show the endpoints differ')
    from twinkle.checkpoint_engine import IPCCheckpointEngine

    torch.cuda.set_device(0)
    first = IPCCheckpointEngine.endpoint()
    torch.cuda.set_device(1)
    second = IPCCheckpointEngine.endpoint()
    torch.cuda.set_device(0)

    assert first != second
    assert first.startswith('ipc://')


def test_mismatched_world_sizes_are_refused():
    """A sampler that does not divide the trainer evenly is not covering whole GPUs, so pairing is wrong."""
    from twinkle.checkpoint_engine import IPCCheckpointEngine

    with pytest.raises(ValueError, match='whole number of trainer'):
        IPCCheckpointEngine.build_topology(4, 3, [])
    with pytest.raises(ValueError, match='whole number of trainer'):
        IPCCheckpointEngine.build_topology(4, 0, [])


@pytest.mark.parametrize(
    'trainer_size, rollout_size, expected',
    [
        (4, 4, [0, 0, 0, 0]),  # one sampler process per GPU: everybody sends
        (4, 2, [0, -1, 0, -1]),  # tp=2: one sender per sampler, its neighbour idle
        (4, 1, [0, -1, -1, -1]),  # tp=4: a single sampler across all four
    ])
def test_a_tensor_parallel_sampler_is_fed_by_one_rank_per_process(trainer_size, rollout_size, expected):
    """A tp sampler is one process over several GPUs, so only the rank on the one it listens on sends.

    Measured: a process holding ``CUDA_VISIBLE_DEVICES=0,1`` reports ``current_device()`` on the first of
    them. So the sampler's endpoint is that GPU's, and the trainer rank there -- index a multiple of tp --
    is its peer. The others have no listener and must stay quiet rather than bind a socket nobody reads.

    Dropping them loses nothing: every trainer rank materialises the same full HF tensors, and the
    sampler fans out to its own tp workers after receiving once.
    """
    from twinkle.checkpoint_engine import IPCCheckpointEngine

    trainer, rollout = IPCCheckpointEngine.build_topology(trainer_size, rollout_size, [])

    assert trainer['rank'] == expected
    assert rollout['rank'] == [1] * rollout_size
    # Every channel stays a two-member pair however many GPUs a sampler covers.
    assert trainer['world_size'] == [2] * trainer_size


def test_an_idle_rank_opens_no_channel_and_still_drains_its_weights():
    """An idle rank must not bind a socket, but must not abandon the generator either.

    The generator is what drives the bridge's export; walking away from it half-way would leave that
    conversion suspended. So the weights are consumed and discarded.
    """
    import asyncio

    from twinkle.checkpoint_engine import IPCCheckpointEngine

    engine = IPCCheckpointEngine()
    engine.init_process_group(rank=-1)
    assert engine.socket is None

    consumed = []

    def weights():
        for name in ('a', 'b'):
            consumed.append(name)
            yield name, torch.zeros(4)

    asyncio.run(engine.send_weights(weights()))

    assert consumed == ['a', 'b']
    assert engine.send_buf is None


def test_the_manager_picks_this_engine_only_when_colocating():
    from twinkle.checkpoint_engine import CheckpointEngineManager, IPCCheckpointEngine, NCCLCheckpointEngine

    assert CheckpointEngineManager.decide_backend_engine('GPU', colocate=True) is IPCCheckpointEngine
    assert CheckpointEngineManager.decide_backend_engine('GPU') is NCCLCheckpointEngine


def test_a_role_builds_the_engine_its_manager_asked_for():
    """The manager decides, but each worker builds its own engine, so the mixin has to honour that."""
    from twinkle.checkpoint_engine import CheckpointEngineMixin, IPCCheckpointEngine, NCCLCheckpointEngine

    class Role(CheckpointEngineMixin):
        pass

    assert isinstance(Role()._get_or_create_checkpoint_engine(), NCCLCheckpointEngine)

    colocated = Role()
    colocated.set_checkpoint_engine_backend('ipc')
    assert isinstance(colocated._get_or_create_checkpoint_engine(), IPCCheckpointEngine)


@pytest.mark.parametrize('backend', ['vllm', 'sglang'])
def test_both_sampler_backends_reach_this_engine_through_the_same_seam(backend):
    """Colocation is transport selection, not a per-backend feature, and both samplers must agree.

    Neither ``vLLMSampler.receive_weights`` nor ``SGLangSampler.receive_weights`` names a transport:
    each hands ``engine.receive_weights()`` to its own engine's ``update_weights``. That is the whole
    reason colocation needed no change on either sampler, and it is worth pinning -- a future rewrite
    that reached for NCCL directly in one of them would break colocation there and nowhere else.

    The engines are never started here; only the mixin's choice is under test.
    """
    from twinkle.checkpoint_engine import IPCCheckpointEngine
    if backend == 'vllm':
        from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler as sampler_cls
    else:
        from twinkle.sampler.sglang_sampler.sglang_sampler import SGLangSampler as sampler_cls

    sampler = object.__new__(sampler_cls)
    sampler.set_checkpoint_engine_backend('ipc')

    assert isinstance(sampler._get_or_create_checkpoint_engine(), IPCCheckpointEngine)
    # sleep/wake_up are what hand the device over; both backends had them before colocation existed.
    assert callable(getattr(sampler_cls, 'sleep', None))
    assert callable(getattr(sampler_cls, 'wake_up', None))
