"""Real trainer-process -> two vLLM-style TP worker processes, without a model engine."""
import asyncio
import os
import threading
import time
import uuid
import weakref
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


def test_owned_direct_ipc_mapping_survives_queued_loader_reads(monkeypatch):
    """Quantized/owned loaders must finish reads before dropping foreign IPC mappings."""
    from twinkle.sampler.vllm_sampler import vllm_worker_extension as extension

    events, references = [], []
    metadata = iter(['bucket', {'bucket_meta': [
        {'name': 'weight', 'shape': (4,), 'dtype': torch.float32, 'handle': 'large'}], 'is_last': True}])

    class Socket:
        def connect(self, endpoint):
            pass

        def send(self, data):
            events.append('ack')

        def close(self):
            pass

    class Receiver(extension.TwinkleWorkerExtension):
        def _load_weights(self, weights, **kwargs):
            events.append('load')
            references.append(weakref.ref(weights[0][1]))

    def rebuild(handle, device):
        return torch.empty(16, dtype=torch.uint8) if handle == 'bucket' else torch.ones(4)

    def synchronize(device):
        if 'load' in events:
            assert references[-1]() is not None, 'IPC mapping released before device reads completed'
            events.append('load_finished')

    worker = object.__new__(Receiver)
    worker.device = torch.device('cpu')
    worker.rank = worker.local_rank = 0
    worker.model_runner = NS(parallel_config=NS(tensor_parallel_size=2))
    worker._quant_reload_state = object()  # Preserve the owned-loading path.
    worker._zmq_ctx = NS(socket=lambda *args: Socket())
    monkeypatch.setattr(extension, '_rebuild_ipc', rebuild)
    monkeypatch.setattr(extension, 'configure_zmq_socket', lambda *args, **kwargs: None)
    monkeypatch.setattr('twinkle.checkpoint_engine.zmq_auth.recv_auth_pyobj', lambda *args: next(metadata))
    monkeypatch.setattr(extension.Platform, 'get_platform', lambda: NS(ensure_ipc_reducer=lambda: None))
    monkeypatch.setattr(extension.Torch, 'synchronize', synchronize)
    monkeypatch.setattr(extension.Torch, 'ipc_collect', lambda: None)
    worker.update_weights_from_ipc(zmq_handle='ipc:///unused', independent=True)
    assert events.index('load_finished') > events.index('load')
    assert references[-1]() is None


def receive(rank, rendezvous, endpoints, directory, independent=False):
    import torch.distributed as dist
    import vllm.distributed

    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension

    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['TWINKLE_VLLM_IPC_TIMEOUT_S'] = '30'
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    vllm.distributed.get_tp_group = lambda: NS(cpu_group=dist.group.WORLD, ranks=[0, 1])

    class Receiver(TwinkleWorkerExtension):
        def _load_weights(self, weights, **kwargs):
            time.sleep(0.005)  # Catch ACK-before-load races when the sender reuses a bucket.
            for name, tensor in weights:
                self.loaded[name] = tensor.detach().clone()

    worker = Receiver()
    worker.rank = worker.local_rank = rank
    worker.device = torch.device('cuda', rank)
    worker.model_runner = NS(parallel_config=NS(tensor_parallel_size=2))
    pointers = []
    try:
        for step, endpoint in enumerate(endpoints):
            worker.loaded = {}
            worker.update_weights_from_ipc(
                zmq_handle=endpoint[rank] if independent else endpoint,
                auth_key='test-key', independent=independent)
            torch.save({name: tensor.cpu() for name, tensor in worker.loaded.items()},
                       Path(directory) / f'rank{rank}-step{step}.pt')
            pointers.append(worker._twinkle_ipc_buffer.data_ptr())
        torch.save(pointers, Path(directory) / f'pointers{rank}.pt')
    finally:
        worker._twinkle_ipc_buffer = None
        dist.destroy_process_group()


@pytest.mark.parametrize('independent', [False, True])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two devices required to validate TP peer mappings')
def test_direct_authenticated_multi_bucket_transfer_and_reused_buffer(tmp_path, monkeypatch, independent):
    import torch.multiprocessing as mp

    from twinkle.sampler.vllm_sampler.weight_transfer import VLLMWeightSender

    monkeypatch.setenv('TWINKLE_VLLM_IPC_TIMEOUT_S', '30')
    torch.cuda.set_device(0)
    weights = [
        ('large', torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')),
        ('transposed', torch.randn(64, 128, device='cuda').T),
        ('small', torch.randn(17, device='cuda', dtype=torch.float16)),
    ]
    tag = uuid.uuid4().hex
    endpoints = [
        [f'ipc:///tmp/twinkle-test-{tag}-{step}-{rank}.sock' for rank in range(2)]
        if independent else f'ipc:///tmp/twinkle-test-{tag}-{step}.sock' for step in range(3)]
    context = mp.get_context('spawn')
    workers = [context.Process(target=receive, args=(
        rank, 'file://' + str(tmp_path / 'gloo'), endpoints, str(tmp_path), independent)) for rank in range(2)]
    for child in workers:
        child.start()
    send_errors = []

    def send(rank):
        try:
            with torch.cuda.device(rank):
                sender = VLLMWeightSender()
                for step, endpoint in enumerate(endpoints):
                    payload = [(name, tensor.to(rank)) for name, tensor in weights] if step < 2 else []
                    if step == 0:
                        payload = [(name, tensor.cpu()) for name, tensor in weights]
                    asyncio.run(sender.update_weights(
                        payload, bucket_size_mb=1,
                        zmq_handle=endpoint[rank] if independent else endpoint,
                        auth_key='test-key', device=torch.device('cuda', rank),
                        direct_large_tensors=independent))
        except BaseException as exc:
            send_errors.append(exc)

    try:
        senders = [threading.Thread(target=send, args=(rank,)) for rank in range(2 if independent else 1)]
        for sender in senders:
            sender.start()
        for sender in senders:
            sender.join(timeout=90)
            assert not sender.is_alive(), 'Sender failed to complete'
        if send_errors:
            raise send_errors[0]
        for child in workers:
            child.join(timeout=60)
            assert child.exitcode == 0
        for rank in range(2):
            for step in range(2):
                loaded = torch.load(tmp_path / f'rank{rank}-step{step}.pt', weights_only=True)
                assert set(loaded) == {name for name, _ in weights}
                for name, expected in weights:
                    assert torch.equal(loaded[name], expected.cpu()), (rank, step, name)
            assert torch.load(tmp_path / f'rank{rank}-step2.pt', weights_only=True) == {}
            pointers = torch.load(tmp_path / f'pointers{rank}.pt', weights_only=True)
            assert len(set(pointers)) == 1, 'IPC mappings must be reused across publications'
    finally:
        for child in workers:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
