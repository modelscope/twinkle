"""Offline DPO loss/gradient equivalence over two CPU/Gloo SP ranks."""
import copy
import os
import pytest
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from types import SimpleNamespace
from transformers import LlamaConfig, LlamaForCausalLM

from twinkle import DeviceMesh
from twinkle.loss import DPOLoss
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelStrategy, _current_ctx
from twinkle.utils import selective_log_softmax


def _worker(rank, port):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = '2'
    torch.set_num_threads(1)
    torch.manual_seed(42)
    # torch_npu sets checkpoint's default device to NPU even for CPU-only
    # tensors. This regression deliberately checkpoints on CPU.
    from torch.utils.checkpoint import DefaultDeviceType
    DefaultDeviceType.set_device_type('cpu')
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                             attention_dropout=0.0, pad_token_id=0)
        config._attn_implementation = 'sdpa'
        baseline = LlamaForCausalLM(config).eval()
        reference = copy.deepcopy(baseline).eval()
        with torch.no_grad():
            for param in baseline.parameters():
                param.add_(torch.randn_like(param) * 1e-3)
        policy = copy.deepcopy(baseline)
        ref_sp = copy.deepcopy(reference)
        # Four rows = two interleaved chosen/rejected pairs. Length 9 exercises SP padding.
        ids = torch.randint(1, 64, (4, 9))
        positions = torch.arange(9).expand(4, -1)
        labels = torch.cat((ids[:, 1:], torch.full((4, 1), -100)), dim=1)
        labels[:, :2] = -100
        loss_fn = DPOLoss(beta=0.1, loss_type='sigmoid')

        def score(model, strategy=None):
            local_labels = labels
            if strategy is not None:
                local_labels = strategy.preprocess_inputs(
                    {'input_ids': ids, 'position_ids': positions, 'labels': labels})['labels']
            logits = model(input_ids=ids, position_ids=positions, use_cache=False).logits
            outputs = {'logps': selective_log_softmax(logits, local_labels.masked_fill(local_labels == -100, 0))}
            inputs = {'labels': local_labels}
            if strategy is not None:
                inputs, outputs = strategy.gather_loss_tensors(inputs, outputs)
            return inputs, outputs

        with torch.no_grad():
            _, base_ref = score(reference)
        base_inputs, base_outputs = score(baseline)
        base_loss = loss_fn(base_inputs, base_outputs, ref_outputs=base_ref)['loss']
        base_loss.backward()
        base_grads = {name: p.grad.clone() for name, p in baseline.named_parameters()}

        mesh = DeviceMesh.from_sizes(dp_size=2, ulysses_size=2, device_type='cpu')
        strategies = []
        for model in (policy, ref_sp):
            strategy = SequenceParallelStrategy(mesh, model=model)
            strategy._get_tokenizer = lambda: SimpleNamespace(pad_token_id=0)
            assert strategy.initialize()
            strategies.append(strategy)
        assert strategies[0]._ctx is not strategies[1]._ctx
        policy.gradient_checkpointing_enable()
        policy.train()
        with torch.no_grad():
            _, parallel_ref = score(ref_sp, strategies[1])
        inputs, outputs = score(policy, strategies[0])
        parallel_loss = loss_fn(inputs, outputs, ref_outputs=parallel_ref)['loss']
        torch.testing.assert_close(parallel_loss, base_loss, atol=1e-6, rtol=1e-5)
        parallel_loss.backward()
        # Mirror the data-parallel gradient averaging performed by DDP.
        for name, param in policy.named_parameters():
            assert param.grad is not None, name
            dist.all_reduce(param.grad)
            param.grad.div_(2)
            torch.testing.assert_close(param.grad, base_grads[name], atol=1e-6, rtol=1e-4, msg=name)
        assert _current_ctx() is None
        # Avoid torch_npu's process-wide optimizer capability probe on this
        # explicitly CPU-only test.
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, foreach=False, fused=False)
        before = policy.lm_head.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(before, policy.lm_head.weight)
        assert all(param.grad is None for param in ref_sp.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_dpo_sp_loss_all_parameter_gradients_and_step():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)
