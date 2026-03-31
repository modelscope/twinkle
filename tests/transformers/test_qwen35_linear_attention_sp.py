# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import socket
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from twinkle.loss import CrossEntropyLoss
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelStrategy, sequence_parallel
from twinkle.utils import DeviceMesh, selective_log_softmax

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_qwen35

    _HAS_QWEN35 = True
except Exception:
    Qwen3_5ForCausalLM = None
    Qwen3_5TextConfig = None
    hf_qwen35 = None
    _HAS_QWEN35 = False


WORLD_SIZE = 2
LOGITS_RTOL = 5e-2
LOGITS_ATOL = 5e-2
LOSS_ATOL = 5e-2
GRAD_RTOL = 1e-1
GRAD_ATOL = 5e-2
_HAS_FLA_PREFILL = bool(_HAS_QWEN35 and getattr(hf_qwen35, 'causal_conv1d_fn', None) is not None)
# CUDA_VISIBLE_DEVICES=0,1 \
# pytest -q tests/transformers/test_qwen35_linear_attention_sp.py -rs -s

# CUDA_VISIBLE_DEVICES=0,1 \
# pytest -q tests/transformers/test_qwen35_linear_attention_sp.py::TestQwen35LinearAttentionSP::test_qwen35_linear_attention_forward_grad_alignment -rs -s

# CUDA_VISIBLE_DEVICES=0,1 \
# pytest -q tests/transformers/test_qwen35_linear_attention_sp.py::TestQwen35LinearAttentionSP::test_qwen35_linear_attention_cache_decode_alignment -rs -s

# CUDA_VISIBLE_DEVICES=0,1 \
# pytest -q tests/transformers/test_qwen35_linear_attention_sp.py::TestQwen35LinearAttentionSP::test_qwen35_linear_attention_packed_forward_alignment -rs -s
# CUDA_VISIBLE_DEVICES=0,1 \
# pytest -q tests/transformers/test_qwen35_linear_attention_sp.py::TestQwen35LinearAttentionSP::test_qwen35_linear_attention_cache_decode_alignment -rs -s

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _init_dist(rank: int, world_size: int, port: int) -> torch.device:
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}',
        device_id=device,
        timeout=timedelta(minutes=15),
    )
    return device


def _set_determinism(seed: int) -> None:
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    os.environ.setdefault('NCCL_DETERMINISTIC', '1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _build_tiny_qwen35(
    device: torch.device,
    *,
    layer_types,
    use_cache: bool,
) -> Qwen3_5ForCausalLM:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=list(layer_types),
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        use_cache=use_cache,
    )
    config._attn_implementation = 'sdpa'
    model = Qwen3_5ForCausalLM(config)
    model.to(device=device, dtype=_model_dtype())
    model.eval()
    return model


def _make_strategy(model: Qwen3_5ForCausalLM, world_size: int) -> SequenceParallelStrategy:
    strategy = SequenceParallelStrategy(
        device_mesh=DeviceMesh.from_sizes(
            world_size=world_size,
            fsdp_size=world_size,
            dp_size=1,
            ulysses_size=world_size,
            device_type='cuda',
        ),
        sp_config={
            'enabled': True,
            'ulysses_size': world_size,
            'gather_logits': True,
        },
        model=model,
        tokenizer_id=None,
    )
    strategy._tokenizer = SimpleNamespace(pad_token_id=0)
    strategy.initialize()
    return strategy


def _make_shift_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    labels[..., :-1] = input_ids[..., 1:]
    labels = labels.clone()
    labels[attention_mask == 0] = -100
    labels[..., -1] = -100
    return labels


def _causal_ce_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction='sum',
    )


def _gather_sequence_hidden(hidden_states: torch.Tensor) -> torch.Tensor:
    if sequence_parallel.world_size is None or sequence_parallel.world_size <= 1:
        return hidden_states
    gathered = sequence_parallel.gather(hidden_states, dim=1, position_ids=sequence_parallel.real_position_ids)
    real_pos = sequence_parallel.real_position_ids
    if real_pos is not None and torch.is_tensor(real_pos) and real_pos.dim() >= 2:
        gathered = gathered[:, :real_pos.shape[1]].contiguous()
    return gathered


def _gather_full_grad(grad: torch.Tensor) -> torch.Tensor:
    if sequence_parallel.world_size is None or sequence_parallel.world_size <= 1:
        return grad
    grad = sequence_parallel.gather(grad, dim=1, position_ids=sequence_parallel.real_position_ids)
    real_pos = sequence_parallel.real_position_ids
    if real_pos is not None and torch.is_tensor(real_pos) and real_pos.dim() >= 2:
        grad = grad[:, :real_pos.shape[1]].contiguous()
    return grad


def _make_train_batch(device: torch.device):
    input_ids = torch.tensor([
        [0, 0, 11, 12, 13, 14, 15, 16],
        [21, 22, 23, 24, 25, 26, 27, 28],
    ], device=device, dtype=torch.long)
    attention_mask = torch.tensor([
        [0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], device=device, dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0).expand_as(input_ids)
    labels = _make_shift_labels(input_ids, attention_mask)
    return input_ids, attention_mask, position_ids, labels


def _make_decode_batch(device: torch.device):
    input_ids, attention_mask, position_ids, _ = _make_train_batch(device)
    next_input_ids = torch.tensor([[31], [32]], device=device, dtype=torch.long)
    next_attention_mask = torch.ones((2, 1), device=device, dtype=torch.long)
    next_position_ids = torch.full((2, 1), input_ids.shape[1], device=device, dtype=torch.long)
    prefill_cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
    decode_cache_position = torch.tensor([input_ids.shape[1]], device=device, dtype=torch.long)
    return (
        input_ids,
        attention_mask,
        position_ids,
        next_input_ids,
        next_attention_mask,
        next_position_ids,
        prefill_cache_position,
        decode_cache_position,
    )


def _make_packed_batch(device: torch.device):
    input_ids = torch.tensor([[11, 12, 13, 21, 22, 23, 24, 25]], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]], device=device, dtype=torch.long)
    return input_ids, attention_mask, position_ids


def _run_forward_grad_alignment_worker(rank: int, world_size: int, port: int):
    device = _init_dist(rank, world_size, port)
    try:
        _set_determinism(1234)
        os.environ['QWEN35_SP_LINEAR_HEAD_PARALLEL'] = '1'

        baseline_model = _build_tiny_qwen35(
            device, layer_types=['linear_attention', 'linear_attention'], use_cache=False)
        sp_model = copy.deepcopy(baseline_model)
        input_ids, attention_mask, position_ids, labels = _make_train_batch(device)

        baseline_embeds = baseline_model.get_input_embeddings()(input_ids).detach().clone().requires_grad_(True)
        baseline_outputs = baseline_model(
            inputs_embeds=baseline_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        baseline_logits = baseline_outputs.logits.float()
        baseline_loss = F.cross_entropy(
            baseline_logits.reshape(-1, baseline_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction='mean',
        )
        baseline_loss.backward()
        baseline_grad = baseline_embeds.grad.detach().float()

        strategy = _make_strategy(sp_model, world_size)
        processed_inputs = strategy.preprocess_inputs({
            'input_ids': input_ids,
            'position_ids': position_ids,
            'labels': labels,
        })
        local_labels = processed_inputs['labels']
        sp_embeds = sp_model.get_input_embeddings()(input_ids).detach().clone().requires_grad_(True)
        sp_outputs = sp_model(
            inputs_embeds=sp_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        gathered_outputs = strategy.postprocess_outputs(copy.copy(sp_outputs))
        gathered_logits = gathered_outputs.logits.float()
        if not torch.allclose(gathered_logits, baseline_logits, rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
            max_diff = (gathered_logits - baseline_logits).abs().max().item()
            raise AssertionError(f'forward logits mismatch on rank {rank}: max_diff={max_diff}')

        loss_instance = CrossEntropyLoss(reduction='mean')
        local_logits = sp_outputs.logits
        masked_local_labels = local_labels.masked_fill(local_labels == -100, 0)
        local_logps = selective_log_softmax(local_logits, masked_local_labels)
        loss_inputs = {'labels': local_labels}
        loss_outputs = {'logits': local_logits, 'logps': local_logps}
        loss_inputs, loss_outputs = strategy.gather_loss_tensors(loss_inputs, loss_outputs)
        result = loss_instance(loss_inputs, loss_outputs)
        sp_loss = result['loss']
        if not torch.allclose(sp_loss.detach(), baseline_loss.detach(), atol=LOSS_ATOL, rtol=0):
            raise AssertionError(
                f'forward loss mismatch on rank {rank}: baseline={baseline_loss.item()} sp={sp_loss.item()}')
        sp_loss.backward()
        sp_grad = _gather_full_grad(sp_embeds.grad.detach().float())
        if not torch.allclose(sp_grad, baseline_grad, rtol=GRAD_RTOL, atol=GRAD_ATOL):
            max_diff = (sp_grad - baseline_grad).abs().max().item()
            raise AssertionError(f'input grad mismatch on rank {rank}: max_diff={max_diff}')
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _run_cache_decode_alignment_worker(rank: int, world_size: int, port: int):
    device = _init_dist(rank, world_size, port)
    try:
        _set_determinism(4321)
        os.environ['QWEN35_SP_LINEAR_HEAD_PARALLEL'] = '1'

        # Keep one full-attention layer so HF's Qwen3.5 cache can derive seq length
        # from KV cache during decode; all-linear cache is not supported upstream.
        baseline_model = _build_tiny_qwen35(
            device, layer_types=['linear_attention', 'full_attention'], use_cache=True)
        sp_model = copy.deepcopy(baseline_model)
        (
            input_ids,
            attention_mask,
            position_ids,
            next_input_ids,
            next_attention_mask,
            next_position_ids,
            prefill_cache_position,
            decode_cache_position,
        ) = _make_decode_batch(device)

        baseline_cache = hf_qwen35.Qwen3_5DynamicCache(config=baseline_model.config)

        baseline_prefill = baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=baseline_cache,
            cache_position=prefill_cache_position,
            output_hidden_states=True,
        )
        baseline_prefill_logits = baseline_prefill.logits.float()
        baseline_decode = baseline_model(
            input_ids=next_input_ids,
            attention_mask=next_attention_mask,
            position_ids=next_position_ids,
            use_cache=True,
            past_key_values=baseline_prefill.past_key_values,
            cache_position=decode_cache_position,
            output_hidden_states=True,
        )
        baseline_decode_logits = baseline_decode.logits.float()

        strategy = _make_strategy(sp_model, world_size)
        sp_cache = hf_qwen35.Qwen3_5DynamicCache(config=sp_model.config)
        sp_prefill = sp_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=sp_cache,
            cache_position=prefill_cache_position,
            output_hidden_states=True,
        )
        sp_prefill = strategy.postprocess_outputs(sp_prefill)
        sp_prefill_logits = sp_prefill.logits.float()
        if not torch.allclose(sp_prefill_logits, baseline_prefill_logits, rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
            max_diff = (sp_prefill_logits - baseline_prefill_logits).abs().max().item()
            raise AssertionError(f'prefill logits mismatch on rank {rank}: max_diff={max_diff}')
        for layer_idx, (sp_hidden, base_hidden) in enumerate(
                zip(sp_prefill.hidden_states, baseline_prefill.hidden_states)):
            gathered_hidden = _gather_sequence_hidden(sp_hidden)
            if not torch.allclose(gathered_hidden.float(), base_hidden.float(), rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
                max_diff = (gathered_hidden.float() - base_hidden.float()).abs().max().item()
                raise AssertionError(
                    f'prefill hidden_states mismatch on rank {rank} layer={layer_idx}: max_diff={max_diff}')

        sp_cache = sp_prefill.past_key_values
        if sp_cache is None:
            raise AssertionError('SP prefill did not return past_key_values.')
        if sp_cache.conv_states[0] is None:
            raise AssertionError('SP prefill did not initialize linear conv_states.')
        if sp_cache.recurrent_states[0] is None:
            raise AssertionError('SP prefill did not initialize linear recurrent_states.')

        sp_decode = sp_model(
            input_ids=next_input_ids,
            attention_mask=next_attention_mask,
            position_ids=next_position_ids,
            use_cache=True,
            past_key_values=sp_cache,
            cache_position=decode_cache_position,
            output_hidden_states=True,
        )
        sp_decode = strategy.postprocess_outputs(sp_decode)
        sp_decode_logits = sp_decode.logits.float()
        for layer_idx, (sp_hidden, base_hidden) in enumerate(
                zip(sp_decode.hidden_states, baseline_decode.hidden_states)):
            gathered_hidden = _gather_sequence_hidden(sp_hidden)
            if not torch.allclose(gathered_hidden.float(), base_hidden.float(), rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
                max_diff = (gathered_hidden.float() - base_hidden.float()).abs().max().item()
                raise AssertionError(
                    f'decode hidden_states mismatch on rank {rank} layer={layer_idx}: max_diff={max_diff}')
        if not torch.allclose(sp_decode_logits, baseline_decode_logits, rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
            max_diff = (sp_decode_logits - baseline_decode_logits).abs().max().item()
            raise AssertionError(f'decode logits mismatch on rank {rank}: max_diff={max_diff}')
        if sp_cache.recurrent_states[0] is None:
            raise AssertionError('SP decode cleared linear recurrent_states unexpectedly.')
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _run_packed_forward_alignment_worker(rank: int, world_size: int, port: int):
    device = _init_dist(rank, world_size, port)
    try:
        _set_determinism(2468)
        os.environ['QWEN35_SP_LINEAR_HEAD_PARALLEL'] = '1'

        baseline_model = _build_tiny_qwen35(device, layer_types=['linear_attention', 'linear_attention'], use_cache=False)
        sp_model = copy.deepcopy(baseline_model)
        input_ids, attention_mask, position_ids = _make_packed_batch(device)

        baseline_outputs = baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        baseline_logits = baseline_outputs.logits.float()

        strategy = _make_strategy(sp_model, world_size)
        sp_outputs = sp_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        sp_outputs = strategy.postprocess_outputs(sp_outputs)
        sp_logits = sp_outputs.logits.float()
        if not torch.allclose(sp_logits, baseline_logits, rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
            max_diff = (sp_logits - baseline_logits).abs().max().item()
            raise AssertionError(f'packed logits mismatch on rank {rank}: max_diff={max_diff}')
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


@unittest.skipUnless(_HAS_QWEN35, 'transformers Qwen3.5 is not available in this environment')
@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE, 'requires 2 CUDA devices')
@unittest.skipUnless(_HAS_FLA_PREFILL, 'requires flash-linear-attention causal_conv1d_fn for Qwen3.5 SP patch')
class TestQwen35LinearAttentionSP(unittest.TestCase):

    def test_qwen35_linear_attention_forward_grad_alignment(self):
        port = _find_free_port()
        mp.spawn(
            _run_forward_grad_alignment_worker,
            args=(WORLD_SIZE, port),
            nprocs=WORLD_SIZE,
            join=True,
        )

    def test_qwen35_linear_attention_cache_decode_alignment(self):
        port = _find_free_port()
        mp.spawn(
            _run_cache_decode_alignment_worker,
            args=(WORLD_SIZE, port),
            nprocs=WORLD_SIZE,
            join=True,
        )

    def test_qwen35_linear_attention_packed_forward_alignment(self):
        port = _find_free_port()
        mp.spawn(
            _run_packed_forward_alignment_worker,
            args=(WORLD_SIZE, port),
            nprocs=WORLD_SIZE,
            join=True,
        )


if __name__ == '__main__':
    unittest.main()
