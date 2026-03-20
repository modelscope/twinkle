# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import unittest
from datetime import timedelta
from transformers.modeling_flash_attention_utils import is_flash_attn_available
from transformers.utils.import_utils import is_flash_linear_attention_available
from types import SimpleNamespace

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
_HAS_FLA_PREFILL = bool(
    _HAS_QWEN35 and (getattr(hf_qwen35, 'causal_conv1d_fn', None) is not None or is_flash_linear_attention_available()))


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


def _build_tiny_qwen35(device: torch.device,
                       *,
                       attn_implementation: str = 'sdpa',
                       layer_types: list[str] | None = None) -> Qwen3_5ForCausalLM:
    if layer_types is None:
        layer_types = ['linear_attention', 'linear_attention']
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=layer_types,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = attn_implementation
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


def _make_train_batch(device: torch.device):
    input_ids = torch.tensor([
        [0, 0, 11, 12, 13, 14, 15, 16],
        [21, 22, 23, 24, 25, 26, 27, 28],
    ],
                             device=device,
                             dtype=torch.long)
    attention_mask = torch.tensor([
        [0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
                                  device=device,
                                  dtype=torch.long)
    position_ids = torch.arange(input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0).expand_as(input_ids)
    labels = _make_shift_labels(input_ids, attention_mask)
    return input_ids, attention_mask, position_ids, labels


def _get_qkv_weight(model: Qwen3_5ForCausalLM) -> torch.nn.Parameter:
    for layer in model.model.layers:
        linear_attn = getattr(layer, 'linear_attn', None)
        if linear_attn is not None:
            return linear_attn.in_proj_qkv.weight
    raise AssertionError('No linear attention layer found in Qwen3.5 test model.')


def _allreduce_sp_grad(grad: torch.Tensor) -> torch.Tensor:
    reduced = grad.detach().float().contiguous()
    if sequence_parallel.world_size is not None and sequence_parallel.world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=sequence_parallel._sp_group)
    return reduced


def _run_prefill_alignment_worker(rank: int,
                                  world_size: int,
                                  port: int,
                                  attn_implementation: str = 'sdpa',
                                  layer_types: list[str] | None = None):
    device = _init_dist(rank, world_size, port)
    try:
        _set_determinism(1234)
        os.environ['QWEN35_SP_LINEAR_HEAD_PARALLEL'] = '1'

        baseline_model = _build_tiny_qwen35(device, attn_implementation=attn_implementation, layer_types=layer_types)
        sp_model = copy.deepcopy(baseline_model)
        input_ids, attention_mask, position_ids, labels = _make_train_batch(device)

        baseline_outputs = baseline_model(
            input_ids=input_ids,
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
        baseline_qkv_grad = _get_qkv_weight(baseline_model).grad.detach().float().cpu()

        strategy = _make_strategy(sp_model, world_size)
        processed_inputs = strategy.preprocess_inputs({
            'input_ids': input_ids,
            'position_ids': position_ids,
            'labels': labels,
        })
        local_labels = processed_inputs['labels']
        sp_outputs = sp_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        gathered_outputs = strategy.postprocess_outputs(copy.copy(sp_outputs))
        gathered_logits = gathered_outputs.logits.float()
        if not torch.allclose(gathered_logits, baseline_logits, rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
            max_diff = (gathered_logits - baseline_logits).abs().max().item()
            raise AssertionError(f'prefill logits mismatch on rank {rank}: max_diff={max_diff}')

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
                f'prefill loss mismatch on rank {rank}: baseline={baseline_loss.item()} sp={sp_loss.item()}')
        sp_loss.backward()

        sp_qkv_grad = _allreduce_sp_grad(_get_qkv_weight(sp_model).grad).cpu()
        if not torch.allclose(sp_qkv_grad, baseline_qkv_grad, rtol=GRAD_RTOL, atol=GRAD_ATOL):
            max_diff = (sp_qkv_grad - baseline_qkv_grad).abs().max().item()
            raise AssertionError(f'qkv grad mismatch on rank {rank}: max_diff={max_diff}')
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


@unittest.skipUnless(_HAS_QWEN35, 'transformers Qwen3.5 is not available in this environment')
@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE, 'requires 2 CUDA devices')
@unittest.skipUnless(
    _HAS_FLA_PREFILL,
    'requires either transformers qwen3.5 causal_conv1d_fn or flash-linear-attention kernels for Qwen3.5 SP patch')
class TestQwen35LinearAttentionSP(unittest.TestCase):

    def test_qwen35_linear_attention_prefill_logits_and_qkv_grad_alignment(self):
        port = _find_free_port()
        mp.spawn(
            _run_prefill_alignment_worker,
            args=(WORLD_SIZE, port, 'sdpa', ['linear_attention', 'linear_attention']),
            nprocs=WORLD_SIZE,
            join=True,
        )

    def test_qwen35_mixed_attention_prefill_logits_and_qkv_grad_alignment(self):
        port = _find_free_port()
        mp.spawn(
            _run_prefill_alignment_worker,
            args=(WORLD_SIZE, port, 'sdpa', ['full_attention', 'linear_attention']),
            nprocs=WORLD_SIZE,
            join=True,
        )

    @unittest.skipUnless(is_flash_attn_available(), 'requires flash_attention_2 support in transformers')
    def test_qwen35_linear_attention_prefill_logits_and_qkv_grad_alignment_fa2(self):
        port = _find_free_port()
        mp.spawn(
            _run_prefill_alignment_worker,
            args=(WORLD_SIZE, port, 'flash_attention_2', ['linear_attention', 'linear_attention']),
            nprocs=WORLD_SIZE,
            join=True,
        )

    @unittest.skipUnless(is_flash_attn_available(), 'requires flash_attention_2 support in transformers')
    def test_qwen35_mixed_attention_prefill_logits_and_qkv_grad_alignment_fa2(self):
        port = _find_free_port()
        mp.spawn(
            _run_prefill_alignment_worker,
            args=(WORLD_SIZE, port, 'flash_attention_2', ['full_attention', 'linear_attention']),
            nprocs=WORLD_SIZE,
            join=True,
        )


if __name__ == '__main__':
    unittest.main()
