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
from types import MethodType, SimpleNamespace

from twinkle.loss import CrossEntropyLoss
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelStrategy, sequence_parallel
from twinkle.model.transformers.strategy.sequence_parallel.linear_attention_sp import Qwen3_5GatedDeltaNetUlyssesPatch
from twinkle.model.transformers.strategy.sequence_parallel.utils import get_cu_seqlens_from_position_ids
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

if is_flash_linear_attention_available():
    from fla.modules.convolution import causal_conv1d as _FLA_CAUSAL_CONV1D_FN
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _FLA_CHUNK_GATED_DELTA_RULE
else:
    _FLA_CAUSAL_CONV1D_FN = None
    _FLA_CHUNK_GATED_DELTA_RULE = None

WORLD_SIZE = 2
LOGITS_RTOL = 5e-3
LOGITS_ATOL = 5e-3
LOSS_ATOL = 5e-3
GRAD_RTOL = 5e-3
GRAD_ATOL = 2e-3
_HAS_FLA_PREFILL = bool(_HAS_QWEN35 and _FLA_CAUSAL_CONV1D_FN is not None and _FLA_CHUNK_GATED_DELTA_RULE is not None)


def _hf_compatible_fla_causal_conv1d_fn(x, weight, bias=None, activation=None, seq_idx=None):
    del seq_idx
    mixed_qkv, _ = _FLA_CAUSAL_CONV1D_FN(
        x=x.transpose(1, 2).contiguous(),
        weight=weight,
        bias=bias,
        activation=activation,
        backend='triton',
    )
    if mixed_qkv.dim() == 2:
        mixed_qkv = mixed_qkv.unsqueeze(0)
    return mixed_qkv.transpose(1, 2).contiguous()


def _force_fla_causal_conv(model: Qwen3_5ForCausalLM) -> Qwen3_5ForCausalLM:
    for layer in model.model.layers:
        linear_attn = getattr(layer, 'linear_attn', None)
        if linear_attn is not None:
            linear_attn.causal_conv1d_fn = _hf_compatible_fla_causal_conv1d_fn
            linear_attn.chunk_gated_delta_rule = _FLA_CHUNK_GATED_DELTA_RULE
    return model


def _force_packed_linear_attention(model: Qwen3_5ForCausalLM, position_ids: torch.Tensor) -> Qwen3_5ForCausalLM:
    packed_cu_seqlens = get_cu_seqlens_from_position_ids(position_ids).to(torch.int32)

    def _make_packed_forward(cu_seqlens: torch.Tensor):

        def _packed_forward(mod, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
            packed_ctx = SimpleNamespace(
                world_size=1,
                sp_world_size=1,
                extra_kwargs={
                    'is_packed': True,
                    'cu_seq_lens_q': cu_seqlens.to(dtype=torch.int32, device=hidden_states.device),
                })
            return Qwen3_5GatedDeltaNetUlyssesPatch._run_forward(
                mod,
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
                cu_seq_lens_q=packed_ctx.extra_kwargs['cu_seq_lens_q'],
                sequence_parallel_context=packed_ctx,
            )

        return _packed_forward

    for layer in model.model.layers:
        linear_attn = getattr(layer, 'linear_attn', None)
        if linear_attn is not None:
            linear_attn.forward = MethodType(_make_packed_forward(packed_cu_seqlens), linear_attn)
    return model


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
    model = _force_fla_causal_conv(model)
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


def _make_packed_train_batch(device: torch.device):
    input_ids = torch.tensor([[11, 12, 13, 21, 22, 23, 24, 25]], device=device, dtype=torch.long)
    attention_mask = None
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]], device=device, dtype=torch.long)
    labels = torch.tensor([[12, 13, -100, 22, 23, 24, 25, -100]], device=device, dtype=torch.long)
    return input_ids, attention_mask, position_ids, labels


def _get_qkv_weight(model: Qwen3_5ForCausalLM) -> torch.nn.Parameter:
    for layer in model.model.layers:
        linear_attn = getattr(layer, 'linear_attn', None)
        if linear_attn is not None:
            return linear_attn.in_proj_qkv.weight
    raise AssertionError('No linear attention layer found in Qwen3.5 test model.')


def _compute_training_path_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    strategy: SequenceParallelStrategy | None = None,
) -> tuple[torch.Tensor, int]:
    masked_labels = labels.masked_fill(labels == -100, 0)
    loss_inputs = {'labels': labels}
    loss_outputs = {'logps': selective_log_softmax(logits, masked_labels)}
    if strategy is not None:
        loss_inputs, loss_outputs = strategy.gather_loss_tensors(loss_inputs, loss_outputs)
    result = CrossEntropyLoss(reduction='sum')(loss_inputs, loss_outputs)
    num_tokens = result['num_tokens']
    if torch.is_tensor(num_tokens):
        num_tokens = int(num_tokens.item())
    else:
        num_tokens = int(num_tokens)
    return result['loss'], num_tokens


def _average_qkv_grad_over_group(model: Qwen3_5ForCausalLM, group: dist.ProcessGroup | None) -> torch.Tensor:
    grad = _get_qkv_weight(model).grad
    if grad is None:
        raise AssertionError('No qkv gradient collected from Qwen3.5 linear attention layer.')
    reduced = grad.detach().float().contiguous()
    if group is None:
        return reduced.cpu()
    group_world_size = dist.get_world_size(group)
    if group_world_size > 1:
        dist.all_reduce(reduced, group=group)
        reduced.div_(group_world_size)
    return reduced.cpu()


def _run_prefill_alignment_worker(rank: int,
                                  world_size: int,
                                  port: int,
                                  attn_implementation: str = 'sdpa',
                                  layer_types: list[str] | None = None,
                                  packed: bool = False):
    device = _init_dist(rank, world_size, port)
    try:
        _set_determinism(1234)

        baseline_model = _build_tiny_qwen35(device, attn_implementation=attn_implementation, layer_types=layer_types)
        sp_model = copy.deepcopy(baseline_model)
        input_ids, attention_mask, position_ids, labels = (
            _make_packed_train_batch(device) if packed else _make_train_batch(device))
        if packed:
            baseline_model = _force_packed_linear_attention(baseline_model, position_ids)

        baseline_outputs = baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        baseline_logits = baseline_outputs.logits.detach().float()
        baseline_loss_sum, baseline_num_tokens = _compute_training_path_loss(baseline_outputs.logits, labels)
        baseline_loss_sum.backward()
        baseline_loss = baseline_loss_sum / max(baseline_num_tokens, 1)
        baseline_qkv_grad = _get_qkv_weight(baseline_model).grad.detach().float().cpu() / max(baseline_num_tokens, 1)

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

        local_logits = sp_outputs.logits
        sp_loss_sum, sp_num_tokens = _compute_training_path_loss(local_logits, local_labels, strategy)
        sp_loss = sp_loss_sum / max(sp_num_tokens, 1)
        if not torch.allclose(sp_loss.detach(), baseline_loss.detach(), atol=LOSS_ATOL, rtol=0):
            raise AssertionError(
                f'prefill loss mismatch on rank {rank}: baseline={baseline_loss.item()} sp={sp_loss.item()}')
        sp_loss_sum.backward()

        if sp_num_tokens != baseline_num_tokens:
            raise AssertionError(
                f'prefill num_tokens mismatch on rank {rank}: baseline={baseline_num_tokens} sp={sp_num_tokens}')
        sp_qkv_grad = _average_qkv_grad_over_group(sp_model, sequence_parallel._data_rank_group) / max(sp_num_tokens, 1)
        if not torch.allclose(sp_qkv_grad, baseline_qkv_grad, rtol=GRAD_RTOL, atol=GRAD_ATOL):
            max_diff = (sp_qkv_grad - baseline_qkv_grad).abs().max().item()
            raise AssertionError(f'qkv grad mismatch on rank {rank}: max_diff={max_diff}')
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


@unittest.skipUnless(_HAS_QWEN35, 'transformers Qwen3.5 is not available in this environment')
@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE, 'requires 2 CUDA devices')
@unittest.skipUnless(_HAS_FLA_PREFILL, 'requires flash-linear-attention kernels for Qwen3.5 SP linear attention tests')
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

    @unittest.skipUnless(is_flash_attn_available(), 'requires flash_attention_2 support in transformers')
    def test_qwen35_linear_attention_packed_prefill_logits_and_qkv_grad_alignment(self):
        port = _find_free_port()
        mp.spawn(
            _run_prefill_alignment_worker,
            args=(WORLD_SIZE, port, 'flash_attention_2', ['linear_attention', 'linear_attention'], True),
            nprocs=WORLD_SIZE,
            join=True,
        )


if __name__ == '__main__':
    unittest.main()
