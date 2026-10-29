# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import os
import pytest
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_flash_attention_utils import is_flash_attn_available
from types import SimpleNamespace

from twinkle.loss import CrossEntropyLoss
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelStrategy
from twinkle.utils import DeviceMesh, Platform, ensure_hccl_socket_env, selective_log_softmax, torch_util



LOGITS_RTOL = 1e-2
LOGITS_ATOL = 5e-3
LOSS_ATOL = 5e-3
GRAD_RTOL = 2e-2
GRAD_ATOL = 1e-2


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _make_labels(input_ids: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    labels[..., :-1] = input_ids[..., 1:]
    return labels


def _make_padded_labels(input_ids: torch.Tensor, valid_lengths: list[int]) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    for row_idx, valid_len in enumerate(valid_lengths):
        if valid_len <= 1:
            continue
        labels[row_idx, :valid_len - 1] = input_ids[row_idx, 1:valid_len]
    return labels


def _make_case(case_name: str) -> dict:
    cases = {
        'sp_only': {
            'expected_mode': 'sp_only',
            'world_size': 2,
            'ulysses_size': 2,
            'num_attention_heads': 8,
            'hidden_size': 128,
            'seq_len': 769,
        },
        'sp_only_multi_sample': {
            'expected_mode': 'sp_only',
            'world_size': 2,
            'ulysses_size': 2,
            'num_attention_heads': 8,
            'hidden_size': 128,
            'seq_len': 769,
            'batch_size': 3,
        },
        'sp_only_multi_sample_masked': {
            'expected_mode': 'sp_only',
            'world_size': 2,
            'ulysses_size': 2,
            'num_attention_heads': 8,
            'hidden_size': 128,
            'seq_len': 769,
            'batch_size': 3,
            'valid_lengths': [769, 513, 257],
        },
        'cp_only': {
            'expected_mode': 'cp_only',
            'world_size': 2,
            'ulysses_size': 2,
            'num_attention_heads': 1,
            'hidden_size': 64,
            'seq_len': 769,
        },
        'cp_sp': {
            'expected_mode': 'cp_sp',
            'world_size': 4,
            'ulysses_size': 4,
            'num_attention_heads': 6,
            'hidden_size': 96,
            'seq_len': 769,
        },
        'sp_only_memory': {
            'expected_mode': 'sp_only',
            'world_size': 4,
            'ulysses_size': 2,
            'num_attention_heads': 8,
            'hidden_size': 128,
            'num_hidden_layers': 2,
            'seq_lens': [255, 511, 1023],
            'batch_sizes': [1, 2, 4],
        },
        'cp_only_memory': {
            'expected_mode': 'cp_only',
            'world_size': 2,
            'ulysses_size': 2,
            'num_attention_heads': 1,
            'hidden_size': 128,
            'num_hidden_layers': 4,
            'seq_lens': [511, 1023, 2047],
            'batch_sizes': [1],
        },
        'cp_sp_memory': {
            'expected_mode': 'cp_sp',
            'world_size': 4,
            'ulysses_size': 4,
            # gcd(6, 4) = 2 -> sp=2, rp=2
            'num_attention_heads': 6,
            'hidden_size': 192,
            'num_hidden_layers': 4,
            'seq_lens': [511, 1023, 2047],
            'batch_sizes': [1],
        },
    }
    return copy.deepcopy(cases[case_name])


def _validate_case_config(case: dict) -> tuple[str, int, int]:
    hidden_size = int(case['hidden_size'])
    num_heads = int(case['num_attention_heads'])
    ulysses_size = int(case['ulysses_size'])
    expected_mode = case.get('expected_mode')

    if hidden_size % num_heads != 0:
        raise ValueError(f'Invalid test case config: hidden_size ({hidden_size}) must be divisible by '
                         f'num_attention_heads ({num_heads}).')

    head_dim = hidden_size // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f'Invalid test case config: head_dim ({head_dim}) must be even for RoPE. '
                         f'Got hidden_size={hidden_size}, num_attention_heads={num_heads}.')

    sp_world_size = math.gcd(num_heads, ulysses_size)
    rp_world_size = ulysses_size // sp_world_size
    mode = 'sp_only'
    if rp_world_size > 1 and sp_world_size == 1:
        mode = 'cp_only'
    elif rp_world_size > 1 and sp_world_size > 1:
        mode = 'cp_sp'

    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f'Invalid test case config: expected {expected_mode}, but derived {mode}. '
                         f'Got ulysses_size={ulysses_size}, num_attention_heads={num_heads}, '
                         f'sp_world_size={sp_world_size}, rp_world_size={rp_world_size}.')
    return mode, sp_world_size, rp_world_size


def _get_runtime_backend() -> dict | None:
    if torch.cuda.is_available():
        return {
            'device_type': 'cuda',
            'dist_backend': 'nccl',
            'device_count': int(torch.cuda.device_count()),
            'label': 'CUDA',
        }
    if torch_util.is_npu_available():
        return {
            'device_type': 'npu',
            'dist_backend': 'hccl',
            'device_count': int(torch.npu.device_count()),
            'label': 'NPU',
        }
    return None


def _get_device_module(device_type: str):
    if device_type == 'cuda':
        return torch.cuda
    if device_type == 'npu':
        return torch.npu
    raise ValueError(f'Unsupported device_type for derived ring tests: {device_type}')


def _supports_peak_memory_stats(device_type: str) -> bool:
    device_module = _get_device_module(device_type)
    required_apis = ('empty_cache', 'reset_peak_memory_stats', 'synchronize', 'max_memory_allocated')
    return all(hasattr(device_module, name) for name in required_apis)


def _get_model_dtype(device_type: str) -> torch.dtype:
    if device_type == 'cuda':
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device_type == 'npu':
        is_bf16_supported = getattr(torch.npu, 'is_bf16_supported', None)
        if callable(is_bf16_supported):
            try:
                if is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16
    raise ValueError(f'Unsupported device_type for derived ring tests: {device_type}')


def _seed_backend(seed: int, device_type: str) -> None:
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif device_type == 'npu':
        torch.npu.manual_seed_all(seed)


def _build_tiny_llama(case: dict,
                      device: torch.device,
                      device_type: str,
                      attn_implementation: str = 'flash_attention_2') -> LlamaForCausalLM:
    _validate_case_config(case)
    hidden_size = int(case['hidden_size'])
    num_heads = int(case['num_attention_heads'])
    num_hidden_layers = int(case.get('num_hidden_layers', 1))
    dtype = _get_model_dtype(device_type)
    max_seq_len = int(case.get('seq_len', max(case.get('seq_lens', [1024])))) + 32
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=max_seq_len,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
    )
    config._attn_implementation = attn_implementation
    model = LlamaForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _make_strategy(model: LlamaForCausalLM, device_mesh: DeviceMesh, ulysses_size: int) -> SequenceParallelStrategy:
    strategy = SequenceParallelStrategy(
        device_mesh=device_mesh,
        sp_config={
            'enabled': True,
            'ulysses_size': ulysses_size,
            'gather_logits': True,
        },
        model=model,
        tokenizer_id=None,
    )
    strategy._tokenizer = SimpleNamespace(pad_token_id=0)
    strategy.initialize()
    return strategy


def _prepare_label_inputs(strategy: SequenceParallelStrategy, input_ids: torch.Tensor,
                          position_ids: torch.Tensor) -> torch.Tensor:
    labels = _make_labels(input_ids)
    processed = strategy.prepare_model_inputs({
        'input_ids': input_ids,
        'position_ids': position_ids,
        'labels': labels,
    })
    return processed['labels']


def _build_precision_inputs(
        case: dict, vocab_size: int,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    seq_len = int(case['seq_len'])
    batch_size = int(case.get('batch_size', 1))
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    valid_lengths = case.get('valid_lengths')
    if not valid_lengths:
        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
        labels = _make_labels(input_ids)
        return input_ids, position_ids, None, labels

    if len(valid_lengths) != batch_size:
        raise ValueError(f'valid_lengths length ({len(valid_lengths)}) must equal batch_size ({batch_size}).')

    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
    for row_idx, valid_len in enumerate(valid_lengths):
        if valid_len <= 0 or valid_len > seq_len:
            raise ValueError(f'valid_len must be in [1, seq_len], got valid_len={valid_len}, seq_len={seq_len}')
        input_ids[row_idx, :valid_len] = torch.randint(low=0, high=vocab_size, size=(valid_len, ), device=device)
        attention_mask[row_idx, :valid_len] = 1
    labels = _make_padded_labels(input_ids, [int(length) for length in valid_lengths])
    return input_ids, position_ids, attention_mask, labels


def _compute_training_path_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    strategy: SequenceParallelStrategy | None = None,
) -> tuple[torch.Tensor, int]:
    masked_labels = labels.masked_fill(labels == -100, 0)
    loss_inputs = {'labels': labels}
    loss_outputs = {'logps': selective_log_softmax(logits, masked_labels)}
    if strategy is not None:
        loss_inputs['_sequence_layout'] = {'sharded': ('labels',)}
        loss_inputs, loss_outputs = strategy.gather_loss_tensors(loss_inputs, loss_outputs)
    result = CrossEntropyLoss(reduction='sum')(loss_inputs, loss_outputs)
    num_tokens = result['num_tokens']
    if torch.is_tensor(num_tokens):
        num_tokens = int(num_tokens.item())
    else:
        num_tokens = int(num_tokens)
    return result['loss'], num_tokens


def _normalize_grad_dict(grads: dict[str, torch.Tensor], num_tokens: int) -> dict[str, torch.Tensor]:
    denom = float(max(num_tokens, 1))
    return {name: grad / denom for name, grad in grads.items()}


def _average_model_grads_over_group(model: LlamaForCausalLM, group: dist.ProcessGroup | None) -> None:
    if group is None:
        return
    group_world_size = dist.get_world_size(group)
    if group_world_size <= 1:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, group=group)
        param.grad.div_(group_world_size)


def _collect_attention_param_grads(model: LlamaForCausalLM) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if '.self_attn.' not in name:
            continue
        if param.grad is None:
            continue
        grads[name] = param.grad.detach().float().cpu()
    if not grads:
        raise AssertionError('No attention gradients were collected from the model.')
    return grads


def _assert_grad_dict_close(case_name: str, rank: int, baseline_grads: dict[str, torch.Tensor],
                            sp_grads: dict[str, torch.Tensor]):
    baseline_keys = sorted(baseline_grads.keys())
    sp_keys = sorted(sp_grads.keys())
    if baseline_keys != sp_keys:
        raise AssertionError(
            f'{case_name} attention grad keys mismatch on rank {rank}: baseline={baseline_keys}, sp={sp_keys}')
    for key in baseline_keys:
        baseline = baseline_grads[key]
        current = sp_grads[key]
        if not torch.allclose(current, baseline, rtol=GRAD_RTOL, atol=GRAD_ATOL):
            max_diff = (current - baseline).abs().max().item()
            raise AssertionError(f'{case_name} attention grad mismatch on rank {rank} for {key}: max_diff={max_diff}')


def _assert_logits_close(case_name: str, rank: int, baseline_logits: torch.Tensor, sp_logits: torch.Tensor,
                         seq_len: int, attention_mask: torch.Tensor | None):
    baseline_slice = baseline_logits[:, :seq_len]
    sp_slice = sp_logits[:, :seq_len]
    if attention_mask is None:
        if not torch.allclose(sp_slice, baseline_slice, rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
            diff = (sp_slice - baseline_slice).abs().max().item()
            raise AssertionError(f'{case_name} logits mismatch on rank {rank}: max_diff={diff}')
        return

    mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(baseline_slice)
    if not torch.allclose(sp_slice[mask], baseline_slice[mask], rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
        diff = (sp_slice[mask] - baseline_slice[mask]).abs().max().item()
        raise AssertionError(f'{case_name} masked logits mismatch on rank {rank}: max_diff={diff}')


def _init_dist(rank: int, world_size: int, port: int, backend_config: dict):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)

    device_type = backend_config['device_type']
    dist_backend = backend_config['dist_backend']
    if dist_backend == 'hccl':
        ensure_hccl_socket_env(port)
    device = torch.device(Platform.get_local_device(rank, platform=device_type))
    _get_device_module(device_type).set_device(rank)
    dist.init_process_group(
        backend=dist_backend,
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}',
        device_id=device,
        timeout=timedelta(minutes=15),
    )
    return device


def _measure_peak_memory(
    model: LlamaForCausalLM,
    strategy: SequenceParallelStrategy,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    device_type: str,
) -> int:
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    local_labels = _prepare_label_inputs(strategy, input_ids, position_ids)

    model.zero_grad(set_to_none=True)
    device_module = _get_device_module(device_type)
    device_module.empty_cache()
    device_module.reset_peak_memory_stats(device)
    prepared = strategy.prepare_model_inputs({'input_ids': input_ids, 'position_ids': position_ids})
    prepared.pop('_sequence_layout')
    with strategy.context():
        outputs = model(**prepared, use_cache=False)
    local_logits = outputs.logits
    loss_sum, _ = _compute_training_path_loss(local_logits, local_labels, strategy)
    loss_sum.backward()
    device_module.synchronize(device)

    peak = torch.tensor([int(device_module.max_memory_allocated(device))], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return int(peak.item())


def _format_memory_table(case_name: str, peaks: list[dict]) -> str:
    header = f'[{case_name}] peak memory'
    columns = (
        'batch_size',
        'seq_len',
        'baseline_bytes',
        'baseline_mib',
        'peak_bytes',
        'peak_mib',
        'delta_bytes',
        'saving_ratio_pct',
    )
    rows = []
    for row in peaks:
        rows.append((
            str(row['batch_size']),
            str(row['seq_len']),
            str(row['baseline_bytes']),
            f"{row['baseline_mib']:.2f}",
            str(row['peak_bytes']),
            f"{row['peak_mib']:.2f}",
            str(row['delta_bytes']),
            f"{row['saving_ratio_pct']:.2f}",
        ))

    widths = [len(col) for col in columns]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def _fmt(values):
        return ' | '.join(value.ljust(widths[i]) for i, value in enumerate(values))

    lines = [
        header,
        _fmt(columns),
        '-+-'.join('-' * width for width in widths),
    ]
    lines.extend(_fmt(row) for row in rows)
    return '\n'.join(lines)


def _run_precision_worker(rank: int,
                          world_size: int,
                          port: int,
                          case_name: str,
                          backend_config: dict,
                          attn_implementation: str = 'flash_attention_2'):
    device = _init_dist(rank, world_size, port, backend_config)
    try:
        _seed_backend(1234, backend_config['device_type'])
        case = _make_case(case_name)

        base_model = _build_tiny_llama(case, device, backend_config['device_type'], attn_implementation)
        sp_model = _build_tiny_llama(case, device, backend_config['device_type'], attn_implementation)
        sp_model.load_state_dict(base_model.state_dict())

        seq_len = int(case['seq_len'])
        input_ids, position_ids, attention_mask, labels = _build_precision_inputs(case,
                                                                                  int(base_model.config.vocab_size),
                                                                                  device)

        base_model.zero_grad(set_to_none=True)
        base_outputs = base_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        base_logits = base_outputs.logits.detach().float()
        base_loss_sum, base_num_tokens = _compute_training_path_loss(base_outputs.logits, labels)
        base_loss_sum.backward()
        base_loss = base_loss_sum / max(base_num_tokens, 1)
        base_attention_grads = _normalize_grad_dict(_collect_attention_param_grads(base_model), base_num_tokens)

        device_mesh = DeviceMesh.from_sizes(
            fsdp_size=world_size,
            dp_size=1,
            ulysses_size=int(case['ulysses_size']),
            device_type=backend_config['device_type'],
        )
        strategy = _make_strategy(sp_model, device_mesh, int(case['ulysses_size']))
        processed_inputs = strategy.prepare_model_inputs({
            'input_ids': input_ids, 'position_ids': position_ids, 'labels': labels,
            'attention_mask': attention_mask,
        })
        local_labels = processed_inputs['labels']

        sp_model.zero_grad(set_to_none=True)
        processed_inputs.pop('labels')
        processed_inputs.pop('_sequence_layout')
        with strategy.context():
            sp_outputs = sp_model(**processed_inputs, use_cache=False)
        local_logits = sp_outputs.logits
        _, gathered_outputs = strategy.gather_loss_tensors({}, sp_outputs)
        sp_logits = gathered_outputs.logits.detach().float()

        sp_loss_sum, sp_num_tokens = _compute_training_path_loss(local_logits, local_labels, strategy)
        sp_loss_sum.backward()
        global_loss = sp_loss_sum / max(sp_num_tokens, 1)
        _average_model_grads_over_group(sp_model, strategy._ctx._data_rank_group)
        sp_attention_grads = _normalize_grad_dict(_collect_attention_param_grads(sp_model), sp_num_tokens)

        if sp_num_tokens != base_num_tokens:
            raise AssertionError(
                f'{case_name} num_tokens mismatch on rank {rank}: sp={sp_num_tokens} base={base_num_tokens}')

        _assert_logits_close(case_name, rank, base_logits, sp_logits, seq_len, attention_mask)
        if abs(global_loss.item() - base_loss.item()) > LOSS_ATOL:
            raise AssertionError(
                f'{case_name} loss mismatch on rank {rank}: sp={global_loss.item()} base={base_loss.item()}')
        _assert_grad_dict_close(case_name, rank, base_attention_grads, sp_attention_grads)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_memory_worker(rank: int, world_size: int, port: int, case_name: str, backend_config: dict):
    device = _init_dist(rank, world_size, port, backend_config)
    try:
        _seed_backend(1234, backend_config['device_type'])
        case = _make_case(case_name)
        baseline_device_mesh = DeviceMesh.from_sizes(
            fsdp_size=world_size,
            dp_size=1,
            ulysses_size=1,
            device_type=backend_config['device_type'],
        )
        baseline_model = _build_tiny_llama(case, device, backend_config['device_type'])
        baseline_strategy = _make_strategy(baseline_model, baseline_device_mesh, 1)

        baseline_peaks = {}
        for batch_size in case['batch_sizes']:
            for seq_len in case['seq_lens']:
                baseline_peak = _measure_peak_memory(
                    baseline_model,
                    baseline_strategy,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    device_type=backend_config['device_type'],
                )
                baseline_peaks[(int(batch_size), int(seq_len))] = int(baseline_peak)

        del baseline_model
        del baseline_strategy
        _get_device_module(backend_config['device_type']).empty_cache()

        device_mesh = DeviceMesh.from_sizes(
            fsdp_size=world_size,
            dp_size=1,
            ulysses_size=int(case['ulysses_size']),
            device_type=backend_config['device_type'],
        )
        model = _build_tiny_llama(case, device, backend_config['device_type'])
        strategy = _make_strategy(model, device_mesh, int(case['ulysses_size']))

        peaks = []
        for batch_size in case['batch_sizes']:
            for seq_len in case['seq_lens']:
                peak = _measure_peak_memory(
                    model,
                    strategy,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    device_type=backend_config['device_type'],
                )
                if rank == 0:
                    baseline_peak = baseline_peaks[(int(batch_size), int(seq_len))]
                    delta_bytes = int(peak) - int(baseline_peak)
                    saving_ratio_pct = 0.0
                    if baseline_peak > 0:
                        saving_ratio_pct = (float(baseline_peak) - float(peak)) / float(baseline_peak) * 100.0
                    peaks.append({
                        'batch_size': int(batch_size),
                        'seq_len': int(seq_len),
                        'baseline_bytes': int(baseline_peak),
                        'baseline_mib': float(baseline_peak) / (1024**2),
                        'peak_bytes': int(peak),
                        'peak_mib': float(peak) / (1024**2),
                        'delta_bytes': delta_bytes,
                        'saving_ratio_pct': saving_ratio_pct,
                    })

        if rank == 0:
            for key in ('peak_bytes', 'baseline_bytes'):
                by_batch = {}
                for row in peaks:
                    by_batch.setdefault(row['batch_size'], []).append(row)
                for rows in by_batch.values():
                    rows.sort(key=lambda item: item['seq_len'])
                    for prev, cur in zip(rows, rows[1:]):
                        if cur[key] < prev[key]:
                            raise AssertionError(
                                f'{case_name} {key} should be non-decreasing with seq_len, got {prev} then {cur}')

                by_seq = {}
                for row in peaks:
                    by_seq.setdefault(row['seq_len'], []).append(row)
                for rows in by_seq.values():
                    rows.sort(key=lambda item: item['batch_size'])
                    for prev, cur in zip(rows, rows[1:]):
                        if cur[key] < prev[key]:
                            raise AssertionError(
                                f'{case_name} {key} should be non-decreasing with batch_size, got {prev} then {cur}')

            print(_format_memory_table(case_name, peaks))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.accel(2)
class TestDerivedRingPrecision:

    def _get_backend_or_skip(self, world_size: int = 4) -> dict:
        backend = _get_runtime_backend()
        if backend is None:
            pytest.skip('CUDA or NPU is required for derived ring runtime tests.')
        if backend['device_count'] < world_size:
            pytest.skip(f'Requires at least {world_size} {backend["label"]} devices.')
        return backend

    def _require_attn_impl_or_skip(self, backend: dict, attn_implementation: str) -> None:
        if attn_implementation == 'flash_attention_2' and not is_flash_attn_available():
            if backend['device_type'] == 'npu':
                pytest.skip(
                    'Derived ring runtime tests currently require flash_attention_2, which is unavailable on NPU in '
                    'this environment.')
            pytest.skip('flash_attention_2 is required for derived ring runtime tests.')

    def test_cp_only_precision_alignment(self):
        case = _make_case('cp_only')
        world_size = int(case['world_size'])
        backend = self._get_backend_or_skip(world_size)
        self._require_attn_impl_or_skip(backend, 'flash_attention_2')
        port = _find_free_port()
        mp.spawn(
            _run_precision_worker,
            args=(world_size, port, 'cp_only', backend, 'flash_attention_2'),
            nprocs=world_size,
            join=True)

    def test_sp_only_multi_sample_precision_alignment(self):
        case = _make_case('sp_only_multi_sample')
        world_size = int(case['world_size'])
        backend = self._get_backend_or_skip(world_size)
        self._require_attn_impl_or_skip(backend, 'flash_attention_2')
        port = _find_free_port()
        mp.spawn(
            _run_precision_worker,
            args=(world_size, port, 'sp_only_multi_sample', backend, 'flash_attention_2'),
            nprocs=world_size,
            join=True)

    def test_sp_only_multi_sample_masked_precision_alignment(self):
        case = _make_case('sp_only_multi_sample_masked')
        world_size = int(case['world_size'])
        backend = self._get_backend_or_skip(world_size)
        self._require_attn_impl_or_skip(backend, 'flash_attention_2')
        port = _find_free_port()
        mp.spawn(
            _run_precision_worker,
            args=(world_size, port, 'sp_only_multi_sample_masked', backend, 'flash_attention_2'),
            nprocs=world_size,
            join=True)

    def test_cp_sp_precision_alignment(self):
        case = _make_case('cp_sp')
        world_size = int(case['world_size'])
        backend = self._get_backend_or_skip(world_size)
        self._require_attn_impl_or_skip(backend, 'flash_attention_2')
        port = _find_free_port()
        mp.spawn(
            _run_precision_worker,
            args=(world_size, port, 'cp_sp', backend, 'flash_attention_2'),
            nprocs=world_size,
            join=True)


@pytest.mark.slow
@pytest.mark.accel(2)
class TestDerivedRingMemoryProfile:

    def _get_backend_or_skip(self, world_size: int = 4) -> dict:
        if os.environ.get('TWINKLE_RUN_MEMORY_TESTS', '0') != '1':
            pytest.skip('Set TWINKLE_RUN_MEMORY_TESTS=1 to run derived ring memory profile tests.')
        backend = _get_runtime_backend()
        if backend is None:
            pytest.skip('CUDA or NPU is required for derived ring memory tests.')
        if backend['device_count'] < world_size:
            pytest.skip(f'Requires at least {world_size} {backend["label"]} devices.')
        if not is_flash_attn_available():
            if backend['device_type'] == 'npu':
                pytest.skip(
                    'Derived ring memory tests currently require flash_attention_2, which is unavailable on NPU in '
                    'this environment.')
            pytest.skip('flash_attention_2 is required for derived ring memory tests.')
        if not _supports_peak_memory_stats(backend['device_type']):
            pytest.skip(f'{backend["label"]} peak-memory stats are unavailable in this environment.')
        return backend

    def test_sp_only_memory_profile_grid(self):
        case = _make_case('sp_only_memory')
        world_size = int(case['world_size'])
        backend = self._get_backend_or_skip(world_size)
        port = _find_free_port()
        mp.spawn(_run_memory_worker, args=(world_size, port, 'sp_only_memory', backend), nprocs=world_size, join=True)

    def test_cp_only_memory_profile_grid(self):
        case = _make_case('cp_only_memory')
        world_size = int(case['world_size'])
        backend = self._get_backend_or_skip(world_size)
        port = _find_free_port()
        mp.spawn(_run_memory_worker, args=(world_size, port, 'cp_only_memory', backend), nprocs=world_size, join=True)

    def test_cp_sp_memory_profile_grid(self):
        case = _make_case('cp_sp_memory')
        world_size = int(case['world_size'])
        backend = self._get_backend_or_skip(world_size)
        port = _find_free_port()
        mp.spawn(_run_memory_worker, args=(world_size, port, 'cp_sp_memory', backend), nprocs=world_size, join=True)


def test_explicit_sp_context_restores_nested_and_exception_paths():
    from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel, _current_ctx, _attention_context
    first, second = SequenceParallel(), SequenceParallel()
    first.model_name = 'policy'
    assert _current_ctx() is None
    with first.context():
        with pytest.raises(RuntimeError, match='unregistered attention'):
            _attention_context(torch.nn.Linear(2, 2))
        with pytest.raises(ValueError, match='forward failed'):
            with second.context():
                assert _current_ctx() is second
                raise ValueError('forward failed')
        assert _current_ctx() is first
    assert _current_ctx() is None


def test_flash_padding_metadata_preserves_real_packed_boundaries():
    from twinkle.model.transformers.strategy.sequence_parallel import _call_flash_attention
    positions = torch.tensor([[0, 1, 2, -1, 0, 1, -1, -1]])
    original = positions.clone()
    normalized = _call_flash_attention(lambda **kw: kw['position_ids'], position_ids=positions)
    assert normalized.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3]]
    # Equivalent boundary rule used by TF 5.15.1; the tail must not become a new sequence.
    assert (normalized == normalized.min()).nonzero()[:, -1].tolist() == [0, 4]
    torch.testing.assert_close(positions, original)


def _slim_cpu_worker(rank, port, checkpoint):
    import torch.nn.functional as F
    from torch.utils.checkpoint import DefaultDeviceType
    from twinkle.processor import InputProcessor
    from twinkle.model.transformers.strategy.sequence_parallel import _current_ctx, SequenceParallel
    DefaultDeviceType.set_device_type('cpu')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                             attention_dropout=0., use_cache=False)
        config._attn_implementation = 'sdpa'
        base_p, base_r = LlamaForCausalLM(config).train(), LlamaForCausalLM(config).train()
        policy, reference = copy.deepcopy(base_p), copy.deepcopy(base_r)
        if checkpoint is not None:
            for model in (base_p, base_r, policy, reference):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': checkpoint})
        mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=1, ulysses_size=2, device_type='cpu')
        policy_sp, ref_sp = _make_strategy(policy, mesh, 2), _make_strategy(reference, mesh, 2)
        inputs_p = torch.tensor([[1, 3, 5, 7, 9, 11, 13]])
        inputs_r = torch.tensor([[1, 4, 6, 8, 10, 12, 14, 16, 18]])

        def baseline(model, ids):
            logits = model(input_ids=ids, use_cache=False).logits
            labels = _make_labels(ids)
            loss = F.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1), ignore_index=-100)
            return logits, loss

        def parallel(strategy, model, ids):
            inputs = strategy.prepare_model_inputs({'input_ids': ids, 'labels': _make_labels(ids)})
            labels, layout = inputs.pop('labels'), inputs.pop('_sequence_layout')
            with strategy.context():
                output = model(**inputs, use_cache=False)
            inputs.update(labels=labels, _sequence_layout=layout)
            inputs, output = strategy.gather_loss_tensors(inputs, output)
            loss = F.cross_entropy(output.logits.reshape(-1, 32), inputs['labels'].reshape(-1), ignore_index=-100)
            return output.logits, loss

        baseline_p, loss_p = baseline(base_p, inputs_p)
        baseline_r, loss_r = baseline(base_r, inputs_r)
        logits_p, parallel_p = parallel(policy_sp, policy, inputs_p)
        logits_r, parallel_r = parallel(ref_sp, reference, inputs_r)
        # Reuse a context with another batch before the first backward: recompute must restore its own positions.
        parallel(policy_sp, policy, inputs_r)
        for actual, expected in ((logits_p, baseline_p), (logits_r, baseline_r),
                                 (parallel_p, loss_p), (parallel_r, loss_r)):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        parallel_r.backward()
        parallel_p.backward()
        loss_r.backward()
        loss_p.backward()
        for model, base in ((policy, base_p), (reference, base_r)):
            for (_, parameter), (_, original) in zip(model.named_parameters(), base.named_parameters()):
                assert parameter.grad is not None and original.grad is not None
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(2)
                torch.testing.assert_close(parameter.grad, original.grad, rtol=3e-4, atol=3e-6)
        assert _current_ctx() is None
        # The non-SP baseline still uses the origin after all global wrappers have been installed.
        again, _ = baseline(base_p, inputs_p)
        torch.testing.assert_close(again, baseline_p, rtol=0, atol=0)

    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('checkpoint', [None, False, True])
def test_slim_sp_cpu_interleaved_gradients_and_recompute(checkpoint):
    mp.spawn(_slim_cpu_worker, args=(_find_free_port(), checkpoint), nprocs=2, join=True)


def _slim_qwen_vl_worker(rank, port):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from torch.utils.checkpoint import DefaultDeviceType
    DefaultDeviceType.set_device_type('cpu')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=2)
    try:
        torch.manual_seed(81)
        import transformers
        text_config = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=2, head_dim=16)
        # 4.x names this test configuration rope_scaling; 5.x uses rope_parameters.
        rope_key = 'rope_scaling' if int(transformers.__version__.split('.')[0]) < 5 else 'rope_parameters'
        text_config[rope_key] = {'rope_type': 'default', 'rope_theta': 10000., 'mrope_section': [2, 3, 3]}
        config = Qwen3VLConfig(
            text_config=text_config,
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                               patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
                               out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0]),
            image_token_id=60, video_token_id=61, vision_start_token_id=59, vision_end_token_id=58)
        config._attn_implementation = 'sdpa'
        baseline = Qwen3VLForConditionalGeneration(config).train()
        model = torch.nn.parallel.DistributedDataParallel(copy.deepcopy(baseline))
        mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=1, ulysses_size=2, device_type='cpu')
        strategy = _make_strategy(model, mesh, 2)
        ids = torch.tensor([[1, 59, 60, 60, 60, 60, 58, 5, 6]])
        inputs = dict(input_ids=ids, pixel_values=torch.randn(16, 12), image_grid_thw=torch.tensor([[1, 4, 4]]),
                      mm_token_type_ids=torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0]]), use_cache=False)
        expected = baseline(**inputs).logits
        prepared = strategy.prepare_model_inputs(inputs)
        prepared.pop('_sequence_layout')
        assert 'position_ids' not in prepared
        with strategy.context():
            output = model(**prepared)
        _, gathered = strategy.gather_loss_tensors({}, output)
        torch.testing.assert_close(gathered.logits, expected, rtol=3e-5, atol=3e-6)
        expected.square().mean().backward()
        gathered.logits.square().mean().backward()
        for (name, parameter), (_, original) in zip(model.named_parameters(), baseline.named_parameters()):
            if original.grad is None:
                assert parameter.grad is None
                continue
            assert parameter.grad is not None, name
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(2)
            torch.testing.assert_close(parameter.grad, original.grad, rtol=3e-4, atol=3e-6, msg=name)
    finally:
        dist.destroy_process_group()


def test_qwen3_vl_vision_stays_local_and_text_is_split_after_visual_merge():
    from transformers.utils import is_torch_available
    assert is_torch_available()
    mp.spawn(_slim_qwen_vl_worker, args=(_find_free_port(),), nprocs=2, join=True)


def test_sp_weak_registry_does_not_retain_the_model():
    import gc
    import weakref
    from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    ref = weakref.ref(model)
    ctx = SequenceParallel()
    ctx._prepare_forward_boundaries(model, model.model)
    ctx.causal_mask_func = model.forward
    del model
    gc.collect()
    assert ref() is None
    assert ctx.causal_mask_func is None


@pytest.mark.parametrize('backend', ['eager', 'flash_attention_2', 'flash_attention_3'])
def test_sp_rejects_unsupported_backend_before_building_groups(backend):
    from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    model.config._attn_implementation = backend
    with pytest.raises(NotImplementedError, match='unsupported attention|kernel unavailable'):
        SequenceParallel().prepare(2, model, SimpleNamespace(pad_token_id=0))




def test_sentence_loss_metadata_stays_global_while_model_tokens_split():
    strategy = SequenceParallelStrategy(sp_config={'ulysses_size': 2})
    ctx = strategy._ctx
    ctx.world_size = ctx.sp_world_size = 2
    ctx.rp_world_size = 1
    ctx.attn_implementation = 'sdpa'
    ctx.tokenizer = SimpleNamespace(pad_token_id=0)
    labels, weights = torch.tensor([[1]]), torch.tensor([[0.3]])
    inputs = strategy.prepare_model_inputs(
        {'input_ids': torch.arange(7).unsqueeze(0), 'labels': labels, 'loss_scale': weights},
        token_labels=False)
    assert inputs['input_ids'].shape == (1, 4)
    assert inputs['labels'] is labels and inputs['loss_scale'] is weights
    assert inputs['_sequence_layout']['sharded'] == []


def _heterogeneous_context_cpu_worker(rank, port):
    import torch.nn.functional as F
    from torch.utils.checkpoint import DefaultDeviceType
    DefaultDeviceType.set_device_type('cpu')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=4)
    try:
        torch.manual_seed(73)
        cases = []
        for sp_size, heads, length in ((4, 4, 7), (2, 2, 9)):
            config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=heads, num_key_value_heads=heads,
                                 attention_dropout=0., use_cache=False)
            config._attn_implementation = 'sdpa'
            base = LlamaForCausalLM(config)
            model = copy.deepcopy(base)
            ids = (torch.arange(length) + 1).unsqueeze(0)
            labels = _make_labels(ids)
            baseline = base(input_ids=ids, use_cache=False).logits
            base_loss = F.cross_entropy(baseline.flatten(0, 1), labels.flatten(), ignore_index=-100)
            mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=1, ulysses_size=sp_size, device_type='cpu')
            strategy = _make_strategy(model, mesh, sp_size)
            prepared = strategy.prepare_model_inputs({'input_ids': ids, 'labels': labels})
            local_labels, layout = prepared.pop('labels'), prepared.pop('_sequence_layout')
            with strategy.context():
                output = model(**prepared, use_cache=False)
            prepared.update(labels=local_labels, _sequence_layout=layout)
            prepared, output = strategy.gather_loss_tensors(prepared, output)
            torch.testing.assert_close(output.logits, baseline, rtol=2e-5, atol=2e-6)
            loss = F.cross_entropy(output.logits.flatten(0, 1), prepared['labels'].flatten(), ignore_index=-100)
            torch.testing.assert_close(loss, base_loss, rtol=2e-5, atol=2e-6)
            cases.append((strategy, model, base, loss, base_loss))
        assert cases[0][0]._ctx is not cases[1][0]._ctx
        for strategy, model, base, loss, base_loss in reversed(cases):
            loss.backward()
            base_loss.backward()
            for parameter, expected in zip(model.parameters(), base.parameters()):
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(4)
                torch.testing.assert_close(parameter.grad, expected.grad, rtol=3e-4, atol=3e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_heterogeneous_sp4_sp2_contexts_logits_loss_and_parameter_gradients_cpu():
    mp.spawn(_heterogeneous_context_cpu_worker, args=(_find_free_port(),), nprocs=4, join=True)


def test_disabled_strategy_context_stays_local_and_failed_registration_is_atomic():
    from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel, _current_ctx, _MODULE_TO_CTX
    strategy = SequenceParallelStrategy(sp_config={'enabled': False, 'ulysses_size': 2})
    with strategy.context():
        assert _current_ctx() is None
    model = torch.nn.Linear(2, 2)
    ctx = SequenceParallel()
    with pytest.raises(NotImplementedError, match='text attention modules'):
        ctx._prepare_forward_boundaries(model, model)
    assert model not in _MODULE_TO_CTX
    with pytest.raises(RuntimeError, match='Initialize sequence parallel'):
        SequenceParallelStrategy(sp_config={'ulysses_size': 2}).context()
