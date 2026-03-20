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
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_flash_attention_utils import is_flash_attn_available

from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelStrategy
from twinkle.utils import DeviceMesh

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


def _make_case(case_name: str) -> dict:
    cases = {
        'cp_only': {
            'world_size': 2,
            'ulysses_size': 2,
            'num_attention_heads': 1,
            'hidden_size': 64,
            'seq_len': 769,
        },
        'cp_sp': {
            'world_size': 4,
            'ulysses_size': 4,
            'num_attention_heads': 6,
            'hidden_size': 96,
            'seq_len': 769,
        },
        'sp_only_memory': {
            'world_size': 4,
            'ulysses_size': 2,
            'num_attention_heads': 8,
            'hidden_size': 128,
            'seq_lens': [255, 511, 1023],
            'batch_sizes': [1, 2, 4],
        },
        'cp_only_memory': {
            'world_size': 2,
            'ulysses_size': 2,
            'num_attention_heads': 1,
            'hidden_size': 64,
            'seq_lens': [511, 1023, 2047],
            'batch_sizes': [1],
        },
        'cp_sp_memory': {
            'world_size': 4,
            'ulysses_size': 4,
            'num_attention_heads': 6,
            'hidden_size': 96,
            'seq_lens': [511, 1023, 2047],
            'batch_sizes': [1],
        },
    }
    return copy.deepcopy(cases[case_name])


def _build_tiny_llama(case: dict, device: torch.device) -> LlamaForCausalLM:
    hidden_size = int(case['hidden_size'])
    num_heads = int(case['num_attention_heads'])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_seq_len = int(case.get('seq_len', max(case.get('seq_lens', [1024])))) + 32
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=1,
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
    config._attn_implementation = 'flash_attention_2'
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
    processed = strategy.preprocess_inputs({
        'input_ids': input_ids,
        'position_ids': position_ids,
        'labels': labels,
    })
    return processed['labels']


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
            raise AssertionError(
                f'{case_name} attention grad mismatch on rank {rank} for {key}: max_diff={max_diff}')


def _init_dist(rank: int, world_size: int, port: int):
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


def _measure_peak_memory(
    model: LlamaForCausalLM,
    strategy: SequenceParallelStrategy,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> int:
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    local_labels = _prepare_label_inputs(strategy, input_ids, position_ids)

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        use_cache=False,
    )
    local_loss = F.cross_entropy(
        outputs.logits.float().view(-1, outputs.logits.size(-1)),
        local_labels.view(-1),
        ignore_index=-100,
        reduction='mean',
    )
    global_loss = strategy.reduce_loss(local_loss, local_labels)
    global_loss.backward()
    torch.cuda.synchronize(device)

    peak = torch.tensor([int(torch.cuda.max_memory_allocated(device))], device=device)
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


def _run_precision_worker(rank: int, world_size: int, port: int, case_name: str):
    device = _init_dist(rank, world_size, port)
    try:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        case = _make_case(case_name)

        base_model = _build_tiny_llama(case, device)
        sp_model = _build_tiny_llama(case, device)
        sp_model.load_state_dict(base_model.state_dict())

        seq_len = int(case['seq_len'])
        input_ids = torch.randint(low=0, high=int(base_model.config.vocab_size), size=(1, seq_len), device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        labels = _make_labels(input_ids)

        base_model.zero_grad(set_to_none=True)
        base_outputs = base_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=False,
        )
        base_logits = base_outputs.logits.detach().float()
        base_loss = F.cross_entropy(
            base_outputs.logits.float().view(-1, base_outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )
        base_loss.backward()
        base_attention_grads = _collect_attention_param_grads(base_model)

        device_mesh = DeviceMesh.from_sizes(
            fsdp_size=world_size,
            dp_size=1,
            ulysses_size=int(case['ulysses_size']),
            device_type='cuda',
        )
        strategy = _make_strategy(sp_model, device_mesh, int(case['ulysses_size']))
        local_labels = _prepare_label_inputs(strategy, input_ids, position_ids)

        sp_model.zero_grad(set_to_none=True)
        sp_outputs = sp_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=False,
        )
        local_logits = sp_outputs.logits
        gathered_outputs = strategy.postprocess_outputs(sp_outputs)
        sp_logits = gathered_outputs.logits.detach().float()

        local_loss = F.cross_entropy(
            local_logits.float().view(-1, local_logits.size(-1)),
            local_labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )
        global_loss = strategy.reduce_loss(local_loss, local_labels)
        global_loss.backward()
        sp_attention_grads = _collect_attention_param_grads(sp_model)

        if not torch.allclose(sp_logits[:, :seq_len], base_logits, rtol=LOGITS_RTOL, atol=LOGITS_ATOL):
            diff = (sp_logits[:, :seq_len] - base_logits).abs().max().item()
            raise AssertionError(f'{case_name} logits mismatch on rank {rank}: max_diff={diff}')
        if abs(global_loss.item() - base_loss.item()) > LOSS_ATOL:
            raise AssertionError(
                f'{case_name} loss mismatch on rank {rank}: sp={global_loss.item()} base={base_loss.item()}')
        _assert_grad_dict_close(case_name, rank, base_attention_grads, sp_attention_grads)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_memory_worker(rank: int, world_size: int, port: int, case_name: str):
    device = _init_dist(rank, world_size, port)
    try:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        case = _make_case(case_name)
        baseline_device_mesh = DeviceMesh.from_sizes(
            fsdp_size=world_size,
            dp_size=1,
            ulysses_size=1,
            device_type='cuda',
        )
        baseline_model = _build_tiny_llama(case, device)
        baseline_strategy = _make_strategy(baseline_model, baseline_device_mesh, 1)

        baseline_peaks = {}
        for batch_size in case['batch_sizes']:
            for seq_len in case['seq_lens']:
                baseline_peak = _measure_peak_memory(
                    baseline_model, baseline_strategy, batch_size=batch_size, seq_len=seq_len, device=device)
                baseline_peaks[(int(batch_size), int(seq_len))] = int(baseline_peak)

        del baseline_model
        del baseline_strategy
        torch.cuda.empty_cache()

        device_mesh = DeviceMesh.from_sizes(
            fsdp_size=world_size,
            dp_size=1,
            ulysses_size=int(case['ulysses_size']),
            device_type='cuda',
        )
        model = _build_tiny_llama(case, device)
        strategy = _make_strategy(model, device_mesh, int(case['ulysses_size']))

        peaks = []
        for batch_size in case['batch_sizes']:
            for seq_len in case['seq_lens']:
                peak = _measure_peak_memory(model, strategy, batch_size=batch_size, seq_len=seq_len, device=device)
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


class TestDerivedRingPrecision(unittest.TestCase):

    def _skip_if_unavailable(self, world_size: int = 4):
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for derived ring precision tests.')
        if torch.cuda.device_count() < world_size:
            self.skipTest(f'Requires at least {world_size} CUDA devices.')
        if not is_flash_attn_available():
            self.skipTest('flash_attention_2 is required for derived ring precision tests.')

    def test_cp_only_precision_alignment(self):
        case = _make_case('cp_only')
        world_size = int(case['world_size'])
        self._skip_if_unavailable(world_size)
        port = _find_free_port()
        mp.spawn(_run_precision_worker, args=(world_size, port, 'cp_only'), nprocs=world_size, join=True)

    def test_cp_sp_precision_alignment(self):
        case = _make_case('cp_sp')
        world_size = int(case['world_size'])
        self._skip_if_unavailable(world_size)
        port = _find_free_port()
        mp.spawn(_run_precision_worker, args=(world_size, port, 'cp_sp'), nprocs=world_size, join=True)


class TestDerivedRingMemoryProfile(unittest.TestCase):

    def _skip_if_unavailable(self, world_size: int = 4):
        if os.environ.get('TWINKLE_RUN_MEMORY_TESTS', '0') != '1':
            self.skipTest('Set TWINKLE_RUN_MEMORY_TESTS=1 to run CUDA memory profile tests.')
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for derived ring memory tests.')
        if torch.cuda.device_count() < world_size:
            self.skipTest(f'Requires at least {world_size} CUDA devices.')
        if not is_flash_attn_available():
            self.skipTest('flash_attention_2 is required for derived ring memory tests.')

    def test_sp_only_memory_profile_grid(self):
        case = _make_case('sp_only_memory')
        world_size = int(case['world_size'])
        self._skip_if_unavailable(world_size)
        port = _find_free_port()
        mp.spawn(_run_memory_worker, args=(world_size, port, 'sp_only_memory'), nprocs=world_size, join=True)

    def test_cp_only_memory_profile_grid(self):
        case = _make_case('cp_only_memory')
        world_size = int(case['world_size'])
        self._skip_if_unavailable(world_size)
        port = _find_free_port()
        mp.spawn(_run_memory_worker, args=(world_size, port, 'cp_only_memory'), nprocs=world_size, join=True)

    def test_cp_sp_memory_profile_grid(self):
        case = _make_case('cp_sp_memory')
        world_size = int(case['world_size'])
        self._skip_if_unavailable(world_size)
        port = _find_free_port()
        mp.spawn(_run_memory_worker, args=(world_size, port, 'cp_sp_memory'), nprocs=world_size, join=True)
