import json
import os
import socket
import tempfile
import traceback
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.utils.import_utils import is_flash_attn_2_available

from twinkle.model.transformers.models.qwen3_5 import modeling_qwen3_5 as tw_qwen35
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel, SequenceParallelContext
from twinkle.utils import DeviceMesh

# Examples:
# CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python cookbook/transformers/qwen3_5_sp_memory_bench.py
# CUDA_VISIBLE_DEVICES=0,1 QWEN35_SP_MEMORY_MODE=linear PYTHONPATH=src \
#   python cookbook/transformers/qwen3_5_sp_memory_bench.py


def _build_linear_bench_config() -> Qwen3_5TextConfig:
    hidden_size = int(os.environ.get('QWEN35_SP_MEMORY_HIDDEN_SIZE', '1024'))
    head_dim = int(os.environ.get('QWEN35_SP_MEMORY_HEAD_DIM', '64'))
    num_attention_heads = hidden_size // head_dim
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=max(1, num_attention_heads // 2),
        head_dim=head_dim,
        hidden_act='silu',
        max_position_embeddings=16384,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=max(2, num_attention_heads // 2),
        linear_num_value_heads=num_attention_heads,
        layer_types=['linear_attention'],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _build_mixed_bench_config() -> Qwen3_5TextConfig:
    hidden_size = int(os.environ.get('QWEN35_SP_MEMORY_HIDDEN_SIZE', '1024'))
    head_dim = int(os.environ.get('QWEN35_SP_MEMORY_HEAD_DIM', '64'))
    num_attention_heads = hidden_size // head_dim
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=max(1, num_attention_heads // 2),
        head_dim=head_dim,
        hidden_act='silu',
        max_position_embeddings=16384,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=max(2, num_attention_heads // 2),
        linear_num_value_heads=num_attention_heads,
        layer_types=['full_attention', 'linear_attention'],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    attn_implementation = os.environ.get(
        'QWEN35_SP_MEMORY_ATTN_IMPLEMENTATION',
        'flash_attention_2' if is_flash_attn_2_available() else 'sdpa',
    )
    config._attn_implementation = attn_implementation
    config._attn_implementation_internal = attn_implementation
    return config


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _parse_cases():
    spec = os.environ.get('QWEN35_SP_MEMORY_CASES', '1x1024,1x2048,2x2048')
    cases = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        batch_size, seq_len = item.lower().split('x', 1)
        cases.append((int(batch_size), int(seq_len)))
    return cases


def _measure_cuda_peak_stats(run_step):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_step()
    torch.cuda.synchronize()
    return {
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def _run_linear_attention_memory_step(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    sequence_parallel_context: SequenceParallelContext | None = None,
) -> dict[str, float]:

    def _step():
        module.zero_grad(set_to_none=True)
        local_hidden_states = hidden_states.detach().clone().requires_grad_(True)
        output = module(
            hidden_states=local_hidden_states,
            attention_mask=attention_mask,
            cu_seq_lens_q=cu_seq_lens_q,
            sequence_parallel_context=sequence_parallel_context,
        )
        loss = output.float().square().mean()
        loss.backward()

    return _measure_cuda_peak_stats(_step)


def _run_text_model_memory_step(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor | bool],
) -> dict[str, float]:

    def _step():
        model.zero_grad(set_to_none=True)
        local_inputs = {}
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                local_inputs[key] = value.detach().clone()
            else:
                local_inputs[key] = value
        outputs = model(**local_inputs)
        loss = outputs.last_hidden_state.float().square().mean()
        loss.backward()

    return _measure_cuda_peak_stats(_step)


def _write_error(error_prefix: str, rank: int) -> None:
    with open(f'{error_prefix}.rank{rank}.err', 'w', encoding='utf-8') as f:
        f.write(traceback.format_exc())


def _run_linear_attention_memory_worker(rank: int, world_size: int, port: int, result_path: str, cases):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(rank)
    error_prefix = result_path
    try:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )

        device = torch.device(f'cuda:{rank}')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        config = _build_linear_bench_config()
        results = []

        for batch_size, seq_len in cases:
            if seq_len % world_size != 0:
                raise ValueError(f'seq_len ({seq_len}) must be divisible by world_size ({world_size})')

            full_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64, device=device)
            full_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            cu_seq_lens_q = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=device,
            )

            baseline_module = tw_qwen35.TwinkleQwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device, dtype=dtype)
            baseline_module.train()
            baseline_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
            baseline_stats = _run_linear_attention_memory_step(
                baseline_module,
                baseline_hidden_states,
                attention_mask=full_attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                sequence_parallel_context=None,
            )
            del baseline_module, baseline_hidden_states
            torch.cuda.empty_cache()

            local_seq_len = seq_len // world_size
            start = rank * local_seq_len
            end = start + local_seq_len
            sp_attention_mask = full_attention_mask[:, start:end].contiguous()
            sp_hidden_states = torch.randn(batch_size, local_seq_len, config.hidden_size, device=device, dtype=dtype)
            sp_context = SequenceParallelContext(
                sp_group=dist.group.WORLD,
                sp_world_size=world_size,
                rank=rank,
                world_size=world_size,
                real_position_ids=full_position_ids,
                is_packed=False,
            )
            sp_module = tw_qwen35.TwinkleQwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device, dtype=dtype)
            sp_module.train()
            sp_stats = _run_linear_attention_memory_step(
                sp_module,
                sp_hidden_states,
                attention_mask=sp_attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                sequence_parallel_context=sp_context,
            )
            del sp_module, sp_hidden_states
            torch.cuda.empty_cache()

            payload = torch.tensor([
                baseline_stats['peak_allocated_mib'],
                baseline_stats['peak_reserved_mib'],
                sp_stats['peak_allocated_mib'],
                sp_stats['peak_reserved_mib'],
            ], device=device)
            gathered = [torch.zeros_like(payload) for _ in range(world_size)]
            dist.all_gather(gathered, payload)

            if rank == 0:
                gathered_cpu = [tensor.cpu().tolist() for tensor in gathered]
                results.append({
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'baseline_peak_allocated_mib_per_rank': [row[0] for row in gathered_cpu],
                    'baseline_peak_reserved_mib_per_rank': [row[1] for row in gathered_cpu],
                    'sp_peak_allocated_mib_per_rank': [row[2] for row in gathered_cpu],
                    'sp_peak_reserved_mib_per_rank': [row[3] for row in gathered_cpu],
                    'baseline_peak_allocated_mib_max': max(row[0] for row in gathered_cpu),
                    'sp_peak_allocated_mib_max': max(row[2] for row in gathered_cpu),
                })

        if rank == 0:
            torch.save(results, result_path)
    except Exception:
        _write_error(error_prefix, rank)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_mixed_text_model_memory_worker(rank: int, world_size: int, port: int, result_path: str, cases):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(rank)
    error_prefix = result_path
    try:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )

        device = torch.device(f'cuda:{rank}')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        baseline_results = []
        sp_results = []

        for batch_size, seq_len in cases:
            config = _build_mixed_bench_config()
            full_input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
            full_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

            baseline_model = tw_qwen35.TwinkleQwen3_5TextModel(config).to(device=device, dtype=dtype)
            baseline_model.train()
            baseline_stats = _run_text_model_memory_step(
                baseline_model,
                {
                    'input_ids': full_input_ids,
                    'position_ids': full_position_ids,
                    'use_cache': False,
                },
            )
            baseline_results.append(baseline_stats)
            del baseline_model
            torch.cuda.empty_cache()

        device_mesh = DeviceMesh.from_sizes(
            world_size=world_size,
            dp_size=world_size,
            ulysses_size=world_size,
            device_type='cuda',
        )
        tokenizer = SimpleNamespace(pad_token_id=0)
        sp = SequenceParallel()

        for batch_size, seq_len in cases:
            if seq_len % world_size != 0:
                raise ValueError(f'seq_len ({seq_len}) must be divisible by world_size ({world_size})')

            config = _build_mixed_bench_config()
            full_input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
            full_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

            sp_model = tw_qwen35.TwinkleQwen3_5TextModel(config).to(device=device, dtype=dtype)
            sp_model.train()
            sp.prepare(world_size, sp_model, tokenizer, device_mesh=device_mesh)
            sp_inputs = sp.prepare_inputs({
                'input_ids': full_input_ids,
                'position_ids': full_position_ids,
                'use_cache': False,
            })
            sp_stats = _run_text_model_memory_step(sp_model, sp_inputs)
            sp_results.append(sp_stats)
            del sp_model
            torch.cuda.empty_cache()

        gathered_results = []
        for (batch_size, seq_len), baseline_stats, sp_stats in zip(cases, baseline_results, sp_results, strict=False):
            payload = torch.tensor([
                baseline_stats['peak_allocated_mib'],
                baseline_stats['peak_reserved_mib'],
                sp_stats['peak_allocated_mib'],
                sp_stats['peak_reserved_mib'],
            ], device=device)
            gathered = [torch.zeros_like(payload) for _ in range(world_size)]
            dist.all_gather(gathered, payload)

            if rank == 0:
                gathered_cpu = [tensor.cpu().tolist() for tensor in gathered]
                gathered_results.append({
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'attn_implementation': getattr(config, '_attn_implementation', None),
                    'baseline_peak_allocated_mib_per_rank': [row[0] for row in gathered_cpu],
                    'baseline_peak_reserved_mib_per_rank': [row[1] for row in gathered_cpu],
                    'sp_peak_allocated_mib_per_rank': [row[2] for row in gathered_cpu],
                    'sp_peak_reserved_mib_per_rank': [row[3] for row in gathered_cpu],
                    'baseline_peak_allocated_mib_max': max(row[0] for row in gathered_cpu),
                    'sp_peak_allocated_mib_max': max(row[2] for row in gathered_cpu),
                })

        if rank == 0:
            torch.save(gathered_results, result_path)
    except Exception:
        _write_error(error_prefix, rank)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(worker, world_size: int, cases):
    port = _find_free_port()
    with tempfile.TemporaryDirectory() as temp_dir:
        result_path = os.path.join(temp_dir, 'memory_results.pt')
        try:
            mp.spawn(
                worker,
                args=(world_size, port, result_path, cases),
                nprocs=world_size,
                join=True,
            )
        except Exception:
            error_logs = []
            for rank in range(world_size):
                error_path = f'{result_path}.rank{rank}.err'
                if os.path.exists(error_path):
                    with open(error_path, 'r', encoding='utf-8') as f:
                        error_logs.append(f'Rank {rank}:\n{f.read()}')
            if error_logs:
                raise RuntimeError('\n\n'.join(error_logs))
            raise
        return torch.load(result_path, weights_only=False)


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the Qwen3.5 SP memory benchmark.')

    world_size = int(os.environ.get('QWEN35_SP_MEMORY_WORLD_SIZE', '2'))
    if torch.cuda.device_count() < world_size:
        raise SystemExit(f'Need at least {world_size} CUDA devices for the Qwen3.5 SP memory benchmark.')

    cases = _parse_cases()
    mode = os.environ.get('QWEN35_SP_MEMORY_MODE', 'both')
    results = {}

    if mode in ('linear', 'both'):
        results['linear_attention'] = _run_spawned(_run_linear_attention_memory_worker, world_size, cases)

    if mode in ('mixed', 'both'):
        results['mixed_text_model'] = _run_spawned(_run_mixed_text_model_memory_worker, world_size, cases)

    output = json.dumps(results, indent=2)
    print(output)

    result_path = os.environ.get('QWEN35_SP_MEMORY_RESULT_PATH')
    if result_path:
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(output)


if __name__ == '__main__':
    main()
