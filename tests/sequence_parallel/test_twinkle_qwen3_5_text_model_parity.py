# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import socket
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
import unittest
from datetime import timedelta
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.utils.import_utils import is_flash_attn_2_available
from types import SimpleNamespace

from twinkle.model.transformers.models.qwen3_5 import modeling_qwen3_5 as tw_qwen35
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel, SequenceParallelContext
from twinkle.utils import DeviceMesh


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _build_linear_parity_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act='silu',
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=['linear_attention'],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _build_mixed_parity_config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act='silu',
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=['full_attention', 'linear_attention'],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    attn_implementation = 'flash_attention_2' if is_flash_attn_2_available() else 'sdpa'
    config._attn_implementation = attn_implementation
    config._attn_implementation_internal = attn_implementation
    return config


def _linear_attention_runtime_available() -> bool:
    return bool(torch.cuda.is_available() and tw_qwen35._FLA_CAUSAL_CONV1D_FN is not None
                and tw_qwen35._FLA_CHUNK_GATED_DELTA_RULE is not None
                and tw_qwen35._FLA_FUSED_RECURRENT_GATED_DELTA_RULE is not None and tw_qwen35._HAS_CAUSAL_CONV1D)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _all_gather_seq(local_tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    chunks = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(chunks, local_tensor.contiguous(), group=group)
    return torch.cat(chunks, dim=1).contiguous()


def _all_reduce_grads(module: torch.nn.Module, group: dist.ProcessGroup) -> None:
    for param in module.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=group)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_fp32 = actual.detach().to(dtype=torch.float32)
    expected_fp32 = expected.detach().to(dtype=torch.float32)
    return float((actual_fp32 - expected_fp32).norm() / (expected_fp32.norm() + 1e-12))


def _assert_relative_error(actual: torch.Tensor, expected: torch.Tensor, rel_tol: float, name: str) -> None:
    rel = _relative_error(actual, expected)
    if rel > rel_tol:
        raise AssertionError(f'{name} relative error {rel:.4e} exceeds tolerance {rel_tol:.4e}')


def _write_error(error_prefix: str, rank: int) -> None:
    with open(f'{error_prefix}.rank{rank}.err', 'w', encoding='utf-8') as f:
        f.write(traceback.format_exc())


def _run_linear_attention_parity_worker(rank: int, world_size: int, port: int, error_prefix: str) -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(rank)

    try:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )

        device = torch.device(f'cuda:{rank}')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        seed = 1234
        batch_size = 2
        seq_len = 8
        local_seq_len = seq_len // world_size
        start = rank * local_seq_len
        end = start + local_seq_len

        _seed_everything(seed)
        config = _build_linear_parity_config()
        baseline_module = tw_qwen35.TwinkleQwen3_5GatedDeltaNet(
            config, layer_idx=0).to(
                device=device, dtype=dtype).eval()
        sp_module = copy.deepcopy(baseline_module).to(device=device, dtype=dtype).eval()

        full_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
        dist.broadcast(full_hidden_states, src=0)
        full_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64, device=device)
        cu_seq_lens_q = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)

        baseline_hidden_states = full_hidden_states.detach().clone().requires_grad_(True)
        baseline_output = baseline_module(
            hidden_states=baseline_hidden_states,
            attention_mask=attention_mask,
            cu_seq_lens_q=cu_seq_lens_q,
        )
        baseline_loss = baseline_output.float().square().sum()
        baseline_loss.backward()
        baseline_input_grad = baseline_hidden_states.grad.detach()
        baseline_param_grads = {
            name: param.grad.detach().clone()
            for name, param in baseline_module.named_parameters() if param.grad is not None
        }

        sp_hidden_states = full_hidden_states[:, start:end].detach().clone().requires_grad_(True)
        sp_output = sp_module(
            hidden_states=sp_hidden_states,
            attention_mask=attention_mask[:, start:end].contiguous(),
            cu_seq_lens_q=cu_seq_lens_q,
            sequence_parallel_context=SequenceParallelContext(
                sp_group=dist.group.WORLD,
                sp_world_size=world_size,
                rank=rank,
                world_size=world_size,
                real_position_ids=full_position_ids,
                is_packed=False,
            ),
        )
        sp_loss = sp_output.float().square().sum()
        sp_loss.backward()
        _all_reduce_grads(sp_module, dist.group.WORLD)

        sp_output_full = _all_gather_seq(sp_output.detach(), dist.group.WORLD)
        sp_input_grad_full = _all_gather_seq(sp_hidden_states.grad.detach(), dist.group.WORLD)

        torch.testing.assert_close(
            sp_output_full.to(dtype=torch.float32),
            baseline_output.detach().to(dtype=torch.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        _assert_relative_error(sp_input_grad_full, baseline_input_grad, 1e-2, 'linear_attention.input_grad')

        for name in (
                'in_proj_qkv.weight',
                'in_proj_z.weight',
                'out_proj.weight',
        ):
            _assert_relative_error(
                sp_module.get_parameter(name).grad,
                baseline_param_grads[name],
                2e-2,
                f'linear_attention.{name}',
            )
    except Exception:
        _write_error(error_prefix, rank)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_mixed_text_model_parity_worker(rank: int, world_size: int, port: int, error_prefix: str) -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(rank)

    try:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )

        device = torch.device(f'cuda:{rank}')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        seed = 5678
        batch_size = 2
        seq_len = 8

        _seed_everything(seed)
        config = _build_mixed_parity_config()
        baseline_model = tw_qwen35.TwinkleQwen3_5TextModel(config).to(device=device, dtype=dtype).eval()
        sp_model = copy.deepcopy(baseline_model).to(device=device, dtype=dtype).eval()

        full_inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
        dist.broadcast(full_inputs_embeds, src=0)
        full_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        baseline_inputs_embeds = full_inputs_embeds.detach().clone().requires_grad_(True)
        baseline_outputs = baseline_model(
            inputs_embeds=baseline_inputs_embeds,
            position_ids=full_position_ids,
            use_cache=False,
        )
        baseline_hidden = baseline_outputs.last_hidden_state
        baseline_loss = baseline_hidden.float().square().sum()
        baseline_loss.backward()
        baseline_input_grad = baseline_inputs_embeds.grad.detach()
        baseline_param_grads = {
            name: param.grad.detach().clone()
            for name, param in baseline_model.named_parameters() if param.grad is not None
        }

        device_mesh = DeviceMesh.from_sizes(
            world_size=world_size,
            dp_size=world_size,
            ulysses_size=world_size,
            device_type='cuda',
        )
        sp = SequenceParallel()
        sp.prepare(world_size, sp_model, SimpleNamespace(pad_token_id=0), device_mesh=device_mesh)

        sp_inputs_embeds = full_inputs_embeds.detach().clone().requires_grad_(True)
        sp_inputs = sp.prepare_inputs({
            'inputs_embeds': sp_inputs_embeds,
            'position_ids': full_position_ids.clone(),
            'use_cache': False,
        })
        sp_outputs = sp_model(**sp_inputs)
        sp_hidden_local = sp_outputs.last_hidden_state
        sp_loss = sp_hidden_local.float().square().sum()
        sp_loss.backward()
        dist.all_reduce(sp_inputs_embeds.grad, group=dist.group.WORLD)
        _all_reduce_grads(sp_model, dist.group.WORLD)

        sp_hidden_full = _all_gather_seq(sp_hidden_local.detach(), dist.group.WORLD)
        torch.testing.assert_close(
            sp_hidden_full.to(dtype=torch.float32),
            baseline_hidden.detach().to(dtype=torch.float32),
            rtol=5e-3,
            atol=5e-3,
        )
        _assert_relative_error(sp_inputs_embeds.grad, baseline_input_grad, 1e-2, 'mixed_text_model.input_grad')

        for name in (
                'layers.0.self_attn.q_proj.weight',
                'layers.1.linear_attn.in_proj_qkv.weight',
                'layers.1.mlp.gate_proj.weight',
        ):
            _assert_relative_error(
                sp_model.get_parameter(name).grad,
                baseline_param_grads[name],
                2e-2,
                f'mixed_text_model.{name}',
            )
    except Exception:
        _write_error(error_prefix, rank)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestTwinkleQwen35TextModelParity(unittest.TestCase):

    WORLD_SIZE = 2

    def _run_spawned_parity_test(self, worker) -> None:
        port = _find_free_port()
        with tempfile.TemporaryDirectory() as temp_dir:
            error_prefix = os.path.join(temp_dir, 'parity')
            try:
                mp.spawn(
                    worker,
                    args=(self.WORLD_SIZE, port, error_prefix),
                    nprocs=self.WORLD_SIZE,
                    join=True,
                )
            except Exception:
                error_logs = []
                for rank in range(self.WORLD_SIZE):
                    error_path = f'{error_prefix}.rank{rank}.err'
                    if os.path.exists(error_path):
                        with open(error_path, encoding='utf-8') as f:
                            error_logs.append(f'Rank {rank}:\n{f.read()}')
                if error_logs:
                    self.fail('\n\n'.join(error_logs))
                raise

    def test_linear_attention_sp_parity(self):
        if not _linear_attention_runtime_available():
            self.skipTest('CUDA + flash-linear-attention + causal-conv1d are required for linear attention parity.')
        if torch.cuda.device_count() < self.WORLD_SIZE:
            self.skipTest(f'Need at least {self.WORLD_SIZE} CUDA devices for SP parity.')
        self._run_spawned_parity_test(_run_linear_attention_parity_worker)

    def test_mixed_text_model_sp_parity(self):
        if not _linear_attention_runtime_available():
            self.skipTest('CUDA + flash-linear-attention + causal-conv1d are required for mixed model parity.')
        if torch.cuda.device_count() < self.WORLD_SIZE:
            self.skipTest(f'Need at least {self.WORLD_SIZE} CUDA devices for SP parity.')
        self._run_spawned_parity_test(_run_mixed_text_model_parity_worker)


if __name__ == '__main__':
    unittest.main()
