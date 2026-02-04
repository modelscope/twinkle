# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import socket
import sys
from pathlib import Path
# QWEN3_0_6B_MODEL_ID=/path/to/Qwen3-0.6B \
# QWEN3_0_6B_LOCAL_ONLY=1 \
# pytest -q tests/transformers/test_sequence_parallel_qwen3.py

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import unittest
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelStrategy
from twinkle.utils import DeviceMesh


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _load_qwen3_config(model_id: str, local_files_only: bool):
    try:
        return AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except Exception as exc:  # noqa: BLE001
        config_path = Path(model_id) / "config.json"
        if not config_path.exists():
            raise exc
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if "model_type" not in data:
            data["model_type"] = "qwen3"
        if "architectures" not in data:
            data["architectures"] = ["Qwen3ForCausalLM"]
        try:
            return AutoConfig.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            print(f"AutoConfig.from_dict fallback to PretrainedConfig for {model_id}: {exc}")
            return PretrainedConfig.from_dict(data)


def _load_qwen3_pretrained(model_id: str, local_files_only: bool, device: torch.device):
    config = _load_qwen3_config(model_id, local_files_only)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = 1
    if hasattr(config, "use_cache"):
        config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return model


def _select_param_names(model, max_params: int = 6) -> List[str]:
    names: List[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.is_floating_point():
            names.append(name)
        if len(names) >= max_params:
            break
    if not names:
        raise RuntimeError("No trainable parameters found for gradient check.")
    return names


def _capture_grads(model, param_names: List[str]) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name not in param_names:
            continue
        if param.grad is None:
            grads[name] = torch.zeros_like(param, dtype=param.dtype).detach().cpu()
        else:
            grads[name] = param.grad.detach().cpu()
    return grads


def _run_worker_sp_align(rank: int, world_size: int, port: int, model_id: str, local_files_only: bool):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA (4 GPUs).")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        device_id=device,
    )
    dist.barrier()

    try:
        torch.manual_seed(1234)
        model = _load_qwen3_pretrained(model_id, local_files_only, device)
        vocab_size = int(model.config.vocab_size)
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            device=device,
        )
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        labels = input_ids.clone()

        def _compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            # Align with standard causal LM shift: predict token t+1 from token t.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            return F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Baseline forward/backward (no SP)
        baseline_out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        baseline_logits = baseline_out.logits.detach()
        baseline_loss = _compute_lm_loss(baseline_out.logits, labels)
        baseline_loss.backward()
        param_names = _select_param_names(model)
        baseline_grads = _capture_grads(model, param_names)
        model.zero_grad(set_to_none=True)

        # Enable sequence parallel and run again
        device_mesh = DeviceMesh.from_sizes(dp_size=world_size, ulysses_size=2, device_type="cuda")
        sp_strategy = SequenceParallelStrategy(
            device_mesh=device_mesh,
            sp_config={"enabled": True, "ulysses_size": 2, "gather_logits": True},
            model=model,
            tokenizer_id=model_id,
        )
        sp_strategy.initialize()
        sp_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        sp_inputs = sp_strategy.preprocess_inputs(sp_inputs)
        sp_out = model(**sp_inputs, use_cache=False)
        sp_out = sp_strategy.postprocess_outputs(sp_out)
        sp_logits = sp_out.logits.detach()
        sp_loss = _compute_lm_loss(sp_out.logits, labels)
        sp_loss.backward()
        sp_grads = _capture_grads(model, param_names)

        # Forward alignment
        diff = (sp_logits - baseline_logits).abs()
        if not torch.allclose(sp_logits, baseline_logits, rtol=1e-3, atol=1e-4):
            print(
                f"[rank{rank}] logits diff mean={diff.mean().item():.6e} max={diff.max().item():.6e}"
            )
        assert torch.allclose(sp_logits, baseline_logits, rtol=1e-3, atol=1e-4)

        # Backward alignment (selected params)
        for name in param_names:
            base = baseline_grads[name].to(device=device, dtype=torch.float32)
            sp = sp_grads[name].to(device=device, dtype=torch.float32)
            diff = sp - base
            rel = diff.norm() / (base.norm() + 1e-12)
            if rel.item() > 1e-3:
                abs_diff = diff.abs()
                base_norm = base.norm().item()
                sp_norm = sp.norm().item()
                ratio = sp_norm / base_norm if base_norm != 0 else float("inf")
                print(
                    f"[rank{rank}] {name} grad diff mean={abs_diff.mean().item():.6e} "
                    f"max={abs_diff.max().item():.6e} sp_norm={sp_norm:.6e} "
                    f"base_norm={base_norm:.6e} ratio={ratio:.6e} rel_norm={rel.item():.6e}"
                )
            assert rel.item() <= 1e-3
    finally:
        dist.destroy_process_group()


class TestSequenceParallelQwen3(unittest.TestCase):
    def test_qwen3_sp_alignment(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this test.")
        world_size = 2
        if torch.cuda.device_count() < world_size:
            self.skipTest("Requires at least 4 GPUs for SP alignment test.")
        model_id = os.environ.get("QWEN3_0_6B_MODEL_ID", "Qwen/Qwen3-0.6B")
        local_files_only = os.environ.get("QWEN3_0_6B_LOCAL_ONLY", "1") != "0"
        try:
            _load_qwen3_config(model_id, local_files_only)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Qwen3 model not available locally: {exc}")
        port = _find_free_port()
        mp.spawn(
            _run_worker_sp_align,
            args=(world_size, port, model_id, local_files_only),
            nprocs=world_size,
            join=True,
        )
