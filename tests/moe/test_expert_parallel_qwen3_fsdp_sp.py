# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import socket
import sys
import copy
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from twinkle.model.transformers.moe import apply_expert_parallel
from twinkle.model.transformers.strategy import NativeFSDPStrategy
from twinkle.model.transformers.strategy.sequence_parallel import (
    SequenceParallelStrategy,
    _get_sp_group_from_device_mesh,
    sequence_parallel,
)
from twinkle.utils import DeviceMesh
# QWEN3_MOE_MODEL_ID=/path/to/Qwen3-MoE \
# QWEN3_MOE_LOCAL_ONLY=1 \
# pytest -q tests/moe/test_expert_parallel_qwen3_fsdp_sp.py -rs


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _enable_strict_determinism(seed: int) -> None:
    """Best-effort deterministic knobs (still not guaranteed bitwise with NCCL collectives)."""
    # These should be set before CUDA context is initialized for best effect.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("NCCL_DETERMINISTIC", "1")
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # Disable reduced-precision bf16 reductions when possible.
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _find_moe_blocks(model: nn.Module) -> List[nn.Module]:
    blocks = []
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        if not isinstance(experts, nn.ModuleList):
            if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
                continue
        gate = getattr(module, "gate", None) or getattr(module, "router", None)
        if gate is None:
            continue
        blocks.append(module)
    return blocks


def _get_top_k(block: nn.Module) -> int:
    if hasattr(block, "num_experts_per_tok") and getattr(block, "num_experts_per_tok") is not None:
        return int(getattr(block, "num_experts_per_tok"))
    if hasattr(block, "top_k") and getattr(block, "top_k") is not None:
        return int(getattr(block, "top_k"))
    gate = getattr(block, "gate", None) or getattr(block, "router", None)
    if gate is not None and hasattr(gate, "top_k") and getattr(gate, "top_k") is not None:
        return int(getattr(gate, "top_k"))
    raise RuntimeError("Cannot infer top_k for MoE block.")


def _capture_router_state(model: nn.Module):
    # Return a list aligned with _find_moe_blocks order.
    states: List[Dict[str, torch.Tensor]] = []
    handles = []
    for block in _find_moe_blocks(model):
        gate = getattr(block, "gate", None) or getattr(block, "router", None)
        if gate is None:
            continue
        top_k = _get_top_k(block)
        norm_topk_prob = getattr(block, "norm_topk_prob", False)

        def _hook(module, inputs, output, *, _top_k=top_k, _norm=norm_topk_prob):
            if isinstance(output, tuple):
                router_logits, routing_weights, selected_experts = output[:3]
            else:
                router_logits = output
                routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
                routing_weights, selected_experts = torch.topk(routing_weights, _top_k, dim=-1)
                if _norm:
                    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            states.append(
                {
                    "selected_experts": selected_experts.detach().cpu(),
                    "routing_weights": routing_weights.detach().cpu(),
                }
            )

        handles.append(gate.register_forward_hook(_hook))
    return states, handles


def _load_qwen3_moe_config(model_id: str, local_files_only: bool):
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
            data["model_type"] = "qwen3_moe"
        if "architectures" not in data:
            data["architectures"] = ["Qwen3MoeForCausalLM"]
        try:
            return AutoConfig.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            print(f"AutoConfig.from_dict fallback to PretrainedConfig for {model_id}: {exc}")
            return PretrainedConfig.from_dict(data)


def _load_qwen3_moe_pretrained(model_id: str, local_files_only: bool, device: torch.device) -> nn.Module:
    config = _load_qwen3_moe_config(model_id, local_files_only)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = 1
    if hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(config, "_experts_implementation"):
        config._experts_implementation = "eager"
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


def _ensure_embed_tokens(model, embed) -> None:
    # SequenceParallel's forward hook looks for `_self.language_model.embed_tokens` or `_self.embed_tokens`
    # where `_self` is the top-level model passed to `sequence_parallel.prepare(...)`.
    #
    # HF models vary: some expose `.language_model`, others expose `.model` (decoder), etc.
    targets = [model]
    for attr in ("language_model", "model"):
        if hasattr(model, attr):
            targets.append(getattr(model, attr))
    for t in targets:
        if t is not None and getattr(t, "embed_tokens", None) is None:
            t.embed_tokens = embed


def _per_token_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # [B,S,V] + [B,S] -> [B,S] (sum/avg applied by caller)
    loss_1d = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    return loss_1d.view(labels.shape)


def _sp_slice_range_for_seq_len(
    seq_len: int,
    *,
    sp_group: Optional[dist.ProcessGroup],
    sp_size: int,
) -> Tuple[int, int]:
    if sp_group is None or sp_size <= 1:
        return 0, seq_len
    sp_rank = dist.get_rank(sp_group)
    if seq_len % sp_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by sp_size ({sp_size}) in this test.")
    local = seq_len // sp_size
    start = sp_rank * local
    end = start + local
    return start, end


def _gather_full_seq_grad_from_sp(local_grad: torch.Tensor, *, sp_group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Gather per-rank local sequence gradients into a full-sequence gradient on every rank."""
    if sp_group is None or dist.get_world_size(sp_group) <= 1:
        return local_grad.contiguous()
    world = dist.get_world_size(sp_group)
    chunks = [torch.empty_like(local_grad) for _ in range(world)]
    dist.all_gather(chunks, local_grad.contiguous(), group=sp_group)
    return torch.cat(chunks, dim=1).contiguous()


def _collect_active_local_expert_grad_tensors(
    block: nn.Module,
    active_global_experts: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return a {f\"expert{global}.{param_name}\": grad_tensor_cpu} dict for active local experts only."""
    active = set(int(x) for x in active_global_experts.reshape(-1).tolist())
    grads: Dict[str, torch.Tensor] = {}
    if isinstance(block.experts, nn.ModuleList):
        for local_idx, expert in enumerate(block.experts):
            global_idx = int(block._ep_local_start + local_idx)
            if global_idx not in active:
                continue
            for name, param in expert.named_parameters():
                if param.grad is None:
                    continue
                grads[f"expert{global_idx}.{name}"] = param.grad.detach().cpu()
        return grads

    # Tensor experts: gradients are indexed by local expert id.
    gate_up = block.experts.gate_up_proj
    down = block.experts.down_proj
    gate_up_grad = gate_up.grad
    down_grad = down.grad
    for local_idx in range(gate_up.shape[0]):
        global_idx = int(block._ep_local_start + local_idx)
        if global_idx not in active:
            continue
        if gate_up_grad is not None:
            grads[f"expert{global_idx}.gate_up_proj"] = gate_up_grad[local_idx].detach().cpu()
        if down_grad is not None:
            grads[f"expert{global_idx}.down_proj"] = down_grad[local_idx].detach().cpu()
    return grads


def _compare_grad_dicts(
    *,
    rank: int,
    baseline: Dict[str, torch.Tensor],
    sp: Dict[str, torch.Tensor],
    rel_tol: float,
) -> None:
    keys = sorted(set(baseline.keys()) | set(sp.keys()))
    for k in keys:
        a = baseline.get(k)
        b = sp.get(k)
        if a is None or b is None:
            raise AssertionError(f"[rank{rank}] Missing grad key={k} baseline={a is not None} sp={b is not None}")
        a32 = a.to(dtype=torch.float32)
        b32 = b.to(dtype=torch.float32)
        diff = b32 - a32
        rel = diff.norm() / (a32.norm() + 1e-12)
        if rel.item() > rel_tol:
            abs_diff = diff.abs()
            print(
                f"[rank{rank}] {k} grad diff mean={abs_diff.mean().item():.6e} "
                f"max={abs_diff.max().item():.6e} rel_norm={rel.item():.6e} tol={rel_tol:.1e}",
                flush=True,
            )
        assert rel.item() <= rel_tol


def _run_worker_ep_fsdp_sp_align(
    rank: int,
    world_size: int,
    port: int,
    model_id: str,
    local_files_only: bool,
):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Some utilities (e.g. Platform.get_local_device()) rely on LOCAL_RANK.
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    strict = os.environ.get("TWINKLE_STRICT_ALIGN", "0") == "1"
    if strict:
        _enable_strict_determinism(seed=1234)

    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA (4 GPUs).")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        device_id=device,
        timeout=timedelta(minutes=15),
    )
    dist.barrier()

    try:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        # 4 GPUs: (fsdp=2, ep=2); SP is derived with ulysses_size=2 over raw data ranks (fsdp).
        device_mesh = DeviceMesh(
            device_type="cuda",
            mesh=np.arange(world_size).reshape(2, 2),
            mesh_dim_names=("fsdp", "ep"),
            ulysses_size=2,
        )
        sp_size = 2
        sp_group = _get_sp_group_from_device_mesh(device_mesh, sp_size)

        # Shared input (same across ranks) + per-rank slice loss (matches SP slice ownership).
        # Keep seq_len divisible by sp_size to avoid padding complexity here.
        batch_size = 2
        seq_len = 8

        # --- Baseline: EP+FSDP (no SP) ---
        model_base = _load_qwen3_moe_pretrained(model_id, local_files_only, device)
        vocab_size = int(model_base.config.vocab_size)
        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
        dist.broadcast(input_ids, src=0)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        # Prepare labels for causal LM: set first token ignore so roll won't create wrap-around target.
        labels_raw = input_ids.clone()
        labels_raw[:, 0] = -100
        labels_shifted = torch.roll(labels_raw, shifts=-1, dims=1)

        embed_base = model_base.get_input_embeddings()
        _ensure_embed_tokens(model_base, embed_base)
        base_embeds = embed_base(input_ids).detach()

        apply_expert_parallel(
            getattr(model_base, "model", model_base),
            device_mesh,
            config={
                "enabled": True,
                "router_dtype": "fp32",
                "all_to_all": "torch",
                "keep_router_logits": False,
            },
        )
        fsdp_strategy = NativeFSDPStrategy(device_mesh=device_mesh, mixed_precision="bf16", fsdp_config={})
        model_base, _ = fsdp_strategy.wrap_model(model_base, optimizer=None)

        base_states, base_state_handles = _capture_router_state(getattr(model_base, "model", model_base))
        base_out = model_base(
            inputs_embeds=base_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        for h in base_state_handles:
            h.remove()
        base_logits = base_out.logits.detach()

        start, end = _sp_slice_range_for_seq_len(seq_len, sp_group=sp_group, sp_size=sp_size)
        base_token_loss = _per_token_ce_loss(base_out.logits, labels_shifted)
        base_loss_sum = base_token_loss[:, start:end].sum()
        base_loss_sum.backward()

        # Collect active experts (slice-only) and corresponding local expert grads.
        base_blocks = _find_moe_blocks(getattr(model_base, "model", model_base))
        if not base_blocks:
            raise RuntimeError("No MoE blocks found in Qwen3 MoE model.")
        assert len(base_states) == len(base_blocks)
        base_active_grads: Dict[str, torch.Tensor] = {}
        for block, state in zip(base_blocks, base_states):
            sel = state["selected_experts"]  # [tokens, top_k] (flattened)
            # Router hook captures all tokens; reshape to [B,S,top_k] and slice same seq range.
            top_k = sel.shape[-1]
            sel = sel.view(batch_size, seq_len, top_k)[:, start:end, :].reshape(-1, top_k)
            active = torch.unique(sel)
            base_active_grads.update(_collect_active_local_expert_grad_tensors(block, active))

        # --- SP variant: EP+FSDP+SP ---
        # Note: SP does global patching; keep it after baseline in this process.
        model_sp = _load_qwen3_moe_pretrained(model_id, local_files_only, device)
        embed_sp = model_sp.get_input_embeddings()
        _ensure_embed_tokens(model_sp, embed_sp)
        sp_embeds = embed_sp(input_ids).detach()

        apply_expert_parallel(
            getattr(model_sp, "model", model_sp),
            device_mesh,
            config={
                "enabled": True,
                "router_dtype": "fp32",
                "all_to_all": "torch",
                "keep_router_logits": False,
            },
        )
        sp_strategy = SequenceParallelStrategy(
            device_mesh=device_mesh,
            sp_config={"enabled": True, "ulysses_size": sp_size, "gather_logits": True},
            model=model_sp,
            tokenizer_id=model_id,
        )
        sp_strategy.initialize()
        model_sp, _ = fsdp_strategy.wrap_model(model_sp, optimizer=None)

        # Preprocess labels through SP strategy so they are shifted + split consistently.
        sp_label_inputs = {"labels": labels_raw, "position_ids": position_ids}
        sp_label_inputs = sp_strategy.preprocess_inputs(sp_label_inputs)
        sp_local_labels = sp_label_inputs["labels"]

        sequence_parallel.extra_kwargs["position_ids"] = position_ids.clone()
        sp_states, sp_state_handles = _capture_router_state(getattr(model_sp, "model", model_sp))
        sp_out = model_sp(
            inputs_embeds=sp_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        for h in sp_state_handles:
            h.remove()
        sp_local_logits = sp_out.logits
        sp_out = sp_strategy.postprocess_outputs(sp_out)
        sp_logits = sp_out.logits.detach()

        # Forward alignment (full-seq logits reconstructed by SP gather).
        if not torch.allclose(sp_logits, base_logits, rtol=1e-3, atol=1e-4):
            diff = (sp_logits - base_logits).abs()
            raise AssertionError(
                f"[rank{rank}] logits not close: mean_abs={diff.mean().item():.6e} "
                f"max_abs={diff.max().item():.6e} (rtol=1e-3, atol=1e-4)"
            )

        # Router alignment on this rank's slice: compare selected experts exactly.
        # SP captures only local tokens; baseline captures full tokens (we slice it).
        sp_blocks = _find_moe_blocks(getattr(model_sp, "model", model_sp))
        assert len(sp_states) == len(sp_blocks) == len(base_blocks)
        for idx, (base_state, sp_state) in enumerate(zip(base_states, sp_states)):
            base_sel = base_state["selected_experts"].view(batch_size, seq_len, -1)[:, start:end, :].contiguous()
            # sp_sel is already local-seq; shape should match [B, local_seq, top_k] or [tokens, top_k]
            sp_sel = sp_state["selected_experts"]
            if sp_sel.dim() == 2:
                sp_sel = sp_sel.view(batch_size, end - start, -1)
            if not torch.equal(base_sel, sp_sel):
                mismatch = (base_sel != sp_sel).sum().item()
                print(f"[rank{rank}] block[{idx}] selected_experts mismatch count={mismatch}", flush=True)
            assert torch.equal(base_sel, sp_sel)

        # Backward alignment (expert grads on active local experts for this slice).
        sp_loss_sum = F.cross_entropy(
            sp_local_logits.view(-1, sp_local_logits.size(-1)),
            sp_local_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        sp_loss_sum.backward()

        sp_active_grads: Dict[str, torch.Tensor] = {}
        for block, state in zip(sp_blocks, sp_states):
            active = torch.unique(state["selected_experts"])
            sp_active_grads.update(_collect_active_local_expert_grad_tensors(block, active))

        # Mixed precision + extra collectives => allow a bit more slack on gradients than logits.
        grad_rel_tol = float(os.environ.get("TWINKLE_EXPERT_GRAD_REL_TOL", "1e-3"))
        _compare_grad_dicts(rank=rank, baseline=base_active_grads, sp=sp_active_grads, rel_tol=grad_rel_tol)
    finally:
        dist.destroy_process_group()


class TestExpertParallelFSDPSequenceParallelPretrained(unittest.TestCase):
    def test_qwen3_moe_pretrained_ep_fsdp_sp_alignment(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this test.")
        world_size = 4
        if torch.cuda.device_count() < world_size:
            self.skipTest("Requires at least 4 GPUs for EP+FSDP+SP alignment test.")
        model_id = os.environ.get("QWEN3_MOE_MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        local_files_only = os.environ.get("QWEN3_MOE_LOCAL_ONLY", "1") != "0"
        try:
            _load_qwen3_moe_config(model_id, local_files_only)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Qwen3 MoE model not available locally: {exc}")
        port = _find_free_port()
        mp.spawn(
            _run_worker_ep_fsdp_sp_align,
            args=(world_size, port, model_id, local_files_only),
            nprocs=world_size,
            join=True,
        )


def _run_worker_fsdp_sp_align(
    rank: int,
    world_size: int,
    port: int,
    model_id: str,
    local_files_only: bool,
):
    """Compare FSDP-only vs FSDP+SP for a Qwen3 MoE pretrained model."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    strict = os.environ.get("TWINKLE_STRICT_ALIGN", "0") == "1"
    if strict:
        _enable_strict_determinism(seed=1234)

    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA (4 GPUs).")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        device_id=device,
        timeout=timedelta(minutes=15),
    )
    dist.barrier()

    try:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        # 4 GPUs: fsdp=4, dp=1; SP is derived via ulysses_size=2 over raw data ranks (fsdp).
        device_mesh = DeviceMesh.from_sizes(
            fsdp_size=world_size,
            dp_size=1,
            ulysses_size=2,
            device_type="cuda",
        )
        sp_size = 2
        sp_group = _get_sp_group_from_device_mesh(device_mesh, sp_size)

        batch_size = 2
        seq_len = 16

        # Loading the pretrained checkpoint twice per-rank is very slow and can look "hung".
        # Load once, then deepcopy to get a second identical model for the SP variant.
        model_fsdp = _load_qwen3_moe_pretrained(model_id, local_files_only, device)
        model_sp = copy.deepcopy(model_fsdp)
        vocab_size = int(model_fsdp.config.vocab_size)

        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
        dist.broadcast(input_ids, src=0)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        labels_raw = input_ids.clone()
        labels_raw[:, 0] = -100
        labels_shifted = torch.roll(labels_raw, shifts=-1, dims=1)

        fsdp_strategy = NativeFSDPStrategy(device_mesh=device_mesh, mixed_precision="bf16", fsdp_config={})

        # --- Baseline: FSDP only (no SP). Use full-sequence loss (sum over all tokens).
        embed_fsdp = model_fsdp.get_input_embeddings()
        _ensure_embed_tokens(model_fsdp, embed_fsdp)
        base_embeds = embed_fsdp(input_ids).detach().requires_grad_(True)
        model_fsdp, _ = fsdp_strategy.wrap_model(model_fsdp, optimizer=None)

        base_out = model_fsdp(
            inputs_embeds=base_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        base_logits = base_out.logits.detach()
        base_loss_sum = F.cross_entropy(
            base_out.logits.view(-1, base_out.logits.size(-1)),
            labels_shifted.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        base_loss_sum.backward()
        base_embed_grad = base_embeds.grad.detach().cpu()
        model_fsdp.zero_grad(set_to_none=True)

        # --- Variant: FSDP + SP.
        sp_strategy = SequenceParallelStrategy(
            device_mesh=device_mesh,
            sp_config={"enabled": True, "ulysses_size": sp_size, "gather_logits": True},
            model=model_sp,
            tokenizer_id=model_id,
        )
        sp_strategy.initialize()

        # Compute inputs_embeds before DTensor wrapping to avoid mixed Tensor/DTensor embedding op.
        embed_sp = model_sp.get_input_embeddings()
        _ensure_embed_tokens(model_sp, embed_sp)
        sp_embeds = embed_sp(input_ids).detach().requires_grad_(True)
        model_sp, _ = fsdp_strategy.wrap_model(model_sp, optimizer=None)

        sp_label_inputs = {"labels": labels_raw, "position_ids": position_ids}
        sp_label_inputs = sp_strategy.preprocess_inputs(sp_label_inputs)
        sp_local_labels = sp_label_inputs["labels"]

        sequence_parallel.extra_kwargs["position_ids"] = position_ids.clone()
        sp_out = model_sp(
            inputs_embeds=sp_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        sp_local_logits = sp_out.logits
        sp_out = sp_strategy.postprocess_outputs(sp_out)
        sp_logits = sp_out.logits.detach()

        # Forward alignment (full-seq logits reconstructed by SP gather).
        if not torch.allclose(sp_logits, base_logits, rtol=1e-3, atol=1e-4):
            diff = (sp_logits - base_logits).abs()
            print(
                f"[rank{rank}] logits diff mean={diff.mean().item():.6e} max={diff.max().item():.6e}",
                flush=True,
            )
        assert torch.allclose(sp_logits, base_logits, rtol=1e-3, atol=1e-4)

        # Backward alignment: local CE(sum) on SP, compare gathered full-seq inputs_embeds grads.
        sp_loss_sum = F.cross_entropy(
            sp_local_logits.view(-1, sp_local_logits.size(-1)),
            sp_local_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        sp_loss_sum.backward()
        sp_embed_grad = sp_embeds.grad.detach().cpu()

        # Backward alignment: gather SP local-seq grads into a full-seq grad and compare.
        start, end = _sp_slice_range_for_seq_len(seq_len, sp_group=sp_group, sp_size=sp_size)
        sp_local = sp_embed_grad.to(device=device, dtype=torch.float32)[:, start:end].contiguous()
        sp_full = _gather_full_seq_grad_from_sp(sp_local, sp_group=sp_group)
        base_full = base_embed_grad.to(device=device, dtype=torch.float32)[:, :seq_len].contiguous()
        diff = sp_full - base_full
        rel = diff.norm() / (base_full.norm() + 1e-12)
        grad_rel_tol = float(os.environ.get("TWINKLE_INPUT_GRAD_REL_TOL", "1e-2"))
        if rel.item() > grad_rel_tol:
            abs_diff = diff.abs()
            raise AssertionError(
                f"[rank{rank}] inputs_embeds.grad(full) not close: mean_abs={abs_diff.mean().item():.6e} "
                f"max_abs={abs_diff.max().item():.6e} rel_norm={rel.item():.6e} tol={grad_rel_tol:.1e}"
            )
        assert rel.item() <= grad_rel_tol
    finally:
        dist.destroy_process_group()


class TestFSDPSequenceParallelQwen3MoePretrained(unittest.TestCase):
    def test_qwen3_pretrained_fsdp_sp_alignment(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this test.")
        world_size = 4
        if torch.cuda.device_count() < world_size:
            self.skipTest("Requires at least 4 GPUs for FSDP+SP alignment test.")
        model_id = os.environ.get("QWEN3_MOE_MODEL_ID", "Qwen/Qwen3-0.6B")
        local_files_only = os.environ.get("QWEN3_MOE_LOCAL_ONLY", "1") != "0"
        try:
            _load_qwen3_moe_config(model_id, local_files_only)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Qwen3 MoE model not available locally: {exc}")
        port = _find_free_port()
        mp.spawn(
            _run_worker_fsdp_sp_align,
            args=(world_size, port, model_id, local_files_only),
            nprocs=world_size,
            join=True,
        )
