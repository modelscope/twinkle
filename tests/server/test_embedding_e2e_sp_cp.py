# Copyright (c) ModelScope Contributors. All rights reserved.
"""Embedding training E2E validation under SP/CP distributed configs (task 5.1).

This module reuses the synthetic contrastive dataset and the embedding
verification call sequence from ``tests/server/test_embedding_e2e.py`` and reruns
the embedding training flow under a MULTI-GPU device mesh that enables sequence
parallel (SP, ``ulysses_size > 1``) and/or context parallel (CP, ``cp_size > 1``),
then asserts that the resulting ``pos_sim`` (anchor-positive cosine similarity)
stays within a reasonable tolerance of the single-card baseline.

The distributed device mesh is constructed with
``DeviceMesh.from_sizes(fsdp_size=..., ulysses_size=..., cp_size=...)`` (see
``src/twinkle/utils/device_mesh.py``) and driven across ranks with
``torch.multiprocessing.spawn``, mirroring the multi-rank harness in
``tests/transformers/test_sequence_parallel_and_cp.py``. Each rank builds a real
``MultiLoraTransformersModel`` bound to that mesh; SP is enabled automatically
inside the model whenever ``device_mesh.ulysses_size > 1``.

==============================================================================
ENVIRONMENT CONSTRAINTS (MANDATORY — READ BEFORE RUNNING)
==============================================================================
1. MULTI-GPU required. These cases spawn ``world_size`` distributed workers, one
   per GPU (SP-only needs >= 2 GPUs; combined CP+SP needs >= 4 GPUs). They are
   gated behind ``TWINKLE_TEST_GPU_E2E=1`` AND an actual multi-GPU CUDA runtime,
   and are SKIPPED automatically on a machine without enough GPUs (including this
   GPU-less dev machine).

2. ``twinkle`` conda env required. Run every case inside the ``twinkle`` conda
   env, e.g.:

       TWINKLE_TEST_GPU_E2E=1 conda run -n twinkle \
           pytest tests/server/test_embedding_e2e_sp_cp.py -v

   On a GPU-less machine the file still collects/imports cleanly and skips:

       conda run -n twinkle pytest tests/server/test_embedding_e2e_sp_cp.py -v
==============================================================================
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
import tempfile
from typing import Any, Dict, List, Optional

# Ensure project root is importable for both pytest and direct execution.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

# Reuse the task 4.1-4.5 scaffolding (dataset, minibatching, model id, metric
# parsing, loss extraction, GPU gate) so the SP/CP flow exercises the SAME
# synthetic dataset and the SAME verification call sequence as the single-card
# path — the only difference is the distributed device mesh.
from tests.server.test_embedding_e2e import (
    DEFAULT_SEQ_LEN,
    DEFAULT_TEMPERATURE,
    MODEL_ID,
    _parse_metric_float,
    build_synthetic_contrastive_dataset,
    extract_loss,
    gpu_e2e_enabled,
    iter_minibatches,
)
from tests.server.integration.e2e_helpers import log

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Dataset / training shape (kept small for GPU speed while still giving InfoNCE a
# real contrastive signal). Mirrors the pos_sim-trend shape used in task 4.5.
SP_CP_NUM_PAIRS = 6        # -> 12 samples
SP_CP_BATCH_SIZE = 4       # -> 3 minibatches == 3 training steps
SP_CP_TRAIN_STEPS = 8      # real training steps before measuring pos_sim
SP_CP_LR = 1e-3            # aggressive LR so weights actually move within few steps
SP_CP_SEED = 1234          # identical seed on baseline + distributed runs

# ``pos_sim`` is a cosine similarity in [-1, 1]. SP/CP pooling is mathematically
# equivalent to the single-card path, so trained values should agree closely;
# a loose absolute tolerance accommodates bf16 + distributed reduction noise.
POS_SIM_ATOL = 5e-2

# Distributed configs exercised by the parametrized case.
#
# In the transformers backend, sequence/context parallel is driven by
# ``device_mesh.ulysses_size`` (see TransformersModel._decide_strategy: SP is
# enabled whenever ``ulysses_size > 1``). Ulysses degree is *derived* from the
# data ranks and therefore does NOT multiply the mesh world size — so
# ``from_sizes(fsdp_size=W, ulysses_size=U)`` yields exactly ``W`` ranks with a
# single ulysses group of size ``U``. Whether that group behaves as pure SP or a
# CP (ring-attention) mode is derived internally from ``ulysses_size`` vs. the
# model's attention-head count (see tests/transformers/test_sequence_parallel_and_cp.py),
# which is why both SP and CP are exercised through ``ulysses_size`` here.
#
# Each config is skipped unless its required GPU count is available.
DIST_CONFIGS = [
    # SP: 2 GPUs, ulysses_size=2 (sequence parallel enabled over 2 ranks).
    {'name': 'sp2', 'world_size': 2, 'ulysses_size': 2, 'cp_size': None},
    # SP over 4 GPUs: ulysses_size=4 (larger sequence/context-parallel group).
    {'name': 'sp4', 'world_size': 4, 'ulysses_size': 4, 'cp_size': None},
]


def _cuda_device_count() -> int:
    """Return the CUDA device count, or 0 when torch/CUDA is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def multi_gpu_e2e_enabled(min_devices: int = 2) -> bool:
    """True only when GPU e2e is enabled AND enough CUDA devices are present."""
    return gpu_e2e_enabled() and _cuda_device_count() >= min_devices


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


# ═══════════════════════════════════════════════════════════════════════════
# Distributed worker: build an SP/CP device mesh, train, report pos_sim.
#
# Runs inside every spawned rank. Must be a module-level function so it is
# picklable by ``torch.multiprocessing.spawn`` (start method: spawn).
# ═══════════════════════════════════════════════════════════════════════════


def _init_distributed(rank: int, world_size: int, port: int):
    """Initialize the process group + pin this rank to its CUDA device."""
    import torch
    import torch.distributed as dist

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://127.0.0.1:{port}',
        )
    return device


def _seed_everything(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_embedding_model_with_mesh(
    world_size: int,
    ulysses_size: Optional[int],
    cp_size: Optional[int],
    *,
    adapter_name: str,
    temperature: float,
    lr: float,
):
    """Build a real ``MultiLoraTransformersModel`` bound to an SP/CP device mesh.

    Uses the SAME bare-library primitives as the single-card path in task 4.5
    (InputProcessor + InfonceLoss + EmbeddingMetric + AdamW), the only difference
    being the distributed ``DeviceMesh`` passed to the constructor. SP is enabled
    automatically by the transformers backend whenever ``ulysses_size > 1``.
    """
    from peft import LoraConfig

    from twinkle.loss import InfonceLoss
    from twinkle.metric import EmbeddingMetric
    from twinkle.model import MultiLoraTransformersModel
    from twinkle.processor import InputProcessor
    from twinkle.utils import DeviceMesh

    # fsdp_size = world_size shards the model across all ranks; ulysses_size
    # enables sequence/context parallel derived from those data ranks (it does
    # not change the world size). cp_size, when provided, adds an explicit CP
    # mesh dimension (and therefore multiplies the world size).
    mesh = DeviceMesh.from_sizes(
        fsdp_size=world_size,
        dp_size=1,
        ulysses_size=ulysses_size,
        cp_size=cp_size,
        device_type='cuda',
    )

    # find_unused_parameters=True is REQUIRED for embedding training: only the last
    # hidden states contribute to the InfoNCE loss, so some parameters receive no
    # gradient and DDP would otherwise abort the reduction (see Embedding训练.md).
    model = MultiLoraTransformersModel(
        model_id=MODEL_ID, device_mesh=mesh,
        ddp_config={'find_unused_parameters': True})
    model.add_adapter_to_model(adapter_name, LoraConfig(target_modules='all-linear'))
    model.set_processor(InputProcessor, adapter_name=adapter_name)
    model.set_loss(
        InfonceLoss, temperature=temperature, use_batch=True, hard_negatives=None,
        adapter_name=adapter_name)
    model.set_optimizer(optimizer_cls='AdamW', lr=lr, adapter_name=adapter_name)
    model.add_metric(EmbeddingMetric, is_training=True, adapter_name=adapter_name)
    return model, adapter_name


def _distributed_embedding_pos_sim_worker(
    rank: int,
    world_size: int,
    port: int,
    ulysses_size: Optional[int],
    cp_size: Optional[int],
    seed: int,
    result_path: str,
) -> None:
    """Spawned worker: run embedding training under the given mesh, emit pos_sim.

    Every rank trains identically (FSDP-sharded, SP/CP-parallel). Rank 0 writes
    the final ``pos_sim`` (plus the per-step losses) to ``result_path`` as JSON so
    the parent process can compare it against the single-card baseline.
    """
    import gc

    import torch
    import torch.distributed as dist

    device = _init_distributed(rank, world_size, port)
    model = None
    try:
        _seed_everything(seed)

        # SAME synthetic dataset + minibatch sequence as the single-card path.
        dataset = build_synthetic_contrastive_dataset(
            SP_CP_NUM_PAIRS, seq_len=DEFAULT_SEQ_LEN, seed=0)
        minibatches = iter_minibatches(dataset, batch_size=SP_CP_BATCH_SIZE)

        adapter_name = f'emb_sp_cp_adapter_u{ulysses_size}_c{cp_size}'
        model, adapter_name = _build_embedding_model_with_mesh(
            world_size, ulysses_size, cp_size,
            adapter_name=adapter_name, temperature=DEFAULT_TEMPERATURE, lr=SP_CP_LR)

        losses: List[float] = []
        # Repeat the (short) minibatch sequence until SP_CP_TRAIN_STEPS steps run.
        step = 0
        while step < SP_CP_TRAIN_STEPS:
            for mb in minibatches:
                if step >= SP_CP_TRAIN_STEPS:
                    break
                outputs = model.forward_backward(inputs=mb, task='embedding', adapter_name=adapter_name)
                losses.append(extract_loss(outputs))
                model.clip_grad_and_step(max_grad_norm=1.0, adapter_name=adapter_name)
                step += 1

        metric = model.calculate_metric(is_training=True, adapter_name=adapter_name)
        pos_sim = _parse_metric_float(metric, 'pos_sim')

        if rank == 0:
            with open(result_path, 'w') as f:
                json.dump({'pos_sim': pos_sim, 'losses': losses}, f)
            log(f'[sp_cp worker u={ulysses_size} c={cp_size}] pos_sim={pos_sim:.6f} '
                f'losses={[round(x, 4) for x in losses]}')
        dist.barrier()
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_embedding_pos_sim(
    world_size: int,
    ulysses_size: Optional[int],
    cp_size: Optional[int],
    *,
    seed: int = SP_CP_SEED,
) -> Dict[str, Any]:
    """Spawn ``world_size`` workers, run embedding training, return rank-0 result.

    ``world_size == 1`` with ``ulysses_size in (None, 1)`` yields the single-card
    baseline (SP/CP disabled). Larger world sizes exercise the SP/CP mesh. The
    spawn path is identical for both so the ONLY variable is the device mesh.
    """
    import torch.multiprocessing as mp

    port = _find_free_port()
    fd, result_path = tempfile.mkstemp(prefix='emb_sp_cp_', suffix='.json')
    os.close(fd)
    try:
        mp.spawn(
            _distributed_embedding_pos_sim_worker,
            args=(world_size, port, ulysses_size, cp_size, seed, result_path),
            nprocs=world_size,
            join=True,
        )
        with open(result_path) as f:
            return json.load(f)
    finally:
        try:
            os.remove(result_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Single-card baseline (module-scoped: computed once, reused by every config).
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope='module')
def single_card_pos_sim():
    """Baseline ``pos_sim`` from a single-card (SP/CP-disabled) embedding run.

    Gated behind the multi-GPU e2e requirement so it skips cleanly on machines
    without enough GPUs. Runs through the same spawn harness with
    ``world_size=1`` / ``ulysses_size=1`` so the baseline and the distributed
    runs differ ONLY in the device mesh.
    """
    if not multi_gpu_e2e_enabled(min_devices=2):
        pytest.skip(
            'Requires TWINKLE_TEST_GPU_E2E=1 and >= 2 CUDA devices '
            '(multi-GPU SP/CP embedding e2e).')

    result = _run_embedding_pos_sim(world_size=1, ulysses_size=1, cp_size=None)
    pos_sim = result['pos_sim']
    log(f'[sp_cp baseline] single-card pos_sim={pos_sim:.6f}')
    return pos_sim


# ═══════════════════════════════════════════════════════════════════════════
# Task 5.1 — SP/CP pos_sim consistency (multi-GPU, GPU-gated).
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize('config', DIST_CONFIGS, ids=[c['name'] for c in DIST_CONFIGS])
def test_gpu_sp_cp_pos_sim_matches_single_card(config, single_card_pos_sim):
    """SP/CP pos_sim agrees with the single-card baseline within tolerance.

    Reruns the task 4.1-4.5 embedding verification flow on the SAME synthetic
    contrastive dataset under a distributed device mesh built with
    ``DeviceMesh.from_sizes(fsdp_size=world_size, ulysses_size=..., cp_size=...)``
    (SP when ``ulysses_size > 1``, CP when ``cp_size > 1``), then asserts the
    resulting ``pos_sim`` stays within ``POS_SIM_ATOL`` of the single-card value.

    Multi-GPU + GPU-gated (TWINKLE_TEST_GPU_E2E=1): each config is skipped unless
    its required GPU count is available, so the file collects/skips cleanly on a
    GPU-less machine and asserts real distributed numeric semantics only on
    multi-GPU CI.

    Validates: Requirements 1.3, 7.2
    """
    world_size = int(config['world_size'])
    if not multi_gpu_e2e_enabled(min_devices=world_size):
        pytest.skip(
            f"Config {config['name']!r} needs TWINKLE_TEST_GPU_E2E=1 and "
            f'>= {world_size} CUDA devices (have {_cuda_device_count()}).')

    result = _run_embedding_pos_sim(
        world_size=world_size,
        ulysses_size=config['ulysses_size'],
        cp_size=config['cp_size'],
    )
    dist_pos_sim = result['pos_sim']

    # pos_sim is a valid cosine similarity.
    assert math.isfinite(dist_pos_sim), f"non-finite pos_sim: {dist_pos_sim!r}"
    assert -1.0001 <= dist_pos_sim <= 1.0001, f'pos_sim out of range: {dist_pos_sim!r}'

    delta = abs(dist_pos_sim - single_card_pos_sim)
    log(f"[sp_cp {config['name']}] dist_pos_sim={dist_pos_sim:.6f} "
        f'baseline={single_card_pos_sim:.6f} delta={delta:.6f}')
    assert delta <= POS_SIM_ATOL, (
        f"[{config['name']}] SP/CP pos_sim diverged from single-card baseline: "
        f'dist={dist_pos_sim:.6f}, baseline={single_card_pos_sim:.6f}, '
        f'delta={delta:.6f} > atol={POS_SIM_ATOL}')


# ═══════════════════════════════════════════════════════════════════════════
# Local (GPU-less) collection guards — verify the file imports/collects/skips
# cleanly and the mesh-config helpers behave, WITHOUT requiring any GPU.
# ═══════════════════════════════════════════════════════════════════════════


def test_multi_gpu_gate_reflects_env_and_devices():
    """The multi-GPU gate is False unless BOTH the env flag and >=N GPUs exist."""
    expected = gpu_e2e_enabled() and _cuda_device_count() >= 2
    assert multi_gpu_e2e_enabled(min_devices=2) == expected


def test_dist_configs_are_well_formed():
    """Every declared SP/CP config enables SP and/or CP with a matching world size."""
    for cfg in DIST_CONFIGS:
        assert cfg['world_size'] >= 2, cfg
        sp = (cfg['ulysses_size'] or 1) > 1
        cp = (cfg['cp_size'] or 1) > 1
        assert sp or cp, f'config enables neither SP nor CP: {cfg}'
        # ulysses/cp parallelism must fit within the world size.
        assert (cfg['ulysses_size'] or 1) <= cfg['world_size'], cfg
        assert (cfg['cp_size'] or 1) <= cfg['world_size'], cfg


def test_device_mesh_from_sizes_builds_sp_cp_mesh():
    """DeviceMesh.from_sizes wires ulysses_size/cp_size into a multi-rank mesh.

    Pure CPU construction (no distributed init, no GPU): confirms the exact
    ``from_sizes(fsdp_size=..., ulysses_size=..., cp_size=...)`` usage the workers
    rely on. ``ulysses_size`` is derived from the data ranks and does NOT change
    the mesh world size (SP degree lives inside the existing ranks), whereas
    ``cp_size`` adds an explicit mesh dimension that multiplies the world size.
    """
    from twinkle.utils import DeviceMesh

    # SP mesh (2 ranks, ulysses_size=2): ulysses does not inflate world_size.
    sp_mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=1, ulysses_size=2, device_type='cuda')
    assert sp_mesh.world_size == 2
    assert sp_mesh.ulysses_size == 2

    # SP mesh over 4 ranks (ulysses_size=4): still exactly fsdp_size ranks.
    sp4_mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=1, ulysses_size=4, device_type='cuda')
    assert sp4_mesh.world_size == 4
    assert sp4_mesh.ulysses_size == 4

    # CP mesh dimension (cp_size=2) multiplies the world size and is recorded as
    # an explicit 'cp' dim: fsdp(2) x dp(1) x cp(2) == 4 ranks.
    cp_mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=1, cp_size=2, device_type='cuda')
    assert cp_mesh.world_size == 4
    assert cp_mesh.has_dim('cp')
    assert cp_mesh.get_dim_size('cp') == 2
