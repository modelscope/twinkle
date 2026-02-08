#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Test STANDALONE weight synchronization between MegatronModel and vLLM sampler.

This script tests the checkpoint engine weight sync flow when the training
model uses Megatron-Core (with TP/PP parallelism) and the inference sampler
uses vLLM:

    1. Create MegatronModel (with real weights, TP=2) and VLLMSampler (with dummy weights)
    2. Sample with dummy weights → garbage output
    3. Sync weights from MegatronModel → VLLMSampler via CheckpointEngineManager
    4. Sample with synced weights → coherent output
    5. Verify that outputs differ (proof that weights were synced)

The Megatron bridge internally handles TP allgather during export, converting
Megatron-format weights to HuggingFace format on-the-fly.

Usage:
    # 2 Megatron GPUs (TP=2) + 2 sampler GPUs (4 GPUs total, using GPUs 4-7)
    CUDA_VISIBLE_DEVICES=4,5,6,7 python tests/sampler/test_megatron_weight_sync.py

    # 2 Megatron GPUs (TP=2) + 1 sampler GPU (3 GPUs total)
    CUDA_VISIBLE_DEVICES=4,5,6 python tests/sampler/test_megatron_weight_sync.py --sampler-gpus 1

    # Custom model
    CUDA_VISIBLE_DEVICES=4,5,6,7 TEST_MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
        python tests/sampler/test_megatron_weight_sync.py --tp-size 2
"""

import os
import sys
import time
import argparse
import logging

# Must set before importing anything
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'
# Prevent hanging during NCCL weight sync in disaggregated mode
os.environ['NCCL_CUMEM_ENABLE'] = '0'

# Model configuration — use a small model for testing
MODEL_ID = os.environ.get('TEST_MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')

logger = logging.getLogger(__name__)


def log(msg):
    """Print message with timestamp."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def wait_result(result):
    """Resolve lazy collect / ray object ref to actual value."""
    if hasattr(result, '_is_lazy_collect') and result._is_lazy_collect:
        return result()
    if hasattr(result, 'wait'):
        return result.wait()
    if callable(result) and hasattr(result, '_get_result'):
        return result()
    return result


def get_model_path():
    """Resolve model_id to a local cache path (for offline environments)."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        _cache = snapshot_download(MODEL_ID, local_files_only=True)
        if _cache:
            return _cache
    except Exception:
        pass
    return MODEL_ID


# =============================================================================
# Test: Megatron Standalone Weight Sync
# =============================================================================

def test_megatron_weight_sync(
    model_gpus: int = 2,
    sampler_gpus: int = 2,
    tp_size: int = 2,
    pp_size: int = 1,
):
    """Test weight sync from MegatronModel to VLLMSampler via NCCL broadcast.

    Architecture:
        Model workers  : GPU 0 .. model_gpus-1   (Megatron, TP=tp_size, real weights)
        Sampler workers: GPU model_gpus .. total-1 (vLLM, dummy weights)

    The Megatron bridge converts weights from Megatron format to HuggingFace
    format during export.  TP allgather is handled internally by the bridge.
    Only model_actor[0] broadcasts via the checkpoint engine's NCCL group;
    other model actors consume the generator (triggering TP allgather) but
    do not participate in the broadcast.
    """
    import twinkle
    from twinkle import DeviceGroup, DeviceMesh
    from twinkle.model import MegatronModel
    from twinkle.sampler import VLLMSampler
    from twinkle.template import Template
    from twinkle.checkpoint_engine import CheckpointEngineManager
    from twinkle.data_format import Trajectory
    from twinkle.data_format.types import SamplingParams

    total_gpus = model_gpus + sampler_gpus
    model_path = get_model_path()

    # Validate parallelism config
    assert model_gpus == tp_size * pp_size, (
        f"model_gpus ({model_gpus}) must equal tp_size * pp_size "
        f"({tp_size} * {pp_size} = {tp_size * pp_size})"
    )

    log("=" * 70)
    log(f"TEST: Megatron Standalone Weight Sync")
    log(f"  Model  : GPU 0-{model_gpus - 1}  ({model_gpus} workers, TP={tp_size}, PP={pp_size})")
    log(f"  Sampler: GPU {model_gpus}-{total_gpus - 1}  ({sampler_gpus} workers)")
    log(f"  Model  : {model_path}")
    log("=" * 70)

    # ── Initialize Twinkle in Ray mode ────────────────────────────────
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total_gpus,
        groups=[
            DeviceGroup(
                name='model',
                ranks=list(range(model_gpus)),
                device_type='GPU',
                gpus_per_worker=1,
            ),
            DeviceGroup(
                name='sampler',
                ranks=list(range(model_gpus, total_gpus)),
                device_type='GPU',
                gpus_per_worker=1,
            ),
        ],
    )

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        from modelscope import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ── Create MegatronModel (real weights) ────────────────────────────
    log("\nCreating MegatronModel (real weights)...")
    model_device_mesh = DeviceMesh.from_sizes(
        world_size=model_gpus,
        dp_size=model_gpus // (tp_size * pp_size),
        tp_size=tp_size,
        pp_size=pp_size,
    )
    model = MegatronModel(
        model_id=model_path,
        device_mesh=model_device_mesh,
        mixed_precision='bf16',
        sequence_parallel=(tp_size > 1),
        remote_group='model',
    )
    log("  MegatronModel created successfully")

    # ── Create Sampler (dummy weights) ────────────────────────────────
    log("Creating Sampler (dummy weights)...")
    sampler = VLLMSampler(
        model_id=model_path,
        engine_args={
            'load_format': 'dummy',
            'gpu_memory_utilization': 0.3,
            'max_model_len': 256,
            'enforce_eager': True,
            'enable_sleep_mode': True,
            'enable_lora': False,
        },
        device_mesh=DeviceMesh.from_sizes(world_size=sampler_gpus, dp_size=sampler_gpus),
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=model_path)
    log("  VLLMSampler created successfully")

    # Wait for vLLM initialization
    log("Waiting for vLLM initialization...")
    time.sleep(5)

    # ── Helper: sample one prompt ─────────────────────────────────────
    def do_sample(prompt: str, max_tokens: int = 32) -> str:
        traj = Trajectory(messages=[{'role': 'user', 'content': prompt}])
        response = wait_result(
            sampler.sample(traj, SamplingParams(max_tokens=max_tokens, temperature=0.0))
        )
        if response and response.sequences:
            tokens = response.sequences[0].tokens
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            return tokenizer.decode(tokens, skip_special_tokens=True)
        return ""

    # ── Sample BEFORE sync (dummy weights → garbage) ──────────────────
    log("\n--- Sampling BEFORE weight sync (dummy weights) ---")
    text_before = do_sample("What is 2+2?")
    log(f"  Output: '{text_before[:100]}'")

    # ── Sync weights: MegatronModel → Sampler via NCCL ────────────────
    log("\n--- Syncing weights via CheckpointEngineManager ---")
    manager = CheckpointEngineManager(
        model=model,
        sampler=sampler,
    )

    sync_start = time.time()
    manager.sync_weights()
    sampler.reset_prefix_cache()
    sync_time = time.time() - sync_start
    log(f"  Weight sync completed in {sync_time:.2f}s")

    # ── Sample AFTER sync (real weights → coherent) ───────────────────
    log("\n--- Sampling AFTER weight sync (real weights) ---")
    text_after = do_sample("What is 2+2?")
    log(f"  Output: '{text_after[:100]}'")

    # ── Verification ──────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("VERIFICATION")
    log("=" * 70)

    outputs_differ = text_before != text_after
    log(f"  Outputs differ after sync: {outputs_differ}")

    if outputs_differ:
        log("  PASS: Weight sync verified — outputs changed after sync.")
        if "4" in text_after.lower() or "four" in text_after.lower():
            log("  BONUS: Model correctly answered '2+2' question!")
    else:
        log("  FAIL: Outputs are identical — weight sync may have failed.")

    sampler.shutdown()
    return outputs_differ


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Megatron standalone weight synchronization')
    parser.add_argument('--model-gpus', type=int, default=2,
                        help='Number of GPUs for Megatron model (default: 2)')
    parser.add_argument('--sampler-gpus', type=int, default=2,
                        help='Number of GPUs for vLLM sampler (default: 2)')
    parser.add_argument('--tp-size', type=int, default=2,
                        help='Tensor parallel size (default: 2)')
    parser.add_argument('--pp-size', type=int, default=1,
                        help='Pipeline parallel size (default: 1)')
    args = parser.parse_args()

    log(f"Starting Megatron standalone weight sync test...")
    log(f"  Model GPUs:   {args.model_gpus}")
    log(f"  Sampler GPUs: {args.sampler_gpus}")
    log(f"  TP size:      {args.tp_size}")
    log(f"  PP size:      {args.pp_size}")
    log(f"  Model ID:     {MODEL_ID}")

    try:
        success = test_megatron_weight_sync(
            model_gpus=args.model_gpus,
            sampler_gpus=args.sampler_gpus,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
        )
    except Exception as e:
        log(f"\nTest failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False

    log("\n" + "=" * 70)
    log(f"RESULT: {'PASS' if success else 'FAIL'}")
    log("=" * 70)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
