#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Test STANDALONE weight synchronization between training model and vLLM sampler.

This script serves as both a test and a minimal demo of the weight sync flow
used during RL training:

    1. Create TransformersModel (with real weights) and VLLMSampler (with dummy weights)
    2. Sample with dummy weights → garbage output
    3. Sync weights from Model → Sampler via CheckpointEngineManager (NCCL broadcast)
    4. Sample with synced weights → coherent output
    5. Verify that outputs differ (proof that weights were synced)

Usage:
    # 2 model GPUs + 2 sampler GPUs (requires 4 GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/sampler/test_weight_sync.py --model-gpus 2 --sampler-gpus 2

    # 1 model GPU + 1 sampler GPU (requires 2 GPUs)
    CUDA_VISIBLE_DEVICES=0,1 python tests/sampler/test_weight_sync.py

Note:
    - Requires Ray and multiple GPUs
    - Set TEST_MODEL_ID environment variable to use a different model
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
# See: https://docs.vllm.ai/en/latest/usage/troubleshooting.html#known-issues
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
# Test: Standalone Weight Sync
# =============================================================================

def test_standalone_weight_sync(model_gpus: int = 1, sampler_gpus: int = 1):
    """Test weight sync in STANDALONE mode (model and sampler on different GPUs).

    Architecture:
        Model workers  : GPU 0 .. model_gpus-1   (training, real weights)
        Sampler workers: GPU model_gpus .. total-1 (inference, dummy weights)

    Weight sync flow (managed by CheckpointEngineManager):
        1. prepare             — allocate NCCL buffers, ZMQ metadata server
        2. build_topology      — model[0]→rank0 (source), sampler→rank1..N
        3. init_process_group  — temporary NCCL group
        4. send / receive      — NCCL broadcast (parallel)
        5. finalize            — release buffers, close ZMQ
    """
    import twinkle
    from twinkle import DeviceGroup, DeviceMesh
    from twinkle.model.transformers import TransformersModel
    from twinkle.sampler import VLLMSampler
    from twinkle.template import Template
    from twinkle.checkpoint_engine import CheckpointEngineManager
    from transformers import AutoTokenizer
    from twinkle.data_format import Trajectory
    from twinkle.data_format.sampling import SamplingParams

    total_gpus = model_gpus + sampler_gpus
    model_path = get_model_path()

    log("=" * 70)
    log(f"TEST: Standalone Weight Sync")
    log(f"  Model  : GPU 0-{model_gpus - 1}  ({model_gpus} workers)")
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

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ── Create Model (real weights) ───────────────────────────────────
    log("\nCreating Model (real weights)...")
    model = TransformersModel(
        model_id=model_path,
        device_mesh=DeviceMesh.from_sizes(world_size=model_gpus, dp_size=model_gpus),
        remote_group='model',
    )

    # ── Create Sampler (dummy weights) ────────────────────────────────
    log("Creating Sampler (dummy weights)...")
    sampler = VLLMSampler(
        model_id=model_path,
        engine_args={
            'load_format': 'dummy',         # start with random weights
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

    # Wait for vLLM initialization
    log("Waiting for vLLM initialization...")
    time.sleep(3)

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

    # ── Sync weights: Model → Sampler via NCCL ────────────────────────
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
    parser = argparse.ArgumentParser(description='Test STANDALONE weight synchronization')
    parser.add_argument('--model-gpus', type=int, default=1,
                        help='Number of GPUs for model (training)')
    parser.add_argument('--sampler-gpus', type=int, default=1,
                        help='Number of GPUs for sampler (inference)')
    args = parser.parse_args()

    log(f"Starting standalone weight sync test...")
    log(f"  Model GPUs:   {args.model_gpus}")
    log(f"  Sampler GPUs: {args.sampler_gpus}")
    log(f"  Model ID:     {MODEL_ID}")

    try:
        success = test_standalone_weight_sync(args.model_gpus, args.sampler_gpus)
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
