#!/usr/bin/env python
"""Test weight sync with Qwen3-30B-A3B-Base (MoE ~30B params).

Verifies:
  1. Streaming weight sync does NOT OOM on rollout GPUs.
  2. vllm_tp > 1 does NOT hang during sync.

Usage:
    # Test: 2 model GPUs + 2 sampler GPUs, TP=2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/sampler/test_30b_weight_sync.py \
        --model-gpus 2 --sampler-gpus 2 --vllm-tp 2

    # Test: 4 model GPUs + 4 sampler GPUs, TP=1
    python tests/sampler/test_30b_weight_sync.py \
        --model-gpus 4 --sampler-gpus 4 --vllm-tp 1
"""
import argparse
import datetime
import os
import pytest
import sys
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'
os.environ['NCCL_CUMEM_ENABLE'] = '0'

MODEL_ID = os.environ.get('TEST_MODEL_ID', 'Qwen/Qwen3-30B-A3B-Base')

# For MoE models, vLLM does not support LoRA on expert layers.
# Only target attention QKV + output projection.
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']


def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def get_model_path():
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        _cache = snapshot_download(MODEL_ID, local_files_only=True)
        if _cache:
            return _cache
    except Exception:
        pass
    return MODEL_ID


@pytest.mark.skip(reason='Requires 4+ GPUs and 30B model, run manually: python tests/sampler/test_30b_weight_sync.py')
def test_weight_sync(model_gpus: int = 2, sampler_gpus: int = 1, vllm_tp: int = 1):
    from peft import LoraConfig

    import twinkle
    from twinkle import DeviceGroup, DeviceMesh
    from twinkle.checkpoint_engine import CheckpointEngineManager
    from twinkle.data_format import Trajectory
    from twinkle.data_format.sampling import SamplingParams
    from twinkle.model.transformers import TransformersModel
    from twinkle.sampler import vLLMSampler
    from twinkle.template import Template

    total_gpus = model_gpus + sampler_gpus
    n_sampler_actors = sampler_gpus // vllm_tp
    model_path = get_model_path()

    log('=' * 70)
    log(f'TEST: Weight Sync with {MODEL_ID}')
    log(f'  Model GPUs    : {model_gpus}')
    log(f'  Sampler GPUs  : {sampler_gpus} (vllm_tp={vllm_tp}, actors={n_sampler_actors})')
    log(f'  LoRA targets  : {LORA_TARGET_MODULES}')
    log(f'  Model path    : {model_path}')
    log('=' * 70)

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
                gpus_per_worker=vllm_tp,
            ),
        ],
    )

    # Model — FSDP across model_gpus
    model_mesh = DeviceMesh.from_sizes(world_size=model_gpus, dp_size=model_gpus)
    model = TransformersModel(
        model_id=model_path,
        device_mesh=model_mesh,
        remote_group='model',
    )

    # Add LoRA — only attention layers, not expert MLP
    lora_config = LoraConfig(
        target_modules=LORA_TARGET_MODULES,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)

    # Sampler — Twinkle sees n_sampler_actors workers, not total GPUs.
    # vLLM TP is internal to each actor.
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=n_sampler_actors,
        dp_size=n_sampler_actors,
    )
    sampler = vLLMSampler(
        model_id=model_path,
        engine_args={
            'load_format': 'dummy',
            'gpu_memory_utilization': 0.8,
            'max_model_len': 256,
            'enforce_eager': True,
            'enable_sleep_mode': False,
            'tensor_parallel_size': vllm_tp,
            'max_loras': 1,
            'enable_lora': True,  # vLLM LoRA + MoE + TP>1 has a bug in dummy run
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=model_path)

    log('Waiting for vLLM initialization...')
    time.sleep(5)

    # Print GPU memory before sync
    log('\n--- GPU memory BEFORE weight sync ---')
    os.system('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader')

    # Weight sync
    log('\n--- Starting weight sync ---')
    manager = CheckpointEngineManager(model=model, sampler=sampler)

    # Base model sync
    sync_start = time.time()
    manager.sync_weights()
    # lora
    manager.sync_weights()
    base_time = time.time() - sync_start
    log(f'  Base weight sync completed in {base_time:.2f}s')

    # Print GPU memory after base sync
    log('\n--- GPU memory AFTER base sync ---')
    os.system('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader')

    sampler.reset_prefix_cache()
    lora_time = 0.0

    # Quick sample to verify model works
    log('\n--- Sampling after sync ---')
    traj = Trajectory(messages=[{'role': 'user', 'content': 'What is 2+2?'}])
    response = sampler.sample(traj, SamplingParams(max_tokens=32, temperature=0.0))
    if callable(response):
        response = response()
    if response and response.sequences:
        tokens = response.sequences[0].tokens
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        from modelscope import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        text = tok.decode(tokens, skip_special_tokens=True)
        log(f"  Output: '{text[:200]}'")

    log('\n--- PASS: Weight sync completed without OOM or hang ---')
    log(f'  Base sync: {base_time:.2f}s, LoRA sync: {lora_time:.2f}s')
    sampler.shutdown()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-gpus', type=int, default=2)
    parser.add_argument('--sampler-gpus', type=int, default=1)
    parser.add_argument('--vllm-tp', type=int, default=1)
    args = parser.parse_args()

    log(f'Test config: model_gpus={args.model_gpus}, sampler_gpus={args.sampler_gpus}, vllm_tp={args.vllm_tp}')

    try:
        success = test_weight_sync(args.model_gpus, args.sampler_gpus, args.vllm_tp)
    except Exception as e:
        log(f'\nTest FAILED with exception: {e}')
        import traceback
        traceback.print_exc()
        success = False

    log(f"\nRESULT: {'PASS' if success else 'FAIL'}")
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
