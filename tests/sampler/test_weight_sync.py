#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
import time

# Must set before importing anything
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
from transformers import AutoTokenizer

# Model configuration
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
from twinkle import remote_class, DeviceMesh, DeviceGroup
from twinkle.sampler import VLLMSampler
from twinkle.sampler.types import SamplingParams
from twinkle.template import Template
from twinkle.data_format import Trajectory
from twinkle.weight_loader import WeightLoader
from twinkle.model.transformers import TransformersModel, MegatronModel
# Resolve to local cache
try:
    from modelscope.hub.snapshot_download import snapshot_download
    _cache = snapshot_download(MODEL_ID, local_files_only=True)
    if _cache:
        MODEL_ID = _cache
except:
    pass


def log(msg):
    """Print message with timestamp."""
    print(f"[TEST] {msg}", flush=True)

@remote_class
class HybridModelSamplerActor:

    def __init__(
        self, 
        model_id: str, 
        device_mesh: DeviceMesh = None,
        remote_group:str = None,
        **kwargs):
        self.sampler = VLLMSampler(
            model_id=model_id,
            engine_args={
                'load_format': 'dummy',
                'gpu_memory_utilization': 0.3,
                'max_model_len': 256,
                'enforce_eager': True,
                'enable_sleep_mode': True,
            },
        )
        self.sampler.set_template(Template, model_id=model_id)
        self.model = TransformersModel(model_id=model_id)
        self.weight_loader = IPCWeightLoader(self.model, self.sampler) # TODO

    def sync_weights(self):
        self.weight_loader.load_weights()

def test_hybrid_weight_sync():
    import twinkle
    twinkle.initialize(
        mode='ray',
        nproc_per_node=8,
        groups=[
            DeviceGroup(name='hybrid', ranks=[0,1,2,3,4,5,6,7], gpus_per_worker=8),
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model_sampler_actor = HybridModelSamplerActor(model_id=MODEL_ID, device_mesh=DeviceMesh.from_sizes(world_size=8, dp_size=4))
    # test sample before weight sync
    response_before = model_sampler_actor.sampler.sample(Trajectory(messages=[{'role': 'user', 'content': 'What is 2+2?'}]), SamplingParams(max_tokens=32, temperature=0.0))
    text_before = tokenizer.decode(response_before.sequences[0].tokens, skip_special_tokens=True)
    log(f"Output with random weights: {text_before[:100]}")
    # test weight sync
    model_sampler_actor.sync_weights()
    # test sample after weight sync
    response_after = model_sampler_actor.sampler.sample(Trajectory(messages=[{'role': 'user', 'content': 'What is 2+2?'}]), SamplingParams(max_tokens=32, temperature=0.0))
    text_after = tokenizer.decode(response_after.sequences[0].tokens, skip_special_tokens=True)
    log(f"Output with synced weights: {text_after[:100]}")
    

def main():    
    results = []
    try:
        test_hybrid_weight_sync()
    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(('hybrid_weight_sync', False))

    # Summary
    log("\n" + "=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        log(f"  {name}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    log(f"\nTotal: {passed_count}/{len(results)} passed")
    
    if passed_count != len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
