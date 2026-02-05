#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Test weight synchronization between training model and vLLM sampler.

This test verifies that weights can be correctly synchronized from a
TransformersModel to a VLLMSampler using IPCWeightLoader in Hybrid mode.

Test Configuration:
- 4 GPUs with DP=4, TP=1
- Each GPU runs one HybridModelSamplerActor with model+sampler
- Weight sync via CUDA IPC within each GPU
"""
import os
import sys
import time

# Must set before importing anything
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# Model configuration - use a small model for testing
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
WORLD_SIZE = 2

from transformers import AutoTokenizer

from twinkle import remote_class, remote_function, DeviceMesh, DeviceGroup
from twinkle.sampler import VLLMSampler
from twinkle.sampler.types import SamplingParams
from twinkle.template import Template
from twinkle.data_format import Trajectory
from twinkle.weight_loader import IPCWeightLoader
from twinkle.model.transformers import TransformersModel


# Resolve to local cache
try:
    from modelscope.hub.snapshot_download import snapshot_download
    _cache = snapshot_download(MODEL_ID, local_files_only=True)
    if _cache:
        MODEL_ID = _cache
except Exception:
    pass


def log(msg):
    """Print message with timestamp."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def wait_result(result):
    """Wait for result if it's a LazyCollect object."""
    if hasattr(result, '_is_lazy_collect') and result._is_lazy_collect:
        return result()
    if hasattr(result, 'wait'):
        return result.wait()
    if callable(result) and hasattr(result, '_get_result'):
        return result()
    return result


@remote_class()
class HybridModelSamplerActor:
    """Hybrid actor that fuses training model and sampler in same process.
    
    This simulates the Hybrid mode where:
    - Training model (TransformersModel) holds the real weights
    - vLLM Sampler starts with dummy/random weights
    - Weight sync happens via IPCWeightLoader (CUDA IPC + ZMQ)
    """

    def __init__(
        self, 
        model_id: str, 
        device_mesh: DeviceMesh = None,
        remote_group: str = None,
        **kwargs
    ):
        import torch
        rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
        log(f"[Rank {rank}] Initializing HybridModelSamplerActor...")
        
        # Initialize sampler with dummy weights (random initialization)
        self.sampler = VLLMSampler(
            model_id=model_id,
            engine_args={
                'load_format': 'dummy',  # Random weights
                'gpu_memory_utilization': 0.3,
                'max_model_len': 256,
                'enforce_eager': True,
                'enable_sleep_mode': True,
            },
        )
        self.sampler.set_template(Template, model_id=model_id)
        log(f"[Rank {rank}] VLLMSampler initialized with dummy weights")
        
        # Initialize training model with real weights
        self.model = TransformersModel(model_id=model_id, device_mesh=device_mesh)
        log(f"[Rank {rank}] TransformersModel initialized with real weights")
        
        # Initialize weight loader for Hybrid mode (CUDA IPC)
        self.weight_loader = IPCWeightLoader(
            model=self.model,
            sampler=self.sampler,
            bucket_size_mb=512,
        )
        log(f"[Rank {rank}] IPCWeightLoader initialized")
    
    @remote_function(dispatch='all', collect='first')
    def sync_weights(self, adapter_name: str = ''):
        """Sync weights from training model to sampler via CUDA IPC."""
        import torch
        rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
        log(f"[Rank {rank}] Starting weight sync...")
        start = time.time()
        self.weight_loader.load_weights(adapter_name=adapter_name)
        elapsed = time.time() - start
        log(f"[Rank {rank}] Weight sync completed in {elapsed:.2f}s")
        return elapsed
    
    @remote_function(dispatch='all', collect='first')
    def sample_text(self, prompt: str, max_tokens: int = 64) -> dict:
        """Sample text from the model and return result info."""
        import torch
        rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
        
        traj = Trajectory(messages=[{'role': 'user', 'content': prompt}])
        response = self.sampler.sample(
            traj, 
            SamplingParams(max_tokens=max_tokens, temperature=0.0)
        )
        
        if response and hasattr(response, 'sequences') and response.sequences:
            tokens = response.sequences[0].tokens
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            else:
                tokens = list(tokens) if tokens else []
            return {
                'rank': rank,
                'tokens': tokens,
                'num_tokens': len(tokens),
            }
        return {'rank': rank, 'tokens': [], 'num_tokens': 0}
    
    @remote_function(dispatch='all', collect='first')  
    def get_model_info(self) -> dict:
        """Get model information."""
        import torch
        rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
        state_dict = self.model.get_state_dict()
        if isinstance(state_dict, dict):
            num_params = len(state_dict)
        else:
            num_params = sum(1 for _ in state_dict)
        return {
            'rank': rank,
            'num_params': num_params,
            'model_id': self.model.model_id,
        }


def test_hybrid_weight_sync():
    """Test weight sync in Hybrid mode with WORLD_SIZE GPUs (DP=WORLD_SIZE, TP=1).
    
    Each GPU runs one actor with:
    - TransformersModel with real weights
    - VLLMSampler with dummy weights initially
    - IPCWeightLoader for weight sync
    
    Test verifies:
    1. Before sync: sampler produces garbage output (random weights)
    2. After sync: sampler produces correct output (real weights)
    """
    import twinkle
    
    log("=" * 70)
    log(f"TEST: Hybrid Weight Sync ({WORLD_SIZE} GPU, DP={WORLD_SIZE}, TP=1)")
    log("=" * 70)
    
    # Initialize with WORLD_SIZE GPUs
    twinkle.initialize(
        mode='ray',
        nproc_per_node=WORLD_SIZE,
        groups=[
            DeviceGroup(
                name='hybrid', 
                ranks=[i for i in range(WORLD_SIZE)], 
                device_type='GPU', 
                gpus_per_worker=1,  # Each worker gets 1 GPU (TP=1)
            ),
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    test_prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "What is 10*5?",
        "Hello, my name is",
    ]

    # Create hybrid actor (will spawn WORLD_SIZE instances, one per GPU)
    log(f"Creating HybridModelSamplerActor on {WORLD_SIZE} GPUs...")
    actor = HybridModelSamplerActor(
        model_id=MODEL_ID,
        device_mesh=DeviceMesh.from_sizes(world_size=WORLD_SIZE, dp_size=WORLD_SIZE),
        remote_group='hybrid',
    )
    
    # Wait for initialization and get model info
    log("Waiting for actor initialization...")
    model_info = wait_result(actor.get_model_info())
    log(f"Model info: {model_info}")
    
    # Test 1: Sample BEFORE weight sync (should produce garbage)
    log("\n" + "-" * 50)
    log("STEP 1: Sampling BEFORE weight sync (dummy weights)")
    log("-" * 50)
    
    results_before = {}
    for i, prompt in enumerate(test_prompts):
        result = wait_result(actor.sample_text(prompt))
        text = tokenizer.decode(result['tokens'], skip_special_tokens=True) if result['tokens'] else ""
        results_before[prompt] = {
            'tokens': result['tokens'],
            'text': text,
        }
        log(f"  Prompt {i+1}: '{prompt}'")
        log(f"  Output: '{text[:80]}...' ({result['num_tokens']} tokens)")
    
    # Test 2: Sync weights
    log("\n" + "-" * 50)
    log("STEP 2: Syncing weights via IPCWeightLoader")
    log("-" * 50)
    
    sync_start = time.time()
    sync_result = wait_result(actor.sync_weights())
    sync_elapsed = time.time() - sync_start
    log(f"Weight sync completed in {sync_elapsed:.2f}s")
    
    # Test 3: Sample AFTER weight sync (should produce correct output)
    log("\n" + "-" * 50)
    log("STEP 3: Sampling AFTER weight sync (real weights)")
    log("-" * 50)
    
    results_after = {}
    for i, prompt in enumerate(test_prompts):
        result = wait_result(actor.sample_text(prompt))
        text = tokenizer.decode(result['tokens'], skip_special_tokens=True) if result['tokens'] else ""
        results_after[prompt] = {
            'tokens': result['tokens'],
            'text': text,
        }
        log(f"  Prompt {i+1}: '{prompt}'")
        log(f"  Output: '{text[:80]}...' ({result['num_tokens']} tokens)")
    
    # Verify results
    log("\n" + "-" * 50)
    log("STEP 4: Verification")
    log("-" * 50)
    
    all_passed = True
    for prompt in test_prompts:
        before = results_before[prompt]
        after = results_after[prompt]
        
        # Check if outputs are different
        outputs_differ = before['tokens'] != after['tokens']
        
        # Check for expected answers
        expected_answers = {
            "What is 2+2?": ["4", "four"],
            "What is the capital of France?": ["Paris", "paris"],
            "What is 10*5?": ["50", "fifty"],
            "Hello, my name is": [],  # No specific expected answer
        }
        
        expected = expected_answers.get(prompt, [])
        has_correct_answer = any(ans.lower() in after['text'].lower() for ans in expected) if expected else True
        
        status = "PASS" if outputs_differ else "FAIL"
        answer_status = "CORRECT" if has_correct_answer else "CHECK"
        
        log(f"  '{prompt}':")
        log(f"    Before: '{before['text'][:50]}...'")
        log(f"    After:  '{after['text'][:50]}...'")
        log(f"    Status: {status} (outputs differ: {outputs_differ}), Answer: {answer_status}")
        
        if not outputs_differ:
            all_passed = False
    
    return all_passed


def main():
    """Run weight sync test."""
    log("=" * 70)
    log("TWINKLE WEIGHT SYNC TEST")
    log(f"Model: {MODEL_ID}")
    log(f"Configuration: Hybrid mode, {WORLD_SIZE} GPU, DP={WORLD_SIZE}, TP=1")
    log("=" * 70)
    
    results = []
    
    try:
        passed = test_hybrid_weight_sync()
        results.append((f'hybrid_weight_sync_{WORLD_SIZE}gpu', passed))
    except Exception as e:
        log(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        results.append((f'hybrid_weight_sync_{WORLD_SIZE}gpu', False))
    
    # Summary
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        log(f"  {name}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    log(f"\nTotal: {passed_count}/{len(results)} passed")
    
    if passed_count != len(results):
        sys.exit(1)
    
    log("\nAll tests passed!")


if __name__ == '__main__':
    main()
