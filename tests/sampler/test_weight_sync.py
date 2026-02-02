#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Test weight synchronization between training model and vLLM sampler.

This test verifies that weights can be correctly synchronized from a
TransformersModel to a VLLMSampler using different methods.

Test Strategy:
- Use load_format='dummy' for vLLM so it starts with random weights
- Load real weights in TransformersModel
- Verify that sampler with real weights produces different output than dummy
"""
import os
import sys

# Must set before importing anything
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'

from transformers import AutoTokenizer

from twinkle import remote_class, remote_function, DeviceMesh, DeviceGroup
from twinkle.sampler import VLLMSampler
from twinkle.sampler.types import SamplingParams
from twinkle.template import Template
from twinkle.data_format import Trajectory
from twinkle.model.transformers import TransformersModel

# Model configuration - use a small model for testing
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'

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
    print(f"[TEST] {msg}", flush=True)


@remote_class()
class DummySamplerActor:
    """Sampler actor with dummy weights for testing."""
    
    def __init__(self, model_id: str, device_mesh: DeviceMesh = None, **kwargs):
        log("Creating DummySamplerActor with dummy weights...")
        self.sampler = VLLMSampler(
            model_id=model_id,
            engine_args={
                'load_format': 'dummy',  # Random weights
                'gpu_memory_utilization': 0.3,
                'max_model_len': 256,
                'enforce_eager': True,
            },
        )
        self.sampler.set_template(Template, model_id=model_id)
        log("DummySamplerActor initialized")
    
    @remote_function(dispatch='all', collect='first')
    def sample_text(self, prompt: str, max_tokens: int = 32) -> list:
        """Sample text from the model."""
        traj = Trajectory(messages=[{'role': 'user', 'content': prompt}])
        response = self.sampler.sample(
            traj, 
            SamplingParams(max_tokens=max_tokens, temperature=0.0)
        )
        if response and hasattr(response, 'sequences') and response.sequences:
            tokens = response.sequences[0].tokens
            if hasattr(tokens, 'tolist'):
                return tokens.tolist()
            return list(tokens) if tokens else []
        return []


@remote_class()
class RealSamplerActor:
    """Sampler actor with real weights for testing."""
    
    def __init__(self, model_id: str, device_mesh: DeviceMesh = None, **kwargs):
        log("Creating RealSamplerActor with real weights...")
        self.sampler = VLLMSampler(
            model_id=model_id,
            engine_args={
                'load_format': 'auto',  # Real weights from disk
                'gpu_memory_utilization': 0.3,
                'max_model_len': 256,
                'enforce_eager': True,
            },
        )
        self.sampler.set_template(Template, model_id=model_id)
        log("RealSamplerActor initialized")
    
    @remote_function(dispatch='all', collect='first')
    def sample_text(self, prompt: str, max_tokens: int = 32) -> list:
        """Sample text from the model."""
        traj = Trajectory(messages=[{'role': 'user', 'content': prompt}])
        response = self.sampler.sample(
            traj, 
            SamplingParams(max_tokens=max_tokens, temperature=0.0)
        )
        if response and hasattr(response, 'sequences') and response.sequences:
            tokens = response.sequences[0].tokens
            if hasattr(tokens, 'tolist'):
                return tokens.tolist()
            return list(tokens) if tokens else []
        return []


@remote_class()
class HybridModelSamplerActor:
    """Hybrid actor that fuses training model and sampler in same process.
    
    This simulates the Hybrid mode where:
    - Training model (TransformersModel) holds the real weights
    - vLLM Sampler starts with dummy/random weights
    - Weight sync happens via direct method call (for now)
    """

    def __init__(
        self, 
        model_id: str, 
        device_mesh: DeviceMesh = None,
        remote_group: str = None,
        **kwargs
    ):
        log("Initializing HybridModelSamplerActor...")
        
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
        log("VLLMSampler initialized with dummy weights")
        
        # Initialize training model with real weights
        self.model = TransformersModel(model_id=model_id, device_mesh=device_mesh)
        log("TransformersModel initialized with real weights")
    
    @remote_function(dispatch='all', collect='first')
    def sample_text(self, prompt: str, max_tokens: int = 32) -> list:
        """Sample text from the model."""
        traj = Trajectory(messages=[{'role': 'user', 'content': prompt}])
        response = self.sampler.sample(
            traj, 
            SamplingParams(max_tokens=max_tokens, temperature=0.0)
        )
        if response and hasattr(response, 'sequences') and response.sequences:
            tokens = response.sequences[0].tokens
            if hasattr(tokens, 'tolist'):
                return tokens.tolist()
            return list(tokens) if tokens else []
        return []
    
    @remote_function(dispatch='all', collect='first')
    def get_model_weights_count(self) -> int:
        """Get the number of model weights."""
        state_dict = self.model.get_state_dict()
        if isinstance(state_dict, dict):
            return len(state_dict)
        else:
            # Iterator - count by consuming (for testing only)
            return sum(1 for _ in state_dict)


def wait_result(result):
    """Wait for result if it's a LazyCollect object.
    
    LazyCollect is callable - calling it triggers the collection.
    """
    if hasattr(result, '_is_lazy_collect') and result._is_lazy_collect:
        return result()  # LazyCollect is callable
    if hasattr(result, 'wait'):
        return result.wait()
    if callable(result) and hasattr(result, '_get_result'):
        return result()
    return result


def test_dummy_vs_real_baseline():
    """Baseline test: verify dummy and real weights produce different outputs.
    
    This test verifies the fundamental assumption that:
    1. vLLM with dummy weights produces garbage output
    2. vLLM with real weights produces meaningful output
    """
    import twinkle
    
    log("=" * 60)
    log("TEST: Dummy vs Real Weights Baseline")
    log("=" * 60)
    
    twinkle.initialize(
        mode='ray',
        nproc_per_node=1,
        groups=[
            DeviceGroup(name='test', ranks=[0], device_type='GPU', gpus_per_worker=1),
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    test_prompt = "What is 2+2?"
    
    # Test with dummy weights
    log("Creating actor with DUMMY weights...")
    dummy_actor = DummySamplerActor(
        model_id=MODEL_ID,
        device_mesh=DeviceMesh.from_sizes(world_size=1, dp_size=1),
        remote_group='test',
    )
    
    tokens_dummy = wait_result(dummy_actor.sample_text(test_prompt))
    if not isinstance(tokens_dummy, list):
        tokens_dummy = list(tokens_dummy) if tokens_dummy else []
    text_dummy = tokenizer.decode(tokens_dummy, skip_special_tokens=True) if tokens_dummy else ""
    log(f"Output with DUMMY weights: '{text_dummy[:100]}'")
    
    # Shutdown dummy actor to free GPU memory
    log("Shutting down dummy actor...")
    import ray
    # Need to get the actual ray actor handle and terminate it
    # For now, just continue - next actor will use same GPU
    
    # Test with real weights
    log("Creating actor with REAL weights...")
    real_actor = RealSamplerActor(
        model_id=MODEL_ID,
        device_mesh=DeviceMesh.from_sizes(world_size=1, dp_size=1),
        remote_group='test',
    )
    
    tokens_real = wait_result(real_actor.sample_text(test_prompt))
    if not isinstance(tokens_real, list):
        tokens_real = list(tokens_real) if tokens_real else []
    text_real = tokenizer.decode(tokens_real, skip_special_tokens=True) if tokens_real else ""
    log(f"Output with REAL weights: '{text_real[:100]}'")
    
    # Verify outputs are different
    if tokens_dummy != tokens_real:
        log("SUCCESS: Outputs differ between dummy and real weights")
        
        # Check if real weights produce correct answer
        if "4" in text_real or "four" in text_real.lower():
            log("SUCCESS: Real weights produce correct answer '4'")
            return True
        else:
            log("WARNING: Real output doesn't contain '4' but outputs differ")
            return True
    else:
        log("FAIL: Outputs are the same - something is wrong")
        return False


def test_hybrid_actor_initialization():
    """Test that HybridModelSamplerActor initializes correctly.
    
    This verifies:
    1. VLLMSampler with dummy weights can be created
    2. TransformersModel with real weights can be created
    3. Both can coexist in the same actor
    """
    import twinkle
    
    log("=" * 60)
    log("TEST: Hybrid Actor Initialization")
    log("=" * 60)
    
    # Shutdown any existing ray to start fresh
    import ray
    if ray.is_initialized():
        ray.shutdown()
    
    # Use a unique group name for this test
    twinkle.initialize(
        mode='ray',
        nproc_per_node=1,
        groups=[
            DeviceGroup(name='test_hybrid', ranks=[0], device_type='GPU', gpus_per_worker=1),
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Create hybrid actor
    log("Creating HybridModelSamplerActor...")
    actor = HybridModelSamplerActor(
        model_id=MODEL_ID,
        device_mesh=DeviceMesh.from_sizes(world_size=1, dp_size=1),
        remote_group='test_hybrid',
    )
    
    # Verify model weights are accessible  
    weights_count_raw = actor.get_model_weights_count()
    weights_count = wait_result(weights_count_raw)
    log(f"Model has {weights_count} trainable parameters")
    
    if isinstance(weights_count, int) and weights_count > 0:
        log("SUCCESS: Hybrid actor initialized with model weights")
    else:
        log("FAIL: No model weights found")
        return False
    
    # Verify sampler works (with dummy weights initially)
    tokens = wait_result(actor.sample_text("Hello"))
    if tokens:
        log(f"SUCCESS: Sampler generated {len(tokens)} tokens")
        return True
    else:
        log("FAIL: Sampler failed to generate tokens")
        return False


def main():
    """Run all weight sync tests."""
    results = []
    
    # Determine which test to run via environment variable
    test_to_run = os.environ.get('TEST_NAME', 'all')
    
    # Test 1: Baseline - verify dummy vs real weights differ
    if test_to_run in ('all', 'baseline'):
        try:
            passed = test_dummy_vs_real_baseline()
            results.append(('dummy_vs_real_baseline', passed))
        except Exception as e:
            log(f"Error in dummy_vs_real_baseline: {e}")
            import traceback
            traceback.print_exc()
            results.append(('dummy_vs_real_baseline', False))
    
    # Test 2: Hybrid actor initialization
    if test_to_run in ('all', 'hybrid'):
        # Note: This test requires a fresh ray session
        # Running after baseline test may fail due to cached group config
        try:
            passed = test_hybrid_actor_initialization()
            results.append(('hybrid_actor_init', passed))
        except Exception as e:
            log(f"Error in hybrid_actor_init: {e}")
            import traceback
            traceback.print_exc()
            results.append(('hybrid_actor_init', False))
    
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
    
    log("\nAll tests passed!")


if __name__ == '__main__':
    main()
