# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for VLLMEngine and related components."""

import pytest
import time

from twinkle.sampler.vllm_sampler.vllm_engine import (
    LoRAAdapter,
    LoRAAdapterManager,
    get_vllm_max_lora_rank,
)
from twinkle.data_format.sampling import SamplingParams, SampleResponse, SampledSequence

# =============================================================================
# Tests for Data Types
# =============================================================================

class TestSamplingParams:
    """Tests for SamplingParams dataclass."""
    
    def test_default_values(self):
        """Test default parameter values."""
        params = SamplingParams()
        assert params.max_tokens is None
        assert params.seed is None
        assert params.stop is None
        assert params.temperature == 1.0
        assert params.top_k == -1
        assert params.top_p == 1.0
        assert params.repetition_penalty == 1.0
    
    def test_custom_values(self):
        """Test custom parameter values."""
        params = SamplingParams(
            max_tokens=256,
            seed=42,
            stop=["<|endoftext|>"],
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
        assert params.max_tokens == 256
        assert params.seed == 42
        assert params.stop == ["<|endoftext|>"]
        assert params.temperature == 0.7
        assert params.top_k == 50
        assert params.top_p == 0.9
    
    @pytest.mark.skipif(True, reason="Requires vLLM installation")
    def test_to_vllm_params(self):
        """Test conversion to vLLM SamplingParams."""
        params = SamplingParams(
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
        )
        vllm_params = params.to_vllm_params(logprobs=True, n=4)
        
        assert vllm_params.max_tokens == 100
        assert vllm_params.temperature == 0.8
        assert vllm_params.top_p == 0.95
        assert vllm_params.n == 4
        assert vllm_params.logprobs == 0  # 0 means only sampled token
    
    @pytest.mark.skipif(True, reason="Requires vLLM installation")
    def test_stop_as_string(self):
        """Test stop sequence as a single string."""
        params = SamplingParams(stop="<|end|>")
        vllm_params = params.to_vllm_params()
        assert vllm_params.stop == ["<|end|>"]
    
    @pytest.mark.skipif(True, reason="Requires vLLM installation")
    def test_stop_as_token_ids(self):
        """Test stop as token IDs (list of ints)."""
        params = SamplingParams(stop=[128001, 128002])
        vllm_params = params.to_vllm_params()
        assert vllm_params.stop_token_ids == [128001, 128002]


class TestSampledSequence:
    """Tests for SampledSequence dataclass."""
    
    def test_basic_sequence(self):
        """Test basic sequence creation."""
        seq = SampledSequence(
            stop_reason="stop",
            tokens=[1, 2, 3, 4, 5],
        )
        assert seq.stop_reason == "stop"
        assert seq.tokens == [1, 2, 3, 4, 5]
        assert seq.logprobs is None
    
    def test_sequence_with_logprobs(self):
        """Test sequence with log probabilities."""
        seq = SampledSequence(
            stop_reason="length",
            tokens=[10, 20, 30],
            logprobs=[-1.5, -2.0, -0.5],
        )
        assert seq.stop_reason == "length"
        assert seq.tokens == [10, 20, 30]
        assert seq.logprobs == [-1.5, -2.0, -0.5]


class TestSampleResponse:
    """Tests for SampleResponse dataclass."""
    
    def test_basic_response(self):
        """Test basic response creation."""
        seq = SampledSequence(stop_reason="stop", tokens=[1, 2, 3])
        response = SampleResponse(sequences=[seq])
        
        assert len(response.sequences) == 1
        assert response.sequences[0].tokens == [1, 2, 3]
        assert response.prompt_logprobs is None
        assert response.topk_prompt_logprobs is None
    
    def test_response_with_prompt_logprobs(self):
        """Test response with prompt log probabilities."""
        seq = SampledSequence(stop_reason="stop", tokens=[1, 2])
        response = SampleResponse(
            sequences=[seq],
            prompt_logprobs=[None, -0.1, -0.2, -0.3],
            topk_prompt_logprobs=[
                None,
                [(100, -0.1), (200, -0.5)],
                [(101, -0.2), (201, -0.6)],
                [(102, -0.3), (202, -0.7)],
            ],
        )
        
        assert response.prompt_logprobs[1] == -0.1
        assert response.topk_prompt_logprobs[1] == [(100, -0.1), (200, -0.5)]


# =============================================================================
# Tests for LoRAAdapterManager
# =============================================================================

class TestLoRAAdapterManager:
    """Tests for LoRAAdapterManager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = LoRAAdapterManager("test-model", max_loras=32)
        assert manager.model_id == "test-model"
        assert manager.max_loras == 32
        assert len(manager.list_adapters()) == 0
    
    def test_add_adapter(self):
        """Test adding an adapter."""
        manager = LoRAAdapterManager("test-model")
        peft_config = {"r": 16, "lora_alpha": 32}
        
        new_adapter, old_adapter = manager.add_adapter(
            user_id="user1",
            peft_config=peft_config,
            lora_path="/tmp/lora1",
        )
        
        assert old_adapter is None
        assert new_adapter.user_id == "user1"
        assert new_adapter.peft_config == peft_config
        assert new_adapter.lora_path == "/tmp/lora1"
        assert len(manager.list_adapters()) == 1
    
    def test_replace_adapter(self):
        """Test replacing an adapter for the same user."""
        manager = LoRAAdapterManager("test-model")
        
        # Add first adapter
        adapter1, _ = manager.add_adapter(
            user_id="user1",
            peft_config={"r": 16},
            lora_path="/tmp/lora1",
        )
        first_id = adapter1.lora_int_id
        
        # Replace with new adapter
        adapter2, old_adapter = manager.add_adapter(
            user_id="user1",
            peft_config={"r": 32},
            lora_path="/tmp/lora2",
        )
        
        assert old_adapter is not None
        assert old_adapter.lora_int_id == first_id
        assert adapter2.lora_int_id != first_id
        assert adapter2.peft_config == {"r": 32}
        assert len(manager.list_adapters()) == 1
    
    def test_get_adapter_by_uri(self):
        """Test getting adapter by URI."""
        manager = LoRAAdapterManager("test-model")
        adapter_new, _ = manager.add_adapter(
            user_id="user1",
            peft_config={},
            lora_path="/tmp/lora1",
        )
        
        uri = manager.get_uri("user1")
        # URI includes version (lora_int_id)
        assert uri == f"twinkle://test-model/lora/user1/{adapter_new.lora_int_id}"
        
        adapter = manager.get_adapter_by_uri(uri)
        assert adapter is not None
        assert adapter.user_id == "user1"
        
        # Invalid URI
        assert manager.get_adapter_by_uri("invalid://uri") is None
    
    def test_remove_adapter(self):
        """Test removing an adapter."""
        manager = LoRAAdapterManager("test-model")
        manager.add_adapter(
            user_id="user1",
            peft_config={},
            lora_path="/tmp/lora1",
        )
        
        removed = manager.remove_adapter("user1")
        assert removed is not None
        assert removed.user_id == "user1"
        assert len(manager.list_adapters()) == 0
        
        # Try to remove again
        removed_again = manager.remove_adapter("user1")
        assert removed_again is None
    
    def test_multiple_users(self):
        """Test multiple users with different adapters."""
        manager = LoRAAdapterManager("test-model")
        
        manager.add_adapter("user1", {"r": 16}, "/tmp/lora1")
        manager.add_adapter("user2", {"r": 32}, "/tmp/lora2")
        manager.add_adapter("user3", {"r": 64}, "/tmp/lora3")
        
        assert len(manager.list_adapters()) == 3
        
        adapter1 = manager.get_adapter("user1")
        adapter2 = manager.get_adapter("user2")
        adapter3 = manager.get_adapter("user3")
        
        assert adapter1.peft_config == {"r": 16}
        assert adapter2.peft_config == {"r": 32}
        assert adapter3.peft_config == {"r": 64}
        
        # All have different IDs
        ids = {adapter1.lora_int_id, adapter2.lora_int_id, adapter3.lora_int_id}
        assert len(ids) == 3
    
    def test_lora_id_generation(self):
        """Test that LoRA IDs are unique and incrementing."""
        manager = LoRAAdapterManager("test-model")
        
        adapters = []
        for i in range(10):
            adapter, _ = manager.add_adapter(f"user{i}", {}, f"/tmp/lora{i}")
            adapters.append(adapter)
        
        ids = [a.lora_int_id for a in adapters]
        assert len(ids) == len(set(ids)), "IDs should be unique"
        assert ids == sorted(ids), "IDs should be incrementing"


# =============================================================================
# Tests for Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    @pytest.mark.skipif(True, reason="Requires vLLM installation")
    def test_get_vllm_max_lora_rank(self):
        """Test LoRA rank normalization."""
        # Should round up to nearest allowed rank
        assert get_vllm_max_lora_rank(8) in [8, 16]
        assert get_vllm_max_lora_rank(16) in [16, 32]
        assert get_vllm_max_lora_rank(33) in [64, 128]


# =============================================================================
# Tests for LoRAAdapter
# =============================================================================

class TestLoRAAdapter:
    """Tests for LoRAAdapter dataclass."""
    
    def test_adapter_creation(self):
        """Test adapter creation."""
        adapter = LoRAAdapter(
            lora_int_id=123,
            lora_name="test_lora",
            lora_path="/tmp/lora",
            peft_config={"r": 16, "lora_alpha": 32},
            user_id="user1",
            created_at=time.time(),
        )
        
        assert adapter.lora_int_id == 123
        assert adapter.lora_name == "test_lora"
        assert adapter.user_id == "user1"
    
    @pytest.mark.skipif(True, reason="Requires vLLM installation")
    def test_to_lora_request(self):
        """Test conversion to vLLM LoRARequest."""
        adapter = LoRAAdapter(
            lora_int_id=123,
            lora_name="test_lora",
            lora_path="/tmp/lora",
            peft_config={},
            user_id="user1",
            created_at=time.time(),
        )
        
        lora_request = adapter.to_lora_request()
        assert lora_request.lora_name == "test_lora"
        assert lora_request.lora_int_id == 123
        assert lora_request.lora_path == "/tmp/lora"


# =============================================================================
# Integration Tests (requires vLLM)
# =============================================================================

@pytest.mark.skip(reason="Requires vLLM and GPU")
class TestVLLMEngineIntegration:
    """Integration tests for VLLMEngine (requires vLLM installation)."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initialization."""
        from twinkle.sampler import VLLMEngine
        
        engine = VLLMEngine(
            model_id="Qwen/Qwen2.5-0.5B",
            gpu_memory_utilization=0.5,
            max_model_len=512,
        )
        
        tokenizer = await engine.get_tokenizer()
        assert tokenizer is not None
    
    @pytest.mark.asyncio
    async def test_basic_sampling(self):
        """Test basic sampling."""
        from twinkle.sampler import VLLMEngine
        
        engine = VLLMEngine(
            model_id="Qwen/Qwen2.5-0.5B",
            gpu_memory_utilization=0.5,
            max_model_len=512,
        )
        
        tokenizer = await engine.get_tokenizer()
        prompt = "Hello, how are you?"
        prompt_ids = tokenizer.encode(prompt)
        
        response = await engine.sample(
            prompt_token_ids=prompt_ids,
            sampling_params=SamplingParams(max_tokens=32),
            num_samples=2,
        )
        
        assert len(response.sequences) == 2
        for seq in response.sequences:
            assert len(seq.tokens) > 0
            assert seq.stop_reason in ("stop", "length")
    
    @pytest.mark.asyncio
    async def test_sampling_with_logprobs(self):
        """Test sampling with log probabilities."""
        from twinkle.sampler import VLLMEngine
        
        engine = VLLMEngine(
            model_id="Qwen/Qwen2.5-0.5B",
            gpu_memory_utilization=0.5,
            max_model_len=512,
        )
        
        tokenizer = await engine.get_tokenizer()
        prompt_ids = tokenizer.encode("Hello")
        
        response = await engine.sample(
            prompt_token_ids=prompt_ids,
            sampling_params=SamplingParams(max_tokens=16),
            logprobs=True,
            prompt_logprobs=True,
            topk_prompt_logprobs=5,
        )
        
        assert len(response.sequences) == 1
        assert response.sequences[0].logprobs is not None
        assert response.prompt_logprobs is not None
        assert response.topk_prompt_logprobs is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
