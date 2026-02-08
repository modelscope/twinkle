# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for Sampler classes (vLLMSampler, TorchSampler)."""

import pytest
from unittest.mock import MagicMock, patch

from twinkle.sampler.base import Sampler, SampleGroup
from twinkle.data_format.sampling import SamplingParams, SampleResponse, SampledSequence
from twinkle.data_format import Trajectory, InputFeature


# =============================================================================
# Tests for Base Sampler
# =============================================================================

class TestSamplerBase:
    """Tests for Sampler base class."""
    
    def test_not_encoded_with_input_ids(self):
        """Test _not_encoded returns False when input_ids present."""
        inputs = {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}
        assert Sampler._not_encoded(inputs) is False
    
    def test_not_encoded_with_input_embedding(self):
        """Test _not_encoded returns False when input_embedding present."""
        inputs = {'input_embedding': [[0.1, 0.2]], 'attention_mask': [1, 1]}
        assert Sampler._not_encoded(inputs) is False
    
    def test_not_encoded_trajectory(self):
        """Test _not_encoded returns True for trajectory dict."""
        trajectory = {'messages': [{'role': 'user', 'content': 'hello'}]}
        assert Sampler._not_encoded(trajectory) is True
    
    def test_not_encoded_empty_dict(self):
        """Test _not_encoded returns True for empty dict."""
        assert Sampler._not_encoded({}) is True


class TestSampleGroup:
    """Tests for SampleGroup dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        group = SampleGroup()
        assert group.adapter_name is None
        assert group.adapter_config is None
        assert group.template is None
        assert group.processor is None
        assert group.lora_int_id is None
        assert group.lora_ready is False


# =============================================================================
# Tests for SamplingParams
# =============================================================================

class TestSamplingParamsConversion:
    """Tests for SamplingParams conversion methods."""
    
    def test_from_dict_basic(self):
        """Test creating SamplingParams from dict."""
        d = {'max_tokens': 100, 'temperature': 0.8, 'top_p': 0.9}
        params = SamplingParams.from_dict(d)
        
        assert params.max_tokens == 100
        assert params.temperature == 0.8
        assert params.top_p == 0.9
    
    def test_from_dict_with_alternative_names(self):
        """Test from_dict with transformers-style names."""
        d = {'max_new_tokens': 128}
        params = SamplingParams.from_dict(d)
        
        assert params.max_tokens == 128
    
    def test_from_dict_filters_invalid(self):
        """Test from_dict filters invalid fields."""
        d = {'max_tokens': 100, 'invalid_field': 'ignored', 'another_bad': 123}
        params = SamplingParams.from_dict(d)
        
        assert params.max_tokens == 100
        assert not hasattr(params, 'invalid_field')
    
    def test_to_transformers_basic(self):
        """Test conversion to transformers kwargs."""
        params = SamplingParams(
            max_tokens=64,
            temperature=0.7,
            top_p=0.95,
        )
        
        gen_kwargs = params.to_transformers()
        
        assert gen_kwargs['max_new_tokens'] == 64
        assert gen_kwargs['temperature'] == 0.7
        assert gen_kwargs['top_p'] == 0.95
        assert gen_kwargs['do_sample'] is True
    
    def test_to_transformers_zero_temperature(self):
        """Test conversion with temperature=0 (greedy)."""
        params = SamplingParams(temperature=0.0)
        gen_kwargs = params.to_transformers()
        
        assert gen_kwargs['do_sample'] is False


# =============================================================================
# Tests for Mock Sampler Implementation
# =============================================================================

class MockSampler(Sampler):
    """Mock sampler for testing base class functionality."""
    
    def __init__(self):
        super().__init__()
        self._sample_called = False
        self._last_inputs = None
        self._last_params = None
    
    def sample(self, inputs, sampling_params=None, adapter_name=''):
        self._sample_called = True
        self._last_inputs = inputs
        self._last_params = sampling_params
        
        return SampleResponse(
            sequences=[SampledSequence(stop_reason='stop', tokens=[1, 2, 3])]
        )


class TestMockSampler:
    """Tests using MockSampler."""
    
    def test_normalize_inputs_dict(self):
        """Test _normalize_inputs with single dict."""
        sampler = MockSampler()
        result = sampler._normalize_inputs({'input_ids': [1, 2, 3]})
        assert result == [{'input_ids': [1, 2, 3]}]
    
    def test_normalize_inputs_list(self):
        """Test _normalize_inputs with list."""
        sampler = MockSampler()
        inputs = [{'input_ids': [1]}, {'input_ids': [2]}]
        result = sampler._normalize_inputs(inputs)
        assert result == inputs
    
    def test_is_trajectory_with_input_feature(self):
        """Test _is_trajectory returns False for InputFeature."""
        sampler = MockSampler()
        feat = {'input_ids': [1, 2, 3]}
        assert sampler._is_trajectory(feat) is False
    
    def test_is_trajectory_with_trajectory(self):
        """Test _is_trajectory returns True for Trajectory."""
        sampler = MockSampler()
        traj = {'messages': [{'role': 'user', 'content': 'hi'}]}
        assert sampler._is_trajectory(traj) is True
    
    def test_is_trajectory_with_list(self):
        """Test _is_trajectory with list input."""
        sampler = MockSampler()
        
        # List of trajectories
        trajs = [{'messages': [{'role': 'user', 'content': 'hi'}]}]
        assert sampler._is_trajectory(trajs) is True
        
        # List of input features
        feats = [{'input_ids': [1, 2, 3]}]
        assert sampler._is_trajectory(feats) is False
    
    def test_is_trajectory_with_empty_list(self):
        """Test _is_trajectory with empty list."""
        sampler = MockSampler()
        assert sampler._is_trajectory([]) is False
    
    def test_check_adapter_valid_default(self):
        """Test _check_adapter_valid with default adapter."""
        sampler = MockSampler()
        # Should not raise
        sampler._check_adapter_valid('')
    
    def test_check_adapter_valid_invalid(self):
        """Test _check_adapter_valid with invalid adapter."""
        sampler = MockSampler()
        with pytest.raises(AssertionError):
            sampler._check_adapter_valid('nonexistent')
    
    def test_add_adapter_to_sampler(self):
        """Test adding adapter to sampler."""
        sampler = MockSampler()
        mock_config = MagicMock()
        
        sampler.add_adapter_to_sampler('test_adapter', mock_config)
        
        assert 'test_adapter' in sampler.sample_group
        group = sampler.sample_group['test_adapter']
        assert group.adapter_name == 'test_adapter'
        assert group.adapter_config == mock_config
        assert group.lora_int_id == 1
        assert group.lora_ready is False
    
    def test_add_multiple_adapters(self):
        """Test adding multiple adapters."""
        sampler = MockSampler()
        
        sampler.add_adapter_to_sampler('adapter1', MagicMock())
        sampler.add_adapter_to_sampler('adapter2', MagicMock())
        sampler.add_adapter_to_sampler('adapter3', MagicMock())
        
        assert len(sampler.sample_group) == 4  # 3 + default
        
        ids = [g.lora_int_id for g in sampler.sample_group.values() if g.lora_int_id]
        assert len(set(ids)) == 3  # All unique
    
    def test_remove_adapter(self):
        """Test removing adapter."""
        sampler = MockSampler()
        sampler.add_adapter_to_sampler('test', MagicMock())
        
        assert 'test' in sampler.sample_group
        sampler.remove_adapter('test')
        assert 'test' not in sampler.sample_group
    
    def test_set_template(self):
        """Test setting template."""
        sampler = MockSampler()
        
        with patch('twinkle.sampler.base.construct_class') as mock_construct:
            mock_template = MagicMock()
            mock_construct.return_value = mock_template
            
            sampler.set_template('MockTemplate')
            
            assert sampler.template == mock_template
            assert sampler.sample_group[''].template == mock_template


# =============================================================================
# Integration Tests (require actual model/vLLM)
# =============================================================================

@pytest.mark.skip(reason="Requires model and GPU")
class TestVLLMSamplerIntegration:
    """Integration tests for vLLMSampler."""
    
    def test_sample_with_trajectory(self):
        """Test sampling with Trajectory input."""
        from twinkle.sampler import vLLMSampler
        
        sampler = vLLMSampler(
            model_id="Qwen/Qwen2.5-0.5B",
            engine_args={'max_model_len': 512, 'gpu_memory_utilization': 0.5}
        )
        sampler.set_template('Template', model_id="Qwen/Qwen2.5-0.5B")
        
        trajectory = Trajectory(
            messages=[{'role': 'user', 'content': 'Hello, how are you?'}]
        )
        
        response = sampler.sample(
            [trajectory],
            sampling_params=SamplingParams(max_tokens=32),
        )
        
        assert isinstance(response, SampleResponse)
        assert len(response.sequences) >= 1
        assert len(response.sequences[0].tokens) > 0
    
    def test_sample_with_input_feature(self):
        """Test sampling with InputFeature input."""
        from twinkle.sampler import vLLMSampler
        
        sampler = vLLMSampler(
            model_id="Qwen/Qwen2.5-0.5B",
            engine_args={'max_model_len': 512, 'gpu_memory_utilization': 0.5}
        )
        
        # Pre-tokenized input
        feat = InputFeature(input_ids=[1, 2, 3, 4, 5])
        
        response = sampler.sample(
            [feat],
            sampling_params=SamplingParams(max_tokens=16),
        )
        
        assert isinstance(response, SampleResponse)
        assert len(response.sequences) >= 1


@pytest.mark.skip(reason="Requires model and GPU")
class TestTorchSamplerIntegration:
    """Integration tests for TorchSampler."""
    
    def test_sample_with_trajectory(self):
        """Test sampling with Trajectory input."""
        from twinkle.sampler import TorchSampler
        
        sampler = TorchSampler(model_id="Qwen/Qwen2.5-0.5B")
        sampler.set_template('Template', model_id="Qwen/Qwen2.5-0.5B")
        
        trajectory = Trajectory(
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        
        response = sampler.sample(
            [trajectory],
            sampling_params=SamplingParams(max_tokens=16),
        )
        
        assert isinstance(response, SampleResponse)
        assert len(response.sequences) >= 1
    
    def test_sample_with_input_feature(self):
        """Test sampling with InputFeature input."""
        from twinkle.sampler import TorchSampler
        
        sampler = TorchSampler(model_id="Qwen/Qwen2.5-0.5B")
        
        feat = InputFeature(input_ids=[1, 2, 3])
        
        response = sampler.sample(
            [feat],
            sampling_params=SamplingParams(max_tokens=16),
        )
        
        assert isinstance(response, SampleResponse)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
