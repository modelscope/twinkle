# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Kernel module unit tests
"""
import os
import pytest
from unittest.mock import MagicMock, Mock, patch

from twinkle.kernel import kernelize_model, register_external_layer, register_kernels, register_layer_kernel
from twinkle.kernel.base import is_kernels_available, is_kernels_enabled, to_kernels_mode
from twinkle.kernel.registry import (ExternalLayerRegistry, LayerRegistry, get_global_external_layer_registry,
                                     get_global_function_registry, get_global_layer_registry, get_layer_spec,
                                     register_layer)


class TestBase:
    """Test base helpers and env vars."""

    def test_is_kernels_available(self):
        """Test kernels availability check."""
        result = is_kernels_available()
        assert isinstance(result, bool)

    def test_kernels_enabled_env_var(self):
        """Test env var controls kernels enablement."""
        original = os.environ.get('TWINKLE_USE_KERNELS')
        try:
            os.environ['TWINKLE_USE_KERNELS'] = 'YES'
            from twinkle.kernel.base import _kernels_enabled
            assert _kernels_enabled()

            os.environ['TWINKLE_USE_KERNELS'] = 'NO'
            import importlib

            import twinkle.kernel.base
            importlib.reload(twinkle.kernel.base)
            from twinkle.kernel.base import _kernels_enabled
            assert not _kernels_enabled()
        finally:
            if original is not None:
                os.environ['TWINKLE_USE_KERNELS'] = original
            else:
                os.environ.pop('TWINKLE_USE_KERNELS', None)

    def test_to_kernels_mode(self):
        """Test mode conversion."""
        if not is_kernels_available():
            pytest.skip('kernels package not available')

        assert to_kernels_mode('train').name == 'TRAINING'
        assert to_kernels_mode('inference').name == 'INFERENCE'
        assert to_kernels_mode('compile').name == 'TORCH_COMPILE'


class TestLayerRegistry:
    """Test layer registry."""

    def setup_method(self):
        self.registry = LayerRegistry()

    def test_register_and_get(self):
        """Test register and lookup."""
        mock_spec = Mock()
        self.registry.register('TestLayer', mock_spec, 'cuda')

        result = self.registry.get('TestLayer', 'cuda')
        assert result == mock_spec

        result = self.registry.get('NonExistent', 'cuda')
        assert result is None

    def test_register_multiple_devices(self):
        """Test registration for multiple devices."""
        mock_cuda = Mock()
        mock_npu = Mock()

        self.registry.register('TestLayer', mock_cuda, 'cuda')
        self.registry.register('TestLayer', mock_npu, 'npu')

        assert self.registry.get('TestLayer', 'cuda') == mock_cuda
        assert self.registry.get('TestLayer', 'npu') == mock_npu

    def test_get_without_device(self):
        """Test lookup without device."""
        mock_spec = Mock()
        self.registry.register('TestLayer', mock_spec, 'cuda')

        result = self.registry.get('TestLayer')
        assert result == mock_spec

    def test_has(self):
        """Test has checks."""
        mock_spec = Mock()
        assert not self.registry.has('TestLayer')

        self.registry.register('TestLayer', mock_spec, 'cuda')
        assert self.registry.has('TestLayer')
        assert self.registry.has('TestLayer', 'cuda')
        assert not self.registry.has('TestLayer', 'npu')

    def test_list_kernel_names(self):
        """Test listing kernel names."""
        mock_spec = Mock()
        self.registry.register('Layer1', mock_spec, 'cuda')
        self.registry.register('Layer2', mock_spec, 'cuda')

        names = self.registry.list_kernel_names()
        assert sorted(names) == sorted(['Layer1', 'Layer2'])


class TestExternalLayerRegistry:
    """Test external layer registry."""

    def setup_method(self):
        self.registry = ExternalLayerRegistry()

    def test_register_and_get(self):
        """Test register and lookup."""
        mock_class = Mock
        self.registry.register(mock_class, 'LlamaAttention')

        result = self.registry.get(mock_class)
        assert result == 'LlamaAttention'

    def test_has(self):
        """Test has checks."""
        mock_class = Mock
        assert not self.registry.has(mock_class)

        self.registry.register(mock_class, 'LlamaAttention')
        assert self.registry.has(mock_class)

    def test_list_mappings(self):
        """Test list mappings."""

        class MockClass1:
            pass

        class MockClass2:
            pass

        self.registry.register(MockClass1, 'LlamaAttention')
        self.registry.register(MockClass2, 'LlamaMLP')

        mappings = self.registry.list_mappings()
        assert len(mappings) == 2


class TestRegisterLayer:
    """Test global register helpers."""

    def setup_method(self):
        get_global_layer_registry()._clear()
        get_global_function_registry()._clear()

    def test_register_and_get_spec(self):
        """Test global register and lookup."""
        mock_spec = Mock()
        register_layer('TestLayer', mock_spec, 'cuda')

        result = get_layer_spec('TestLayer', 'cuda')
        assert result == mock_spec


class TestRegisterLayerKernel:
    """Test register_layer_kernel."""

    def setup_method(self):
        get_global_layer_registry()._clear()

    def test_register_without_kernels_package(self):
        """Test registration when kernels package missing."""
        with patch('twinkle.kernel.layer.is_kernels_available', return_value=False):
            register_layer_kernel('TestLayer', repo_id='test/repo')
            assert get_layer_spec('TestLayer') is None

    def test_register_with_kernels_package(self):
        """Test registration when kernels package available."""
        if not is_kernels_available():
            pytest.skip('kernels package not available')

        register_layer_kernel(
            kernel_name='TestLayer',
            repo_id='kernels-community/test',
        )

        assert get_layer_spec('TestLayer') is not None


class TestKernelizeModel:
    """Test kernelize_model."""

    def test_kernelize_without_kernels_enabled(self):
        """Test returns original model when kernels disabled."""
        with patch('twinkle.kernel.layer.is_kernels_enabled', return_value=False):
            mock_model = Mock()
            result = kernelize_model(mock_model)
            assert result == mock_model

    @patch('twinkle.kernel.layer.is_kernels_available', return_value=False)
    def test_kernelize_without_kernels_available(self, mock_available):
        """Test returns original model when kernels unavailable."""
        mock_model = Mock()
        result = kernelize_model(mock_model)
        assert result == mock_model


class TestRegisterExternalLayer:
    """Test register_external_layer."""

    def setup_method(self):
        get_global_external_layer_registry()._clear()

    def test_register_external_layer(self):
        """Test registering external layer."""
        mock_class = Mock

        register_external_layer(mock_class, 'LlamaAttention')

        result = get_global_external_layer_registry().get(mock_class)
        assert result == 'LlamaAttention'

    def test_register_external_qwen_layer(self):
        """Test registering Qwen2 external layer mapping."""
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        except ImportError:
            pytest.skip('transformers package not available')

        register_external_layer(Qwen2Attention, 'LlamaAttention')

        registry = get_global_external_layer_registry()
        assert registry.has(Qwen2Attention)
        assert registry.get(Qwen2Attention) == 'LlamaAttention'

    def test_register_external_layer_adds_kernel_layer_name(self):
        """Test register_external_layer sets kernel_layer_name."""
        if not is_kernels_available():
            pytest.skip('kernels package not available')

        class TestLayer:
            pass

        register_external_layer(TestLayer, 'TestKernel')

        assert hasattr(TestLayer, 'kernel_layer_name')
        assert TestLayer.kernel_layer_name == 'TestKernel'


class TestRegisterKernels:
    """Test register_kernels batch registration."""

    def setup_method(self):
        get_global_layer_registry()._clear()

    @patch('twinkle.kernel.layer.is_kernels_available', return_value=False)
    def test_register_layers_without_kernels(self, mock_available):
        """Test layer batch registration when kernels missing."""
        config = {
            'layers': {
                'LlamaAttention': {
                    'repo_id': 'kernels-community/llama-attention'
                },
                'LlamaMLP': {
                    'repo_id': 'kernels-community/llama-mlp'
                },
            }
        }

        register_kernels(config)

        assert get_layer_spec('LlamaAttention') is None
        assert get_layer_spec('LlamaMLP') is None

    def test_register_functions(self):
        """Test function batch registration."""
        config = {
            'functions': {
                'apply_rotary_pos_emb': {
                    'func_impl': Mock,
                    'target_module': 'test',
                    'device': 'cpu',
                    'mode': 'inference',
                }
            }
        }

        register_kernels(config)
        specs = get_global_function_registry().list_specs()
        assert len(specs) == 1
        spec = specs[0]
        assert spec.func_name == 'apply_rotary_pos_emb'
        assert spec.target_module == 'test'
        assert spec.func_impl == Mock
        assert spec.device == 'cpu'
        assert spec.mode == 'inference'


class TestModeSupport:
    """Test mode support."""

    def setup_method(self):
        get_global_layer_registry()._clear()

    @patch('twinkle.kernel.layer.is_kernels_available', return_value=False)
    def test_register_with_mode_fallback(self, mock_available):
        """Test fallback mode mapping when mode is None."""
        from kernels import Mode

        from twinkle.kernel.layer import _to_hf_mode, register_layer_kernel

        result = _to_hf_mode(None)
        assert result == Mode.FALLBACK

    def test_to_hf_mode_conversion(self):
        """Test Twinkle mode to HF kernels Mode conversion."""
        if not is_kernels_available():
            pytest.skip('kernels package not available')

        from kernels import Mode

        from twinkle.kernel.layer import _to_hf_mode

        assert _to_hf_mode('train') == Mode.TRAINING
        assert _to_hf_mode('inference') == Mode.INFERENCE
        assert _to_hf_mode('compile') == Mode.TORCH_COMPILE

    @patch('twinkle.kernel.layer.is_kernels_available', return_value=False)
    def test_register_multiple_modes(self, mock_available):
        """Test registering multiple modes for the same layer."""
        registry = get_global_layer_registry()

        class MockRepo:
            pass

        repo_inference = MockRepo()
        repo_training = MockRepo()

        from kernels import Mode

        registry.register('TestLayer', repo_inference, 'cuda', Mode.INFERENCE)
        registry.register('TestLayer', repo_training, 'cuda', Mode.TRAINING)

        assert registry.has('TestLayer', 'cuda', Mode.INFERENCE)
        assert registry.has('TestLayer', 'cuda', Mode.TRAINING)

        result = registry.get('TestLayer', 'cuda', Mode.INFERENCE)
        assert result == repo_inference


if __name__ == '__main__':
    pytest.main([__file__])
