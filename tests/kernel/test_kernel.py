# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Kernel module unit tests
"""
import os
import unittest
from unittest.mock import Mock, MagicMock, patch

from twinkle.kernel import (
    kernelize_model,
    register_layer_kernel,
    register_external_layer,
    register_kernels,
)
from twinkle.kernel.base import (
    get_device_type,
    is_kernels_available,
    is_kernels_enabled,
    to_kernels_mode,
)
from twinkle.kernel.registry import (
    LayerRegistry,
    ExternalLayerRegistry,
    register_layer,
    get_layer_spec,
    get_global_layer_registry,
    get_global_external_layer_registry,
    get_global_function_registry,
)


class TestBase(unittest.TestCase):
    """Test base helpers and env vars."""

    def test_get_device_type_no_torch(self):
        """Test device detection without torch."""
        with patch("twinkle.kernel.base.exists", return_value=False):
            result = get_device_type()
            self.assertIsNone(result)

    def test_is_kernels_available(self):
        """Test kernels availability check."""
        result = is_kernels_available()
        self.assertIsInstance(result, bool)

    def test_kernels_enabled_env_var(self):
        """Test env var controls kernels enablement."""
        original = os.environ.get("TWINKLE_USE_KERNELS")
        try:
            os.environ["TWINKLE_USE_KERNELS"] = "YES"
            from twinkle.kernel.base import _kernels_enabled
            self.assertTrue(_kernels_enabled())

            os.environ["TWINKLE_USE_KERNELS"] = "NO"
            import importlib
            import twinkle.kernel.base
            importlib.reload(twinkle.kernel.base)
            from twinkle.kernel.base import _kernels_enabled
            self.assertFalse(_kernels_enabled())
        finally:
            if original is not None:
                os.environ["TWINKLE_USE_KERNELS"] = original
            else:
                os.environ.pop("TWINKLE_USE_KERNELS", None)

    def test_to_kernels_mode(self):
        """Test mode conversion."""
        if not is_kernels_available():
            self.skipTest("kernels package not available")

        self.assertEqual(to_kernels_mode("train").name, "TRAINING")
        self.assertEqual(to_kernels_mode("inference").name, "INFERENCE")
        self.assertEqual(to_kernels_mode("compile").name, "TORCH_COMPILE")


class TestLayerRegistry(unittest.TestCase):
    """Test layer registry."""

    def setUp(self):
        self.registry = LayerRegistry()

    def test_register_and_get(self):
        """Test register and lookup."""
        mock_spec = Mock()
        self.registry.register("TestLayer", mock_spec, "cuda")

        result = self.registry.get("TestLayer", "cuda")
        self.assertEqual(result, mock_spec)

        result = self.registry.get("NonExistent", "cuda")
        self.assertIsNone(result)

    def test_register_multiple_devices(self):
        """Test registration for multiple devices."""
        mock_cuda = Mock()
        mock_npu = Mock()

        self.registry.register("TestLayer", mock_cuda, "cuda")
        self.registry.register("TestLayer", mock_npu, "npu")

        self.assertEqual(self.registry.get("TestLayer", "cuda"), mock_cuda)
        self.assertEqual(self.registry.get("TestLayer", "npu"), mock_npu)

    def test_get_without_device(self):
        """Test lookup without device."""
        mock_spec = Mock()
        self.registry.register("TestLayer", mock_spec, "cuda")

        result = self.registry.get("TestLayer")
        self.assertEqual(result, mock_spec)

    def test_has(self):
        """Test has checks."""
        mock_spec = Mock()
        self.assertFalse(self.registry.has("TestLayer"))

        self.registry.register("TestLayer", mock_spec, "cuda")
        self.assertTrue(self.registry.has("TestLayer"))
        self.assertTrue(self.registry.has("TestLayer", "cuda"))
        self.assertFalse(self.registry.has("TestLayer", "npu"))

    def test_list_kernel_names(self):
        """Test listing kernel names."""
        mock_spec = Mock()
        self.registry.register("Layer1", mock_spec, "cuda")
        self.registry.register("Layer2", mock_spec, "cuda")

        names = self.registry.list_kernel_names()
        self.assertCountEqual(names, ["Layer1", "Layer2"])


class TestExternalLayerRegistry(unittest.TestCase):
    """Test external layer registry."""

    def setUp(self):
        self.registry = ExternalLayerRegistry()

    def test_register_and_get(self):
        """Test register and lookup."""
        mock_class = Mock
        self.registry.register(mock_class, "LlamaAttention")

        result = self.registry.get(mock_class)
        self.assertEqual(result, "LlamaAttention")

    def test_has(self):
        """Test has checks."""
        mock_class = Mock
        self.assertFalse(self.registry.has(mock_class))

        self.registry.register(mock_class, "LlamaAttention")
        self.assertTrue(self.registry.has(mock_class))

    def test_list_mappings(self):
        """Test list mappings."""
        class MockClass1:
            pass

        class MockClass2:
            pass

        self.registry.register(MockClass1, "LlamaAttention")
        self.registry.register(MockClass2, "LlamaMLP")

        mappings = self.registry.list_mappings()
        self.assertEqual(len(mappings), 2)


class TestRegisterLayer(unittest.TestCase):
    """Test global register helpers."""

    def setUp(self):
        get_global_layer_registry()._clear()
        get_global_function_registry()._clear()

    def test_register_and_get_spec(self):
        """Test global register and lookup."""
        mock_spec = Mock()
        register_layer("TestLayer", mock_spec, "cuda")

        result = get_layer_spec("TestLayer", "cuda")
        self.assertEqual(result, mock_spec)


class TestRegisterLayerKernel(unittest.TestCase):
    """Test register_layer_kernel."""

    def setUp(self):
        get_global_layer_registry()._clear()

    def test_register_without_kernels_package(self):
        """Test registration when kernels package missing."""
        with patch("twinkle.kernel.layer.is_kernels_available", return_value=False):
            register_layer_kernel("TestLayer", repo_id="test/repo")
            self.assertIsNone(get_layer_spec("TestLayer"))

    def test_register_with_kernels_package(self):
        """Test registration when kernels package available."""
        if not is_kernels_available():
            self.skipTest("kernels package not available")

        register_layer_kernel(
            kernel_name="TestLayer",
            repo_id="kernels-community/test",
        )

        self.assertIsNotNone(get_layer_spec("TestLayer"))


class TestKernelizeModel(unittest.TestCase):
    """Test kernelize_model."""

    def test_kernelize_without_kernels_enabled(self):
        """Test returns original model when kernels disabled."""
        with patch("twinkle.kernel.layer.is_kernels_enabled", return_value=False):
            mock_model = Mock()
            result = kernelize_model(mock_model)
            self.assertEqual(result, mock_model)

    @patch("twinkle.kernel.layer.is_kernels_available", return_value=False)
    def test_kernelize_without_kernels_available(self, mock_available):
        """Test returns original model when kernels unavailable."""
        mock_model = Mock()
        result = kernelize_model(mock_model)
        self.assertEqual(result, mock_model)


class TestRegisterExternalLayer(unittest.TestCase):
    """Test register_external_layer."""

    def setUp(self):
        get_global_external_layer_registry()._clear()

    def test_register_external_layer(self):
        """Test registering external layer."""
        mock_class = Mock

        register_external_layer(mock_class, "LlamaAttention")

        result = get_global_external_layer_registry().get(mock_class)
        self.assertEqual(result, "LlamaAttention")

    def test_register_external_qwen_layer(self):
        """Test registering Qwen2 external layer mapping."""
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        except ImportError:
            self.skipTest("transformers package not available")

        register_external_layer(Qwen2Attention, "LlamaAttention")

        registry = get_global_external_layer_registry()
        self.assertTrue(registry.has(Qwen2Attention))
        self.assertEqual(registry.get(Qwen2Attention), "LlamaAttention")

    def test_register_external_layer_adds_kernel_layer_name(self):
        """Test register_external_layer sets kernel_layer_name."""
        if not is_kernels_available():
            self.skipTest("kernels package not available")

        class TestLayer:
            pass

        register_external_layer(TestLayer, "TestKernel")

        self.assertTrue(hasattr(TestLayer, "kernel_layer_name"))
        self.assertEqual(TestLayer.kernel_layer_name, "TestKernel")


class TestRegisterKernels(unittest.TestCase):
    """Test register_kernels batch registration."""

    def setUp(self):
        get_global_layer_registry()._clear()

    @patch("twinkle.kernel.layer.is_kernels_available", return_value=False)
    def test_register_layers_without_kernels(self, mock_available):
        """Test layer batch registration when kernels missing."""
        config = {
            "layers": {
                "LlamaAttention": {"repo_id": "kernels-community/llama-attention"},
                "LlamaMLP": {"repo_id": "kernels-community/llama-mlp"},
            }
        }

        register_kernels(config)

        self.assertIsNone(get_layer_spec("LlamaAttention"))
        self.assertIsNone(get_layer_spec("LlamaMLP"))

    def test_register_functions(self):
        """Test function batch registration."""
        config = {
            "functions": {
                "apply_rotary_pos_emb": {
                    "func_impl": Mock,
                    "target_module": "test",
                    "device": "cpu",
                    "mode": "inference",
                }
            }
        }

        register_kernels(config)
        specs = get_global_function_registry().list_specs()
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec.func_name, "apply_rotary_pos_emb")
        self.assertEqual(spec.target_module, "test")
        self.assertEqual(spec.func_impl, Mock)
        self.assertEqual(spec.device, "cpu")
        self.assertEqual(spec.mode, "inference")


class TestModeSupport(unittest.TestCase):
    """Test mode support."""

    def setUp(self):
        get_global_layer_registry()._clear()

    @patch("twinkle.kernel.layer.is_kernels_available", return_value=False)
    def test_register_with_mode_fallback(self, mock_available):
        """Test fallback mode mapping when mode is None."""
        from twinkle.kernel.layer import register_layer_kernel, _to_hf_mode
        from kernels import Mode

        result = _to_hf_mode(None)
        self.assertEqual(result, Mode.FALLBACK)

    def test_to_hf_mode_conversion(self):
        """Test Twinkle mode to HF kernels Mode conversion."""
        if not is_kernels_available():
            self.skipTest("kernels package not available")

        from twinkle.kernel.layer import _to_hf_mode
        from kernels import Mode

        self.assertEqual(_to_hf_mode("train"), Mode.TRAINING)
        self.assertEqual(_to_hf_mode("inference"), Mode.INFERENCE)
        self.assertEqual(_to_hf_mode("compile"), Mode.TORCH_COMPILE)

    @patch("twinkle.kernel.layer.is_kernels_available", return_value=False)
    def test_register_multiple_modes(self, mock_available):
        """Test registering multiple modes for the same layer."""
        registry = get_global_layer_registry()

        class MockRepo:
            pass

        repo_inference = MockRepo()
        repo_training = MockRepo()

        from kernels import Mode

        registry.register("TestLayer", repo_inference, "cuda", Mode.INFERENCE)
        registry.register("TestLayer", repo_training, "cuda", Mode.TRAINING)

        self.assertTrue(registry.has("TestLayer", "cuda", Mode.INFERENCE))
        self.assertTrue(registry.has("TestLayer", "cuda", Mode.TRAINING))

        result = registry.get("TestLayer", "cuda", Mode.INFERENCE)
        self.assertEqual(result, repo_inference)


if __name__ == "__main__":
    unittest.main()
