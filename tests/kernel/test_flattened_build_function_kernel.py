import sys
import types
import unittest

import torch
import torch.nn.functional as F

from twinkle.kernel.base import is_kernels_available
from twinkle.kernel.function import register_function_kernel, apply_function_kernel
from twinkle.kernel.registry import get_global_function_registry


def _ensure_test_packages() -> None:
    if "tests" not in sys.modules:
        tests_pkg = types.ModuleType("tests")
        tests_pkg.__path__ = []
        sys.modules["tests"] = tests_pkg
    if "tests.kernel" not in sys.modules:
        kernel_pkg = types.ModuleType("tests.kernel")
        kernel_pkg.__path__ = []
        sys.modules["tests.kernel"] = kernel_pkg


def _reference_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


class TestFlattenedBuildFunctionKernel(unittest.TestCase):
    def setUp(self):
        if not is_kernels_available():
            self.skipTest("kernels package not available in this environment.")
        get_global_function_registry()._clear()

    def tearDown(self):
        get_global_function_registry()._clear()

    def test_flattened_build_replaces_function(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available in this environment.")
        try:
            from kernels import has_kernel
        except Exception:
            self.skipTest("kernels package missing has_kernel.")
        if not has_kernel("kernels-test/flattened-build"):
            self.skipTest("kernels-test/flattened-build not available.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_flattened_build_module"
        temp_module = types.ModuleType(module_name)

        def original(x: torch.Tensor) -> torch.Tensor:
            return _reference_silu_and_mul(x)

        temp_module.silu_and_mul = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="silu_and_mul",
                target_module=module_name,
                repo_id="kernels-test/flattened-build",
                device="cuda",
                mode="inference",
            )

            applied = apply_function_kernel(
                target_module=module_name,
                device="cuda",
                mode="inference",
            )

            self.assertEqual(applied, [f"{module_name}.silu_and_mul"])
            self.assertIsNot(temp_module.silu_and_mul, original)

            x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
            y_kernel = temp_module.silu_and_mul(x)
            y_ref = _reference_silu_and_mul(x)
            self.assertTrue(torch.allclose(y_kernel, y_ref, atol=1e-3, rtol=1e-3))
        finally:
            sys.modules.pop(module_name, None)

    def test_flattened_build_device_filter(self):
        _ensure_test_packages()
        module_name = "tests.kernel._tmp_flattened_build_device"
        temp_module = types.ModuleType(module_name)

        def original(x: torch.Tensor) -> torch.Tensor:
            return _reference_silu_and_mul(x)

        temp_module.silu_and_mul = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="silu_and_mul",
                target_module=module_name,
                repo_id="kernels-test/flattened-build",
                device="cuda",
                mode="inference",
            )

            applied = apply_function_kernel(
                target_module=module_name,
                device="cpu",
                mode="inference",
            )

            self.assertEqual(applied, [])
            self.assertIs(temp_module.silu_and_mul, original)
        finally:
            sys.modules.pop(module_name, None)

    def test_flattened_build_mode_filter(self):
        _ensure_test_packages()
        module_name = "tests.kernel._tmp_flattened_build_mode"
        temp_module = types.ModuleType(module_name)

        def original(x: torch.Tensor) -> torch.Tensor:
            return _reference_silu_and_mul(x)

        temp_module.silu_and_mul = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="silu_and_mul",
                target_module=module_name,
                repo_id="kernels-test/flattened-build",
                device="cuda",
                mode="inference",
            )

            applied = apply_function_kernel(
                target_module=module_name,
                device="cuda",
                mode="train",
            )

            self.assertEqual(applied, [])
            self.assertIs(temp_module.silu_and_mul, original)
        finally:
            sys.modules.pop(module_name, None)

    def test_flattened_build_strict_raises_on_no_match(self):
        _ensure_test_packages()
        module_name = "tests.kernel._tmp_flattened_build_strict"
        temp_module = types.ModuleType(module_name)

        def original(x: torch.Tensor) -> torch.Tensor:
            return _reference_silu_and_mul(x)

        temp_module.silu_and_mul = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="silu_and_mul",
                target_module=module_name,
                repo_id="kernels-test/flattened-build",
                device="cuda",
                mode="inference",
            )

            with self.assertRaises(ValueError):
                apply_function_kernel(
                    target_module=module_name,
                    device="cpu",
                    mode="inference",
                    strict=True,
                )
        finally:
            sys.modules.pop(module_name, None)
