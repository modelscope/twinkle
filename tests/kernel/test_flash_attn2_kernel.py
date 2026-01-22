import sys
import types
import unittest
from typing import Optional, List

import torch

from twinkle.kernel.base import is_kernels_available
from twinkle.kernel.function import register_function_kernel, apply_function_kernel
from twinkle.kernel.registry import get_global_function_registry

#python -m unittest tests.kernel.test_flash_attn2_kernel.TestFlashAttn2FunctionKernel
def _ensure_test_packages() -> None:
    if "tests" not in sys.modules:
        tests_pkg = types.ModuleType("tests")
        tests_pkg.__path__ = []
        sys.modules["tests"] = tests_pkg
    if "tests.kernel" not in sys.modules:
        kernel_pkg = types.ModuleType("tests.kernel")
        kernel_pkg.__path__ = []
        sys.modules["tests.kernel"] = kernel_pkg


class TestFlashAttn2FunctionKernel(unittest.TestCase):
    def setUp(self):
        get_global_function_registry()._clear()

    def tearDown(self):
        get_global_function_registry()._clear()

    def test_flash_attn2_function_kernel_replacement(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available in this environment.")
        if not is_kernels_available():
            self.skipTest("kernels package not available in this environment.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_flash_attn2_module"
        temp_module = types.ModuleType(module_name)

        def fwd(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            out: Optional[torch.Tensor] = None,
            alibi_slopes: Optional[torch.Tensor] = None,
            p_dropout: float = 0.0,
            softmax_scale: Optional[float] = None,
            is_causal: bool = False,
            window_size_left: int = -1,
            window_size_right: int = -1,
            softcap: float = 0.0,
            return_softmax: bool = False,
            gen: Optional[torch.Generator] = None,
        ) -> List[torch.Tensor]:
            raise RuntimeError("Original fwd should be replaced by flash-attn2.")

        def varlen_fwd(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            out: Optional[torch.Tensor] = None,
            seqused_k: Optional[torch.Tensor] = None,
            leftpad_k: Optional[torch.Tensor] = None,
            block_table: Optional[torch.Tensor] = None,
            alibi_slopes: Optional[torch.Tensor] = None,
            max_seqlen_q: int = 0,
            max_seqlen_k: int = 0,
            p_dropout: float = 0.0,
            softmax_scale: Optional[float] = None,
            zero_tensors: bool = False,
            is_causal: bool = False,
            window_size_left: int = -1,
            window_size_right: int = -1,
            softcap: float = 0.0,
            return_softmax: bool = False,
            gen: Optional[torch.Generator] = None,
        ) -> List[torch.Tensor]:
            raise RuntimeError("Original varlen_fwd should be replaced by flash-attn2.")

        def bwd(
            dout: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            out: torch.Tensor,
            softmax_lse: torch.Tensor,
            dq: Optional[torch.Tensor] = None,
            dk: Optional[torch.Tensor] = None,
            dv: Optional[torch.Tensor] = None,
            alibi_slopes: Optional[torch.Tensor] = None,
            p_dropout: float = 0.0,
            softmax_scale: Optional[float] = None,
            is_causal: bool = False,
            window_size_left: int = -1,
            window_size_right: int = -1,
            softcap: float = 0.0,
            deterministic: bool = False,
            gen: Optional[torch.Generator] = None,
            rng_state: Optional[torch.Tensor] = None,
        ) -> List[torch.Tensor]:
            raise RuntimeError("Original bwd should be replaced by flash-attn2.")

        def varlen_bwd(
            dout: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            out: torch.Tensor,
            softmax_lse: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            dq: Optional[torch.Tensor] = None,
            dk: Optional[torch.Tensor] = None,
            dv: Optional[torch.Tensor] = None,
            alibi_slopes: Optional[torch.Tensor] = None,
            max_seqlen_q: int = 0,
            max_seqlen_k: int = 0,
            p_dropout: float = 0.0,
            softmax_scale: Optional[float] = None,
            zero_tensors: bool = False,
            is_causal: bool = False,
            window_size_left: int = -1,
            window_size_right: int = -1,
            softcap: float = 0.0,
            deterministic: bool = False,
            gen: Optional[torch.Generator] = None,
            rng_state: Optional[torch.Tensor] = None,
        ) -> List[torch.Tensor]:
            raise RuntimeError("Original varlen_bwd should be replaced by flash-attn2.")

        temp_module.fwd = fwd
        temp_module.varlen_fwd = varlen_fwd
        temp_module.bwd = bwd
        temp_module.varlen_bwd = varlen_bwd
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            repo_id = "kernels-community/flash-attn2"
            register_function_kernel(
                func_name="fwd",
                target_module=module_name,
                repo_id=repo_id,
                device="cuda",
                mode="inference",
            )
            register_function_kernel(
                func_name="varlen_fwd",
                target_module=module_name,
                repo_id=repo_id,
                device="cuda",
                mode="inference",
            )
            register_function_kernel(
                func_name="bwd",
                target_module=module_name,
                repo_id=repo_id,
                device="cuda",
                mode="train",
            )
            register_function_kernel(
                func_name="varlen_bwd",
                target_module=module_name,
                repo_id=repo_id,
                device="cuda",
                mode="train",
            )

            apply_function_kernel(
                target_module=module_name,
                device="cuda",
                mode="inference",
            )
            apply_function_kernel(
                target_module=module_name,
                device="cuda",
                mode="train",
            )

            self.assertIsNot(temp_module.fwd, fwd)
            self.assertIsNot(temp_module.varlen_fwd, varlen_fwd)
            self.assertIsNot(temp_module.bwd, bwd)
            self.assertIsNot(temp_module.varlen_bwd, varlen_bwd)

            q = torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.float16)
            k = torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.float16)
            v = torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.float16)
            out = temp_module.fwd(q, k, v)
            self.assertIsInstance(out, (list, tuple))
            self.assertGreaterEqual(len(out), 1)
            self.assertEqual(out[0].shape, q.shape)
        finally:
            sys.modules.pop(module_name, None)
