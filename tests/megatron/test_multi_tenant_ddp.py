"""
Unit tests for Multi-Tenant LoRA DDP.

Tests:
1. Tenant context (ContextVar)
2. Tenant manager lifecycle
3. Dynamic tenant add/remove
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class TestTenantContext(unittest.TestCase):
    """Tests for tenant_context module."""
    
    def setUp(self):
        from twinkle.megatron.distributed.tenant_context import set_current_tenant
        set_current_tenant(None)
    
    def test_get_set(self):
        from twinkle.megatron.distributed.tenant_context import (
            get_current_tenant, set_current_tenant
        )
        
        self.assertIsNone(get_current_tenant())
        set_current_tenant("a")
        self.assertEqual(get_current_tenant(), "a")
        set_current_tenant(None)
        self.assertIsNone(get_current_tenant())
    
    def test_scope(self):
        from twinkle.megatron.distributed.tenant_context import (
            get_current_tenant, tenant_scope
        )
        
        with tenant_scope("x"):
            self.assertEqual(get_current_tenant(), "x")
            with tenant_scope("y"):
                self.assertEqual(get_current_tenant(), "y")
            self.assertEqual(get_current_tenant(), "x")
        self.assertIsNone(get_current_tenant())
    
    def test_require_tenant(self):
        from twinkle.megatron.distributed.tenant_context import (
            require_tenant, tenant_scope
        )
        
        with self.assertRaises(RuntimeError):
            require_tenant()
        
        with tenant_scope("t"):
            self.assertEqual(require_tenant(), "t")
    
    def test_thread_isolation(self):
        from twinkle.megatron.distributed.tenant_context import (
            get_current_tenant, set_current_tenant
        )
        
        results = {}
        
        def worker(tid):
            set_current_tenant(tid)
            import time
            time.sleep(0.01)
            results[tid] = get_current_tenant()
        
        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        for i in range(5):
            self.assertEqual(results[f"t{i}"], f"t{i}")
    
    def test_generate_id(self):
        from twinkle.megatron.distributed.tenant_context import generate_tenant_id
        
        ids = [generate_tenant_id() for _ in range(100)]
        self.assertEqual(len(ids), len(set(ids)))


class TestTenantManager(unittest.TestCase):
    """Tests for TenantManager."""
    
    def test_initialize_finalize(self):
        from twinkle.megatron.distributed.tenant_manager import TenantManager
        
        model = nn.Linear(10, 10)
        manager = TenantManager(model)
        
        # Mock PEFT
        with patch('twinkle.megatron.distributed.tenant_manager.PEFT_AVAILABLE', False):
            # Add fake lora param
            lora_param = nn.Parameter(torch.randn(4, 10))
            lora_param.requires_grad = True
            model.lora_A = nn.ParameterDict({'test': lora_param})
            
            # Need to patch named_parameters
            original_named_params = model.named_parameters
            def mock_named_params():
                yield 'weight', model.weight
                yield 'lora_A.test.lora_A', lora_param
            model.named_parameters = mock_named_params
            
            tid = manager.initialize(
                optimizer_kwargs={'lr': 1e-4},
                adapter_name='test',
            )
            
            self.assertTrue(manager.has(tid))
            self.assertIn(tid, manager.list())
            
            state = manager.get(tid)
            self.assertEqual(state.adapter_name, 'test')
            
            manager.finalize(tid)
            self.assertFalse(manager.has(tid))
    
    def test_callbacks(self):
        from twinkle.megatron.distributed.tenant_manager import TenantManager
        
        model = nn.Linear(10, 10)
        manager = TenantManager(model)
        
        added = []
        removed = []
        
        manager.register_add_callback(lambda s: added.append(s.tenant_id))
        manager.register_remove_callback(lambda s: removed.append(s.tenant_id))
        
        with patch('twinkle.megatron.distributed.tenant_manager.PEFT_AVAILABLE', False):
            lora_param = nn.Parameter(torch.randn(4, 10))
            original_named_params = model.named_parameters
            def mock_named_params():
                yield 'lora_A.test.lora_A', lora_param
            model.named_parameters = mock_named_params
            
            tid = manager.initialize(adapter_name='test')
            self.assertEqual(added, [tid])
            
            manager.finalize(tid)
            self.assertEqual(removed, [tid])


class TestMultiTenantDDP(unittest.TestCase):
    """Tests for MultiTenantLoRADDP."""
    
    @patch('twinkle.megatron.distributed.multi_tenant_ddp.MEGATRON_AVAILABLE', False)
    def test_requires_megatron(self):
        from twinkle.megatron.distributed.multi_tenant_ddp import MultiTenantLoRADDP
        
        with self.assertRaises(ImportError):
            MultiTenantLoRADDP(
                config=MagicMock(),
                ddp_config=MagicMock(),
                module=nn.Linear(10, 10),
            )


class TestMegatronMultiAdapter(unittest.TestCase):
    """Tests for MegatronMultiAdapter."""
    
    def test_adapter_var(self):
        from twinkle.megatron.model.multi_tenant_megatron import MegatronMultiAdapter
        
        MegatronMultiAdapter._patched = False
        
        self.assertIsNone(MegatronMultiAdapter.get_current_adapter_name())
        MegatronMultiAdapter.set_current_adapter_name("a")
        self.assertEqual(MegatronMultiAdapter.get_current_adapter_name(), "a")
        MegatronMultiAdapter.set_current_adapter_name(None)


if __name__ == "__main__":
    unittest.main()
