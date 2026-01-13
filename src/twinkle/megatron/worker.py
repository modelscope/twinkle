# Copyright (c) twinkle authors. All rights reserved.
"""Megatron Worker for Ray-based distributed training.

This module provides MegatronWorkerGroup for coordinated Ray actor-based
training with Megatron's collective operations.

NOTE: Currently PP > 1 is required for Ray mode training with LoRA.
PP=1 has gradient flow issues that need further investigation.

Example:
    worker_group = MegatronWorkerGroup(world_size=4, tp_size=2, pp_size=2)
    worker_group.init_all()
    worker_group.create_model_all('Qwen/Qwen2.5-0.5B-Instruct')
    worker_group.add_lora_all({'target_modules': ['linear_qkv'], 'r': 8})
    worker_group.set_optimizer_all(lr=1e-4)
    
    for batch in dataloader:
        losses = worker_group.forward_backward_all(batch)
        worker_group.step_all()
"""
import os
from typing import Any, Dict, List

import torch


def get_megatron_worker_class():
    """Returns a Ray remote class for Megatron workers."""
    import ray
    
    @ray.remote(num_gpus=1)
    class MegatronWorker:
        """Ray actor for a single Megatron rank."""
        
        def __init__(
            self,
            rank: int,
            world_size: int,
            master_addr: str,
            master_port: int,
            tp_size: int = 1,
            pp_size: int = 1,
            cp_size: int = 1,
            ep_size: int = 1,
        ):
            self.rank = rank
            self.world_size = world_size
            self.master_addr = master_addr
            self.master_port = master_port
            self.tp_size = tp_size
            self.pp_size = pp_size
            self.cp_size = cp_size
            self.ep_size = ep_size
            self.model = None
            self.optimizer = None
            self.hf_config = None
            
        def _get_local_gpu_id(self) -> int:
            """Get local GPU ID for this actor."""
            import ray
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if cvd is None:
                gpu_ids = ray.get_gpu_ids()
                return int(gpu_ids[0]) if gpu_ids else 0
            else:
                gpu_ids = ray.get_gpu_ids()
                if gpu_ids:
                    return cvd.split(",").index(str(int(gpu_ids[0])))
                return 0
                
        def init(self, model_config: Dict[str, Any] = None) -> bool:
            """Initialize distributed and Megatron parallel state."""
            import torch.distributed as dist
            from datetime import timedelta
            
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = str(self.master_port)
            os.environ["WORLD_SIZE"] = str(self.world_size)
            os.environ["RANK"] = str(self.rank)
            
            local_rank = self._get_local_gpu_id()
            os.environ["LOCAL_RANK"] = str(local_rank)
            torch.cuda.set_device(local_rank)
            
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
            
            from megatron.core import parallel_state as mpu
            if not mpu.is_initialized():
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=self.tp_size,
                    pipeline_model_parallel_size=self.pp_size,
                    context_parallel_size=self.cp_size,
                    expert_model_parallel_size=self.ep_size,
                )
            
            from megatron.core import tensor_parallel
            torch.manual_seed(42 + self.rank)
            tensor_parallel.model_parallel_cuda_manual_seed(42 + self.rank)
            
            print(f"[Worker rank={self.rank}] Initialized TP={self.tp_size} PP={self.pp_size}")
            return True
            
        def create_model(
            self,
            pretrained_model_name_or_path: str,
            mixed_precision: str = 'bf16',
            recompute_granularity: str = 'full',
            **kwargs,
        ) -> bool:
            """Create Megatron model."""
            from twinkle.megatron.model.bridge import TwinkleBridgeInitializer
            
            dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
            params_dtype = dtype_map.get(mixed_precision, torch.bfloat16)
            
            initializer = TwinkleBridgeInitializer(
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                cp_size=self.cp_size,
                ep_size=self.ep_size,
                params_dtype=params_dtype,
                recompute_granularity=recompute_granularity,
                **kwargs,
            )
            
            self.model = initializer.create_model(pretrained_model_name_or_path)
            self.hf_config = initializer._hf_config
            print(f"[Worker rank={self.rank}] Model created")
            return True
            
        def add_lora(self, lora_config: Dict[str, Any]) -> bool:
            """Add LoRA adapter."""
            from peft import get_peft_model, LoraConfig
            from peft.tuners.tuners_utils import BaseTuner
            import torch.nn as nn
            
            # Patch for Megatron's TransformerConfig
            orig_fn = BaseTuner._get_tied_target_modules
            def patched_fn(self, model: nn.Module):
                try:
                    return orig_fn(self, model)
                except AttributeError:
                    return []
            BaseTuner._get_tied_target_modules = patched_fn
            
            from twinkle.megatron.utils import set_linear_is_expert
            set_linear_is_expert(self.model)
            
            config = LoraConfig(**lora_config)
            self.model = get_peft_model(self.model, config)
            
            # Add compatibility methods for Megatron DDP
            if not hasattr(self.model, 'finish_grad_sync'):
                self.model.finish_grad_sync = lambda: None
            if not hasattr(self.model, 'start_grad_sync'):
                self.model.start_grad_sync = lambda: None
            if not hasattr(self.model, 'no_sync'):
                from contextlib import nullcontext
                self.model.no_sync = nullcontext
            
            # Create a dummy ddp_config that has necessary attributes
            if not hasattr(self.model, 'ddp_config') or self.model.ddp_config is None:
                class DummyDDPConfig:
                    use_megatron_fsdp = False
                    use_distributed_optimizer = False
                    overlap_grad_reduce = False
                    overlap_param_gather = False
                    bucket_size = None
                self.model.ddp_config = DummyDDPConfig()
            
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[Worker rank={self.rank}] LoRA added, trainable params={trainable}")
            return True
            
        def set_optimizer(self, lr: float = 1e-4, **kwargs) -> bool:
            """Set up optimizer."""
            from torch.optim import AdamW
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = AdamW(trainable_params, lr=lr, **kwargs)
            print(f"[Worker rank={self.rank}] Optimizer set")
            return True
            
        def forward_backward(self, batch: Dict[str, torch.Tensor]) -> float:
            """Execute forward-backward pass."""
            from functools import partial
            from megatron.core.pipeline_parallel import get_forward_backward_func
            
            local_rank = self._get_local_gpu_id()
            batch = {k: v.cuda(local_rank) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            seq_length = batch['input_ids'].shape[1]
            micro_batch_size = batch['input_ids'].shape[0]
            
            def forward_step_func(data_iterator, model):
                batch = next(data_iterator)
                input_ids = batch['input_ids']
                labels = batch.get('labels')
                attention_mask = batch.get('attention_mask')
                
                position_ids = torch.arange(
                    input_ids.shape[1], device=input_ids.device, dtype=torch.long
                ).unsqueeze(0).expand(input_ids.shape[0], -1)
                
                output = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                def loss_func(labels, output):
                    mask = (labels != -100).float()
                    loss = (output.float().view(-1) * mask.view(-1)).sum() / mask.sum().clamp(min=1)
                    return loss, {'loss': loss.detach()}
                
                return output, partial(loss_func, labels)
            
            self.model.train()
            forward_backward_func = get_forward_backward_func()
            
            losses = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=iter([batch]),
                model=[self.model],
                num_microbatches=1,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                forward_only=False,
            )
            
            if losses and isinstance(losses[0], dict) and 'loss' in losses[0]:
                return losses[0]['loss'].item()
            return 0.0
            
        def step(self) -> bool:
            """Optimizer step."""
            if self.optimizer is None:
                return False
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True
            
        def cleanup(self) -> bool:
            """Clean up resources."""
            import torch.distributed as dist
            from megatron.core import parallel_state as mpu
            try:
                if dist.is_initialized():
                    dist.barrier()
                if mpu.is_initialized():
                    mpu.destroy_model_parallel()
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                print(f"[Worker rank={self.rank}] Cleanup error: {e}")
            return True
    
    return MegatronWorker


class MegatronWorkerGroup:
    """Manager for coordinated Megatron Ray workers.
    
    Handles synchronized creation, initialization, and execution
    of Megatron workers for distributed training.
    
    NOTE: PP > 1 is required for training with LoRA. PP=1 has gradient issues.
    """
    
    def __init__(
        self,
        world_size: int,
        tp_size: int = 1,
        pp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        master_addr: str = None,
        master_port: int = None,
    ):
        import ray
        import socket
        
        # Warn if PP=1 (known gradient issue)
        if pp_size == 1:
            print("[MegatronWorkerGroup] WARNING: PP=1 has known gradient issues. "
                  "Training loss may not decrease. Use PP > 1 for training.")
        
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.cp_size = cp_size
        self.ep_size = ep_size
        
        if master_addr is None:
            master_addr = ray.util.get_node_ip_address()
        if master_port is None:
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
        
        self.master_addr = master_addr
        self.master_port = master_port
        
        MegatronWorker = get_megatron_worker_class()
        self.workers = [
            MegatronWorker.remote(
                rank=rank,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                tp_size=tp_size,
                pp_size=pp_size,
                cp_size=cp_size,
                ep_size=ep_size,
            )
            for rank in range(world_size)
        ]
        print(f"[MegatronWorkerGroup] Created {world_size} workers (TP={tp_size}, PP={pp_size})")
        
    def init_all(self, model_config: Dict[str, Any] = None) -> List[bool]:
        """Initialize all workers."""
        import ray
        return ray.get([w.init.remote(model_config) for w in self.workers])
        
    def create_model_all(self, pretrained_model_name_or_path: str, **kwargs) -> List[bool]:
        """Create model on all workers."""
        import ray
        return ray.get([w.create_model.remote(pretrained_model_name_or_path, **kwargs) for w in self.workers])
        
    def add_lora_all(self, lora_config: Dict[str, Any]) -> List[bool]:
        """Add LoRA to all workers."""
        import ray
        return ray.get([w.add_lora.remote(lora_config) for w in self.workers])
        
    def set_optimizer_all(self, lr: float = 1e-4, **kwargs) -> List[bool]:
        """Set optimizer on all workers."""
        import ray
        return ray.get([w.set_optimizer.remote(lr, **kwargs) for w in self.workers])
        
    def forward_backward_all(self, batch: Dict[str, torch.Tensor]) -> List[float]:
        """Execute forward/backward on all workers."""
        import ray
        return ray.get([w.forward_backward.remote(batch) for w in self.workers])
        
    def step_all(self) -> List[bool]:
        """Optimizer step on all workers."""
        import ray
        return ray.get([w.step.remote() for w in self.workers])
        
    def cleanup_all(self) -> List[bool]:
        """Cleanup all workers."""
        import ray
        return ray.get([w.cleanup.remote() for w in self.workers])
        
    def shutdown(self):
        """Shutdown all workers."""
        import ray
        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception:
                pass
        self.workers = []
