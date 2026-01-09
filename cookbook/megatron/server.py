# Copyright (c) twinkle authors. All rights reserved.
"""Megatron LoRA training server.

This server hosts the Megatron model and handles training requests from clients.

Usage:
    python cookbook/megatron/server.py --port 8000 --tp_size 2
"""
import argparse
from typing import Any, Dict

import numpy as np

import twinkle
from twinkle import get_logger, DeviceMesh, DeviceGroup, Platform
from twinkle.model import MegatronModel
from twinkle.loss import CrossEntropyLoss
from twinkle.processor import InputProcessor

logger = get_logger()


class MegatronModelServer:
    """Server wrapper for Megatron model."""
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.is_initialized = False
    
    def initialize_model(self, model_name: str, lora_config: Dict[str, Any] = None):
        """Initialize the Megatron model with optional LoRA configuration.
        
        Args:
            model_name: HuggingFace model name or path.
            lora_config: Optional LoRA configuration dict.
        """
        logger.info(f'Initializing model: {model_name}')
        
        self.model = MegatronModel(
            pretrained_model_name_or_path=model_name,
            tensor_model_parallel_size=self.args.tp_size,
            sequence_parallel=self.args.sequence_parallel,
            mixed_precision=self.args.mixed_precision,
        )
        
        if lora_config:
            from peft import LoraConfig
            config = LoraConfig(**lora_config)
            self.model.add_adapter_to_model(
                'default',
                config,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            )
        
        self.model.set_template('Qwen3Template')
        self.model.set_processor(InputProcessor, padding_side='right')
        self.model.set_loss(CrossEntropyLoss)
        
        self.is_initialized = True
        logger.info('Model initialized successfully')
        
        return {'status': 'success', 'message': 'Model initialized'}
    
    def set_optimizer(self, optimizer_type: str = 'AdamW', **kwargs):
        """Set optimizer for the model."""
        if not self.is_initialized:
            return {'status': 'error', 'message': 'Model not initialized'}
        
        from torch.optim import AdamW, SGD
        optimizer_map = {'AdamW': AdamW, 'SGD': SGD}
        
        if optimizer_type not in optimizer_map:
            return {'status': 'error', 'message': f'Unknown optimizer: {optimizer_type}'}
        
        self.model.set_optimizer(optimizer_map[optimizer_type], **kwargs)
        return {'status': 'success', 'message': f'Optimizer {optimizer_type} set'}
    
    def set_lr_scheduler(self, scheduler_type: str = 'CosineAnnealingLR', **kwargs):
        """Set learning rate scheduler."""
        if not self.is_initialized:
            return {'status': 'error', 'message': 'Model not initialized'}
        
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR
        scheduler_map = {
            'CosineAnnealingLR': CosineAnnealingLR,
            'LinearLR': LinearLR,
            'StepLR': StepLR,
        }
        
        if scheduler_type not in scheduler_map:
            return {'status': 'error', 'message': f'Unknown scheduler: {scheduler_type}'}
        
        self.model.set_lr_scheduler(scheduler_map[scheduler_type], **kwargs)
        return {'status': 'success', 'message': f'Scheduler {scheduler_type} set'}
    
    def train_step(self, batch: Dict[str, Any]):
        """Execute one training step.
        
        Args:
            batch: Input batch data.
            
        Returns:
            Training step result with loss.
        """
        if not self.is_initialized:
            return {'status': 'error', 'message': 'Model not initialized'}
        
        # Forward-backward pass
        loss = self.model.forward_backward(inputs=batch)
        
        # Optimizer step
        self.model.clip_grad_norm(self.args.max_grad_norm)
        self.model.step()
        self.model.zero_grad()
        self.model.lr_step()
        
        return {'status': 'success', 'loss': float(loss) if loss else None}
    
    def save_checkpoint(self, output_path: str):
        """Save model checkpoint.
        
        Args:
            output_path: Path to save checkpoint.
        """
        if not self.is_initialized:
            return {'status': 'error', 'message': 'Model not initialized'}
        
        self.model.save(output_path)
        return {'status': 'success', 'message': f'Checkpoint saved to {output_path}'}
    
    def get_train_configs(self):
        """Get current training configuration."""
        if not self.is_initialized:
            return {'status': 'error', 'message': 'Model not initialized'}
        
        return {'status': 'success', 'configs': self.model.get_train_configs()}


def create_device_mesh(args) -> DeviceMesh:
    """Create device mesh for Megatron parallelism."""
    mesh = np.arange(args.nproc_per_node).reshape(args.dp_size, args.tp_size)
    
    device_mesh = DeviceMesh(
        device_type='cuda',
        mesh=mesh,
        mesh_dim_names=('dp', 'tp'),
    )
    return device_mesh


def create_device_group(args):
    """Create device group for model placement."""
    device_group = [
        DeviceGroup(
            name='model',
            ranks=list(range(args.nproc_per_node)),
            device_type=Platform.get_platform().device_prefix(),
        )
    ]
    return device_group


def parse_args():
    parser = argparse.ArgumentParser(description='Megatron Model Server')
    
    # Server arguments
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='Server port')
    
    # Parallelism arguments
    parser.add_argument('--nproc_per_node', type=int, default=4,
                        help='Number of processes per node')
    parser.add_argument('--tp_size', type=int, default=2,
                        help='Tensor parallel size')
    parser.add_argument('--dp_size', type=int, default=2,
                        help='Data parallel size')
    parser.add_argument('--sequence_parallel', action='store_true',
                        help='Enable sequence parallelism')
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision mode')
    
    # Training defaults
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize distributed environment
    device_mesh = create_device_mesh(args)
    device_group = create_device_group(args)
    
    twinkle.initialize(
        mode='local',
        nproc_per_node=args.nproc_per_node,
        groups=device_group,
        global_device_mesh=device_mesh,
        lazy_collect=False,
    )
    
    # Create model server
    server = MegatronModelServer(args)
    
    # Start HTTP server
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.error('Flask not installed. Install with: pip install flask')
        return
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy'})
    
    @app.route('/initialize', methods=['POST'])
    def initialize():
        data = request.json
        result = server.initialize_model(
            model_name=data.get('model_name'),
            lora_config=data.get('lora_config'),
        )
        return jsonify(result)
    
    @app.route('/set_optimizer', methods=['POST'])
    def set_optimizer():
        data = request.json
        result = server.set_optimizer(**data)
        return jsonify(result)
    
    @app.route('/set_lr_scheduler', methods=['POST'])
    def set_lr_scheduler():
        data = request.json
        result = server.set_lr_scheduler(**data)
        return jsonify(result)
    
    @app.route('/train_step', methods=['POST'])
    def train_step():
        data = request.json
        result = server.train_step(batch=data.get('batch', {}))
        return jsonify(result)
    
    @app.route('/save', methods=['POST'])
    def save():
        data = request.json
        result = server.save_checkpoint(output_path=data.get('output_path'))
        return jsonify(result)
    
    @app.route('/configs', methods=['GET'])
    def configs():
        result = server.get_train_configs()
        return jsonify(result)
    
    logger.info(f'Starting server on {args.host}:{args.port}')
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == '__main__':
    main()

