# Copyright (c) twinkle authors. All rights reserved.
"""Megatron LoRA training client.

This client sends training requests to the Megatron model server.

Usage:
    # First start the server:
    python cookbook/megatron/server.py --port 8000
    
    # Then run the client:
    python cookbook/megatron/client.py --server_url http://localhost:8000
"""
import argparse
from typing import Any, Dict, Optional

import requests

from twinkle import get_logger
from twinkle.dataset import Dataset, DatasetMeta

logger = get_logger()


class MegatronModelClient:
    """Client for remote Megatron model training."""
    
    def __init__(self, server_url: str, timeout: int = 300):
        """Initialize client.
        
        Args:
            server_url: URL of the model server.
            timeout: Request timeout in seconds.
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
    
    def _request(self, endpoint: str, method: str = 'POST', data: Dict = None) -> Dict:
        """Send request to server.
        
        Args:
            endpoint: API endpoint.
            method: HTTP method.
            data: Request data.
            
        Returns:
            Response data.
        """
        url = f'{self.server_url}/{endpoint}'
        
        try:
            if method == 'GET':
                response = requests.get(url, timeout=self.timeout)
            else:
                response = requests.post(url, json=data or {}, timeout=self.timeout)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f'Request failed: {e}')
            return {'status': 'error', 'message': str(e)}
    
    def health_check(self) -> bool:
        """Check if server is healthy.
        
        Returns:
            True if server is healthy.
        """
        result = self._request('health', method='GET')
        return result.get('status') == 'healthy'
    
    def initialize_model(
        self,
        model_name: str,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Initialize model on server.
        
        Args:
            model_name: HuggingFace model name or path.
            lora_config: Optional LoRA configuration.
            
        Returns:
            Server response.
        """
        return self._request('initialize', data={
            'model_name': model_name,
            'lora_config': lora_config,
        })
    
    def set_optimizer(self, optimizer_type: str = 'AdamW', **kwargs) -> Dict:
        """Set optimizer on server.
        
        Args:
            optimizer_type: Optimizer type name.
            **kwargs: Optimizer arguments.
            
        Returns:
            Server response.
        """
        return self._request('set_optimizer', data={
            'optimizer_type': optimizer_type,
            **kwargs,
        })
    
    def set_lr_scheduler(self, scheduler_type: str = 'CosineAnnealingLR', **kwargs) -> Dict:
        """Set learning rate scheduler on server.
        
        Args:
            scheduler_type: Scheduler type name.
            **kwargs: Scheduler arguments.
            
        Returns:
            Server response.
        """
        return self._request('set_lr_scheduler', data={
            'scheduler_type': scheduler_type,
            **kwargs,
        })
    
    def train_step(self, batch: Dict[str, Any]) -> Dict:
        """Execute one training step.
        
        Args:
            batch: Input batch data.
            
        Returns:
            Server response with loss.
        """
        return self._request('train_step', data={'batch': batch})
    
    def save_checkpoint(self, output_path: str) -> Dict:
        """Save model checkpoint.
        
        Args:
            output_path: Path to save checkpoint.
            
        Returns:
            Server response.
        """
        return self._request('save', data={'output_path': output_path})
    
    def get_train_configs(self) -> Dict:
        """Get training configuration from server.
        
        Returns:
            Training configuration.
        """
        return self._request('configs', method='GET')


def create_dataset(args):
    """Create and preprocess dataset."""
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset))
    dataset.set_template('Qwen3Template', model_id=args.model_name)
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Megatron Model Client')
    
    # Server arguments
    parser.add_argument('--server_url', type=str, default='http://localhost:8000',
                        help='Model server URL')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Request timeout in seconds')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='ms://Qwen/Qwen2.5-7B-Instruct',
                        help='HuggingFace model name or path')
    parser.add_argument('--output_dir', type=str, default='./output/megatron_lora',
                        help='Output directory for checkpoints')
    
    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    parser.add_argument('--target_modules', type=str, default='all-linear',
                        help='Target modules for LoRA')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum training steps')
    parser.add_argument('--save_steps', type=int, default=50,
                        help='Checkpoint save interval')
    parser.add_argument('--log_steps', type=int, default=10,
                        help='Logging interval')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ms://modelscope/competition_math',
                        help='Dataset name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create client
    client = MegatronModelClient(
        server_url=args.server_url,
        timeout=args.timeout,
    )
    
    # Health check
    if not client.health_check():
        logger.error('Server is not available')
        return
    
    logger.info('Server is healthy, initializing model...')
    
    # Initialize model with LoRA
    lora_config = {
        'r': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'target_modules': args.target_modules,
    }
    
    result = client.initialize_model(
        model_name=args.model_name,
        lora_config=lora_config,
    )
    
    if result.get('status') != 'success':
        logger.error(f'Failed to initialize model: {result}')
        return
    
    logger.info('Model initialized, setting optimizer...')
    
    # Set optimizer and scheduler
    client.set_optimizer(optimizer_type='AdamW', lr=args.learning_rate, weight_decay=0.01)
    client.set_lr_scheduler(scheduler_type='CosineAnnealingLR', T_max=args.max_steps)
    
    # Print training configuration
    configs = client.get_train_configs()
    logger.info(f'Training configs: {configs}')
    
    # Create dataset and dataloader
    logger.info('Loading dataset...')
    dataset = create_dataset(args)
    
    # Training loop
    logger.info('Starting training...')
    global_step = 0
    
    for step, batch in enumerate(dataset.iter(batch_size=args.batch_size)):
        if global_step >= args.max_steps:
            break
        
        # Send batch to server for training
        result = client.train_step(batch)
        
        if result.get('status') != 'success':
            logger.error(f'Training step failed: {result}')
            continue
        
        global_step += 1
        
        # Log progress
        if global_step % args.log_steps == 0:
            loss = result.get('loss', 'N/A')
            logger.info(f'Step {global_step}, Loss: {loss}')
        
        # Save checkpoint
        if global_step % args.save_steps == 0:
            save_result = client.save_checkpoint(
                f'{args.output_dir}/checkpoint-{global_step}'
            )
            logger.info(f'Checkpoint saved: {save_result}')
    
    # Save final model
    client.save_checkpoint(args.output_dir)
    logger.info(f'Training completed. Model saved to {args.output_dir}')


if __name__ == '__main__':
    main()

