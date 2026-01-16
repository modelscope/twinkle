"""
Multi-Tenant Megatron LoRA Training - Client Example.

Simple training loop using remote multi-tenant server.
Inspired by tinker-cookbook's minimal training scripts.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Training configuration."""
    server_url: str = "http://localhost:8080"
    lora_rank: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    log_every: int = 10


class TrainingClient:
    """
    Simple client for multi-tenant LoRA training.
    
    Example:
        >>> client = TrainingClient(server_url)
        >>> client.initialize(lora_rank=8, learning_rate=1e-4)
        >>> 
        >>> for batch in dataloader:
        ...     result = client.forward_backward(batch)
        ...     if client.should_step():
        ...         client.step()
        >>> 
        >>> client.finalize()
    """
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url.rstrip('/')
        self.tenant_id: Optional[str] = None
        self._session = requests.Session()
        self._accumulated = 0
        self._ga_steps = 1
    
    def _post(self, endpoint: str, **kwargs) -> Dict:
        """Make POST request."""
        headers = {"X-Tenant-ID": self.tenant_id} if self.tenant_id else {}
        resp = self._session.post(
            f"{self.server_url}{endpoint}",
            headers=headers,
            json=kwargs,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    
    def initialize(
        self,
        lora_rank: int = 8,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        **kwargs,
    ) -> str:
        """Initialize tenant on server."""
        result = self._post(
            "/initialize",
            lora_config={"r": lora_rank, "target_modules": "all-linear"},
            optimizer_kwargs={"lr": learning_rate},
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.tenant_id = result["tenant_id"]
        self._ga_steps = gradient_accumulation_steps
        logger.info(f"Initialized: {self.tenant_id}")
        return self.tenant_id
    
    def finalize(self):
        """Cleanup tenant."""
        if self.tenant_id:
            self._post("/finalize")
            logger.info(f"Finalized: {self.tenant_id}")
            self.tenant_id = None
    
    def forward_backward(self, inputs: Any) -> Dict:
        """Forward + backward pass."""
        result = self._post("/forward_backward", inputs=inputs)
        self._accumulated += 1
        return result.get("data", {})
    
    def should_step(self) -> bool:
        """Check if optimizer step should happen."""
        return self._accumulated >= self._ga_steps
    
    def step(self):
        """Optimizer step."""
        self._post("/finish_grad_sync")
        self._post("/clip_grad_norm")
        self._post("/step")
        self._post("/zero_grad")
        self._post("/lr_step")
        self._accumulated = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.finalize()


def main(config: Config):
    """Example training loop."""
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    client = TrainingClient(config.server_url)
    
    # Initialize
    client.initialize(
        lora_rank=config.lora_rank,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    try:
        # Training loop
        for step in range(100):
            start = time.time()
            
            # Create dummy batch (replace with your data loading)
            batch = {
                "input_ids": list(range(128)),
                "attention_mask": [1] * 128,
                "labels": list(range(128)),
            }
            
            # Forward + backward
            result = client.forward_backward(batch)
            
            # Optimizer step
            if client.should_step():
                client.step()
                
                if step % config.log_every == 0:
                    elapsed = time.time() - start
                    logger.info(f"Step {step}, time: {elapsed:.2f}s")
        
        logger.info("Training complete!")
        
    finally:
        client.finalize()


if __name__ == "__main__":
    main(Config())
