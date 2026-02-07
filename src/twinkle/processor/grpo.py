# Copyright (c) ModelScope Contributors. All rights reserved.
"""
GRPO Processor for RL training.

This processor is a simple pass-through that uses the base InputProcessor.
The GRPO loss now operates on logps directly and computes loss_mask from labels,
so no special preprocessing is needed.

This file is kept for backward compatibility but can be replaced with InputProcessor.
"""
from typing import Optional

from twinkle import DeviceMesh, remote_class
from twinkle.processor import InputProcessor

@remote_class()
class GRPOLossProcessor(InputProcessor):
    """
    Processor for GRPO training.
    
    This is now a thin wrapper around InputProcessor since the GRPO loss
    computes loss_mask directly from labels. It exists for backward compatibility
    and can be used interchangeably with InputProcessor.
    
    The GRPO loss expects:
    - inputs['labels']: [batch, seq_len] target tokens, -100 for ignored positions
    - outputs['logps']: [batch, seq_len] log probabilities from current policy
    
    These are provided by the standard template encoding and model forward.
    """
    
    def __init__(self, device_mesh: Optional[DeviceMesh] = None, **kwargs):
        super(GRPOLossProcessor, self).__init__(device_mesh=device_mesh, **kwargs)
