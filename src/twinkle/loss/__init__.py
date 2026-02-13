# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
from .chunked_cross_entropy import ChunkedCrossEntropyLoss
from .cross_entropy import CrossEntropyLoss
from .grpo import BNPOLoss, CISPOLoss, DRGRPOLoss, GRPOLoss, GSPOLoss, SAPOLoss
from .mse import MSELoss
from .vocab_parallel_cross_entropy import VocabParallelCrossEntropyLoss

torch_loss_mapping = {
    'mse': MSELoss,
    'cross_entropy': CrossEntropyLoss,
    'chunked_cross_entropy': ChunkedCrossEntropyLoss,
    'vocab_parallel_cross_entropy': VocabParallelCrossEntropyLoss,
    # RL losses
    'grpo': GRPOLoss,
    'gspo': GSPOLoss,
    'sapo': SAPOLoss,
    'cispo': CISPOLoss,
    'bnpo': BNPOLoss,
    'dr_grpo': DRGRPOLoss,
}
