# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
from .chunked_cross_entropy import ChunkedCrossEntropyLoss
from .cross_entropy import CrossEntropyLoss
from .dpo import CPOLoss, DPOLoss, ORPOLoss, SimPOLoss
from .gkd import GKDLoss
from .grpo import BNPOLoss, CISPOLoss, DRGRPOLoss, GRPOLoss, GSPOLoss, SAPOLoss
from .mse import MSELoss

torch_loss_mapping = {
    'mse': MSELoss,
    'chunked_cross_entropy': ChunkedCrossEntropyLoss,
    'cross_entropy': CrossEntropyLoss,
    # KD losses
    'gkd': GKDLoss,
    # RL losses
    'grpo': GRPOLoss,
    'gspo': GSPOLoss,
    'sapo': SAPOLoss,
    'cispo': CISPOLoss,
    'bnpo': BNPOLoss,
    'dr_grpo': DRGRPOLoss,
    # DPO family losses
    'dpo': DPOLoss,
    'simpo': SimPOLoss,
    'cpo': CPOLoss,
    'orpo': ORPOLoss,
}
