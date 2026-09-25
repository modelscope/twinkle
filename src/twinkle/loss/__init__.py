# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
from .channel import ChannelLoss
from .chunked_cross_entropy import ChunkedCrossEntropyLoss
from .cross_entropy import CrossEntropyLoss
from .dpo import CPOLoss, DPOLoss, ORPOLoss, SimPOLoss
from .gkd import GKDLoss
from .grpo import BNPOLoss, CISPOLoss, DRGRPOLoss, GRPOLoss, GSPOLoss, PPOLoss, SAPOLoss
from .infonce import ContrastiveLoss, CosineSimilarityLoss, EmbeddingLoss, InfonceLoss, OnlineContrastiveLoss
from .liger_fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from .liger_fused_linear_grpo import LigerFusedLinearGRPOLoss
from .mse import MSELoss
from .opsd import OPSDLoss
from .reranker import ListwiseRerankerLoss, PointwiseRerankerLoss
from .reward import RewardLoss
from .seq_cls import SeqClsLoss
from .value import PPOValueLoss

torch_loss_mapping = {
    'mse': MSELoss,
    'channel': ChannelLoss,
    'chunked_cross_entropy': ChunkedCrossEntropyLoss,
    'cross_entropy': CrossEntropyLoss,
    'liger_fused_linear_cross_entropy': LigerFusedLinearCrossEntropyLoss,
    'liger_fused_linear_grpo': LigerFusedLinearGRPOLoss,
    # KD losses
    'gkd': GKDLoss,
    # RL losses
    'grpo': GRPOLoss,
    'ppo': PPOLoss,
    'ppo_value': PPOValueLoss,
    'gspo': GSPOLoss,
    'sapo': SAPOLoss,
    'cispo': CISPOLoss,
    'bnpo': BNPOLoss,
    'dr_grpo': DRGRPOLoss,
    # Self-distillation losses
    'opsd': OPSDLoss,
    # DPO family losses
    'dpo': DPOLoss,
    'simpo': SimPOLoss,
    'cpo': CPOLoss,
    'orpo': ORPOLoss,
    # Reward model (pairwise Bradley-Terry)
    'reward': RewardLoss,
    'rm': RewardLoss,
    # Embedding / contrastive losses
    'infonce': InfonceLoss,
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    # Reranker (cross-encoder) losses
    'pointwise_reranker': PointwiseRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
    # Sequence classification
    'seq_cls': SeqClsLoss,
}
