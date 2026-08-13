# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
from .chunked_cross_entropy import ChunkedCrossEntropyLoss
from .cross_entropy import CrossEntropyLoss
from .dpo import CPOLoss, DPOLoss, ORPOLoss, SimPOLoss
from .gkd import GKDLoss
from .grpo import BNPOLoss, CISPOLoss, DRGRPOLoss, GRPOLoss, GSPOLoss, SAPOLoss
from .infonce import ContrastiveLoss, CosineSimilarityLoss, EmbeddingLoss, InfonceLoss, OnlineContrastiveLoss
from .liger_fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from .liger_fused_linear_grpo import LigerFusedLinearGRPOLoss
from .mse import MSELoss
from .reranker import ListwiseRerankerLoss, PointwiseRerankerLoss
from .seq_cls import SeqClsLoss

torch_loss_mapping = {
    'mse': MSELoss,
    'chunked_cross_entropy': ChunkedCrossEntropyLoss,
    'cross_entropy': CrossEntropyLoss,
    'liger_fused_linear_cross_entropy': LigerFusedLinearCrossEntropyLoss,
    'liger_fused_linear_grpo': LigerFusedLinearGRPOLoss,
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
