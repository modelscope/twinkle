# Copyright (c) ModelScope Contributors. All rights reserved.
from .mse import MSELoss
from .contrastive_loss import ContrastiveLoss
from .online_contrastive_loss import OnlineContrastiveLoss
from .infonce import InfoNCELoss
from .cross_entropy import CrossEntropyLoss
from .chunked_cross_entropy import ChunkedCrossEntropyLoss, ChunkedCrossEntropyLossFunc
from .cosine_similarity import CosineSimilarityLoss
from .generative_reranker import GenerativeRerankerLoss
from .reranker import RerankerLoss
from .listwise_reranker import ListwiseRerankerLoss
from .listwise_generative_reranker import ListwiseGenerativeRerankerLoss
from .grpo import GRPOLoss
from .base import Loss

torch_loss_mapping = {
    'mse': MSELoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfoNCELoss,
    'cross_entropy': CrossEntropyLoss,
    'chunked_cross_entropy': ChunkedCrossEntropyLoss,
    'cosine_similarity': CosineSimilarityLoss,
    'generative_reranker': GenerativeRerankerLoss,
    'reranker': RerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
    'listwise_generative_reranker': ListwiseGenerativeRerankerLoss,
    'grpo': GRPOLoss,
}