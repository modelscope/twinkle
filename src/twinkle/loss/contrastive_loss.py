# Copyright (c) ModelScope Contributors. All rights reserved.
from enum import Enum
from .base import Loss
import torch


# Code borrowed from sentence_transformers
class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)  # noqa
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)  # noqa
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)  # noqa


class ContrastiveLoss(Loss):

    def __call__(self, inputs, outputs, **kwargs):
        sentence1 = outputs['sentence1']
        sentence2 = outputs['sentence2']
        labels = inputs['labels']
        distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        distances = distance_metric(sentence1, sentence2)
        margin = 0.5
        labels = labels.to(sentence1.dtype)
        losses = 0.5 * (labels * distances.pow(2) + (1 - labels) * torch.nn.functional.relu(margin - distances).pow(2))
        return losses.mean()