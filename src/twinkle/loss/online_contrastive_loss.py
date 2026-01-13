# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from .base import Loss
from .contrastive_loss import SiameseDistanceMetric


class OnlineContrastiveLoss(Loss):

    def __call__(self, inputs, outputs, **kwargs):
        sentence1 = outputs['sentence1']
        sentence2 = outputs['sentence2']
        labels = inputs['labels']
        distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        distance_matrix = distance_metric(sentence1, sentence2)
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        margin = 0.5
        negative_loss = torch.nn.functional.relu(margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss