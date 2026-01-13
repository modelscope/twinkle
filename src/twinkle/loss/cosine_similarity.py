# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
import torch


class CosineSimilarityLoss(Loss):

    def __call__(self, inputs, outputs, **kwargs):
        sentence1 = outputs['sentence1']
        sentence2 = outputs['sentence2']
        labels = inputs['labels']
        cos_score_transformation = torch.nn.Identity()
        loss_fct = torch.MSELoss()
        output = cos_score_transformation(torch.cosine_similarity(sentence1, sentence2))
        return loss_fct(output, labels.to(output.dtype).view(-1))