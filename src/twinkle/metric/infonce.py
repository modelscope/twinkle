# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
from twinkle import Platform
from .base import Metric


class InfoNCEMetric(Metric):

    def __init__(self,
                 cross_batch:bool=True,
                 hard_negatives:int=None,
                 ):
        """InfoNCE loss

        Args:
            cross_batch: If to calculate similarity across all samples in one batch.
            hard_negatives: Padding or cutting off negative samples to a fixed length. This will help to speed up calculation.
        """
        self.cross_batch = cross_batch
        self.hard_negatives = hard_negatives

    def __call__(self, inputs, outputs, **kwargs):
        import torch
        from twinkle.loss.infonce import InfoNCELoss
        logits = outputs['logits']
        labels = inputs['labels']
        world_size = Platform.get_world_size()
        if world_size > 1 and self.cross_batch:
            logits, labels = InfoNCELoss.gather_data(logits, labels)
        split_tensors = InfoNCELoss.parse_multi_negative_sentences(torch.tensor(logits), torch.tensor(labels), self.hard_negatives)
        split_tensors = [t.numpy() for t in split_tensors]
        can_batched = self.hard_negatives is not None
        if self.hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
            can_batched = True
        all_similarity_matrix = []
        all_labels = []
        pos_neg_margins = []
        if not self.cross_batch:
            if can_batched:
                sentences = np.stack(split_tensors, axis=0)
                similarity_matrix = np.matmul(sentences[:, 0:1], sentences[:, 1:].transpose((0, 2, 1))).squeeze(1)
                all_similarity_matrix.append(similarity_matrix)
                labels = np.zeros_like(similarity_matrix)
                labels[:, 0] = 1
                all_labels.append(labels)
            else:
                for tensor in split_tensors:
                    similarity_matrix = np.matmul(tensor[0], tensor[1:].T)
                    all_similarity_matrix.append(similarity_matrix)
                    labels = np.zeros_like(similarity_matrix)
                    labels[0] = 1
                    all_labels.append(labels)
                    max_neg_scores = np.max(similarity_matrix[labels == 0], axis=-1)
                    pos_neg_margins.append(np.mean(similarity_matrix[labels == 1] - max_neg_scores).item())
        else:
            if can_batched:
                sentences = np.stack(split_tensors, axis=0)
                similarity_matrix = np.matmul(sentences[:, 0], sentences[:, 1:].reshape(-1, sentences.shape[2]).T)
                all_similarity_matrix.append(similarity_matrix)
                labels = np.zeros_like(similarity_matrix)
                for row, col in enumerate(
                        range(0, sentences.shape[0] * (sentences.shape[1] - 1), sentences.shape[1] - 1)):
                    labels[row, col] = 1
                all_labels.append(labels)
            else:
                all_tensors = []
                for tensor in split_tensors:
                    all_tensors.append(tensor[1:])
                sentences = np.concatenate(all_tensors, axis=0)
                length = 0
                for idx, tensor in enumerate(split_tensors):
                    similarity_matrix = np.matmul(tensor[0], sentences.T)
                    all_similarity_matrix.append(similarity_matrix)
                    labels = np.zeros_like(similarity_matrix)
                    labels[length] = 1
                    all_labels.append(labels)
                    length += tensor.shape[0] - 1
                    max_neg_scores = np.max(similarity_matrix[labels == 0], axis=-1)
                    pos_neg_margins.append(np.mean(similarity_matrix[labels == 1] - max_neg_scores).item())

        similarity_matrix = np.concatenate(all_similarity_matrix, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        if can_batched:
            pos_scores = similarity_matrix[labels == 1].reshape(similarity_matrix.shape[0], -1)
            neg_scores = similarity_matrix[labels == 0].reshape(similarity_matrix.shape[0], -1)
            max_neg_scores = np.max(neg_scores, axis=-1)
            pos_neg_margin = np.mean(pos_scores - max_neg_scores).item()
        else:
            pos_scores = similarity_matrix[labels == 1]
            neg_scores = similarity_matrix[labels == 0]
            pos_neg_margin = np.mean(pos_neg_margins)

        mean_neg = np.mean(neg_scores)
        mean_pos = np.mean(pos_scores)
        return {'margin': pos_neg_margin, 'mean_neg': mean_neg, 'mean_pos': mean_pos}