# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
from ..utils import torch_util
import numpy as np
import torch


class InfoNCELoss(Loss):

    def __init__(self,
                 temperature:float=0.1,
                 cross_batch:bool=True,
                 hard_negatives:int=None,
                 mask_fake_negative:bool=False,
                 fake_negative_margin:float=0.1,
                 include_qq:bool=False,
                 include_dd:bool=False,
                 ):
        """InfoNCE loss

        Args:
            temperature: The temperature before calculating similarity, default `0.1`. The loss will be smoother if the temperature is smaller.
            cross_batch: If to calculate similarity across all samples in one batch.
            hard_negatives: Padding or cutting off negative samples to a fixed length. This will help to speed up calculation.
            mask_fake_negative: If to mask negative samples if the similarities are big enough
            fake_negative_margin: Used with `mask_fake_negative`, If the negative similarities are big than pos-similarities+fake_negative_margin, the sample will be masked.
            include_qq: Whether to include query-query similarities.
            include_dd: Whether to include document-document similarities.
        """
        self.temperature = temperature
        self.cross_batch = cross_batch
        self.hard_negatives = hard_negatives
        self.mask_fake_negative = mask_fake_negative
        self.fake_negative_margin = fake_negative_margin
        self.include_qq = include_qq
        self.include_dd = include_dd

    @staticmethod
    def parse_multi_negative_sentences(sentences, labels, hard_negatives=None):
        split_indices = torch.nonzero(labels, as_tuple=False).squeeze().tolist()
        if isinstance(split_indices, int):
            split_indices = [split_indices]
        split_indices.append(len(labels))
        split_indices = np.array(split_indices) + np.array(list(range(len(split_indices))))
        split_tensors = []

        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]
            split_part = sentences[start:end]
            if hard_negatives is not None:
                negatives = len(split_part) - 2
                assert negatives > 0
                if negatives > hard_negatives:
                    split_part = split_part[:hard_negatives + 2]
                elif negatives < hard_negatives:
                    selected = np.random.choice(list(range(negatives)), size=hard_negatives - negatives, replace=True)
                    selected += 1  # skip positive
                    split_part = torch.cat((split_part, split_part[selected]), dim=0)
            split_tensors.append(split_part)
        return split_tensors

    @staticmethod
    def gather_data(logits, labels):
        # gather all the logits and labels across the gpus when calculate loss across all batches of all gpus
        from accelerate.utils import gather_object
        rank = torch_util.get_rank()
        all_preds = gather_object(logits.unsqueeze(0))
        labels = gather_object(labels)
        # override the gathered one
        all_preds[rank] = logits
        for idx in range(len(all_preds)):
            if idx == rank:
                continue
            # we don't calculate grad from other gpus
            all_preds[idx] = all_preds[idx].detach().to(logits.device)
        logits = torch.cat(all_preds, dim=0)
        labels = [tensor.to(logits.device) for tensor in labels]
        labels = torch.stack(labels, dim=0)
        return logits, labels

    def _calculate_batched_loss(self, split_tensors):
        # negative numbers are equal
        # [B, neg+2, D]
        sentences = torch.stack(split_tensors, dim=0)
        # [B, 1, D] * [B, neg+1, D]
        similarity_matrix = torch.matmul(sentences[:, 0:1], sentences[:, 1:].transpose(1, 2)) / self.temperature
        # The positive one is the first element
        labels = torch.zeros(len(split_tensors), dtype=torch.int64).to(sentences.device)
        return torch.nn.CrossEntropyLoss()(similarity_matrix.squeeze(1), labels)

    def _calculate_looped_loss(self, split_tensors):
        loss = 0
        # the negative numbers may be different, use for loop
        for tensor in split_tensors:
            # [D] * [neg+1, D]
            similarity_matrix = torch.matmul(tensor[0], tensor[1:].T) / self.temperature
            # The positive one is the first element
            labels = torch.tensor(0).to(tensor.device)
            loss += torch.nn.CrossEntropyLoss()(similarity_matrix, labels)
        # avg between all batches in one gpu
        loss /= len(split_tensors)
        return loss

    def _calculate_batched_loss_across_device(self, split_tensors):
        # [B, neg+2, D]
        sentences = torch.stack(split_tensors, dim=0)
        # base q->d similarities (includes own positive and all in-batch documents)
        queries = sentences[:, 0].squeeze(1)  # [B, D]
        docs_all = sentences[:, 1:].reshape(-1, sentences.size(2))  # [B*(neg+1), D]
        qd_matrix = torch.matmul(queries, docs_all.T)  # [B, B*(neg+1)]
        # target indices: start of each group's document block (its positive)
        labels = torch.tensor(range(0,
                                    sentences.size(0) * (sentences.size(1) - 1),
                                    sentences.size(1) - 1)).view(-1).to(sentences.device)

        logits_list = [qd_matrix]

        if self.include_qq:
            # q->q similarities; exclude self via -inf on diagonal to avoid accidental positives
            qq_matrix = torch.matmul(queries, queries.T)  # [B, B]
            qq_matrix = qq_matrix.clone()
            qq_matrix.fill_diagonal_(float('-inf'))
            logits_list.append(qq_matrix)

        if self.include_dd:
            # d+ -> d (doc-doc) similarities; exclude self-positive column per row
            pos_docs = sentences[:, 1].squeeze(1)  # [B, D]
            dd_matrix = torch.matmul(pos_docs, docs_all.T)  # [B, B*(neg+1)]
            # mask self positive per row: column index = row_idx * (neg+1)
            block = sentences.size(1) - 1  # (neg+1)
            if block > 0:
                row_idx = torch.arange(dd_matrix.size(0), device=dd_matrix.device)
                col_idx = row_idx * block
                dd_matrix[row_idx, col_idx] = float('-inf')
            logits_list.append(dd_matrix)

        if self.mask_fake_negative:
            # thresholds derived from positive q->d scores per row
            row_idx = torch.arange(qd_matrix.size(0), device=qd_matrix.device)
            pos_scores = qd_matrix[row_idx, labels]
            thresholds = pos_scores.view(-1, 1).detach() + self.fake_neg_margin

            # qd block mask
            qd_block = qd_matrix.clone()
            qd_mask = qd_block > thresholds
            qd_block[qd_mask] = float('-inf')

            components = [qd_block]

            # qq block mask (if present)
            if self.include_qq:
                qq_block = qq_matrix.clone()
                qq_mask = qq_block > thresholds
                qq_block[qq_mask] = float('-inf')
                # diagonal already masked unconditionally at construction time
                components.append(qq_block)

            # dd block (if present): self-positive column already masked unconditionally
            if self.include_dd:
                # align with Qwen3-Embedding, no threshold masking for d-d
                components.append(dd_matrix)

            similarity_matrix = torch.cat(components, dim=1)
        else:
            # concatenate all components without masking
            similarity_matrix = torch.cat(logits_list, dim=1)
        # temperature scaling and CE
        similarity_matrix = similarity_matrix / self.temperature
        return torch.nn.CrossEntropyLoss()(similarity_matrix, labels)

    def _calculate_looped_loss_across_device(self, split_tensors):
        all_tensors = []
        loss = 0
        for tensor in split_tensors:
            all_tensors.append(tensor[1:])
        # cat all neg+1 tensors
        sentences = torch.cat(all_tensors, dim=0)
        # prepare query anchors list if q-q is included
        if self.include_qq:
            queries_all = torch.stack([t[0] for t in split_tensors], dim=0)  # [B, D]
        length = 0
        for idx, tensor in enumerate(split_tensors):
            # [D] * [B*(neg+1), D], neg numbers are different
            qd_vec = torch.matmul(tensor[0], sentences.T)
            target = torch.tensor(length).to(tensor.device)
            logits_parts = []

            # compute threshold from positive q->d score
            threshold = (qd_vec[target].detach() + self.fake_neg_margin)

            # qd part with masking
            if self.mask_fake_negative:
                qd_masked = torch.where(qd_vec > threshold, torch.tensor(float('-inf'), device=qd_vec.device),
                                        qd_vec)
            else:
                qd_masked = qd_vec
            logits_parts.append(qd_masked)

            # qq part
            if self.include_qq:
                qq_vec = torch.matmul(tensor[0], queries_all.T)  # [B]
                # exclude self
                qq_vec = qq_vec.clone()
                qq_vec[idx] = float('-inf')
                if self.mask_fake_negative:
                    qq_vec = torch.where(qq_vec > threshold, torch.tensor(float('-inf'), device=qq_vec.device),
                                         qq_vec)
                logits_parts.append(qq_vec)

            # dd part
            if self.include_dd:
                dd_vec = torch.matmul(tensor[1], sentences.T)  # [B*(neg+1)]
                # mask self positive column for this row only (no threshold masking for d-d)
                block = split_tensors[idx].size(0) - 1  # (neg+1) for this group
                dd_vec[length] = float('-inf')
                logits_parts.append(dd_vec)

            logits_row = torch.cat(logits_parts, dim=-1)
            logits_row = logits_row / self.temperature
            loss += torch.nn.CrossEntropyLoss()(logits_row.unsqueeze(0), target.unsqueeze(0))
            # next positive is neg+1
            length += tensor.size(0) - 1
        loss /= len(split_tensors)
        return loss

    def __call__(self, inputs, outputs, **kwargs):
        """Calculate loss

        Args:
            logits: shape [Length * Hidden_size], Length=Repeat of (anchor(1)+positive(1)+negatives(n))
            labels: shape [Length], Length= Repeat of (positive(1)+negatives(n))
            For example:
                logits = [[0.1],    [0.25],    [0.15], ...   [0.2],    [0.3],    [0.14]]
                labels = [           1,         0,                      1,         0]
                         anchor   positive     negative     anchor   positive    negative

        Returns:
            The loss tensor
        """
        logits = outputs['logits']
        labels = inputs['labels']
        world_size = torch_util.get_world_size()
        if world_size > 1 and self.cross_batch:
            logits, labels = self._gather_data(logits, labels)

        # split tensors into single sample
        # Example: batch_size=2 with tensor anchor(1)+positive(1)+negatives(3) + anchor(1)+positive(1)+negatives(2)
        # labels will be [1,0,0,0,1,0,0], meaning 1 positive, 3 negatives, 1 positive, 2 negatives
        split_tensors = self.parse_multi_negative_sentences(logits, labels, self.hard_negatives)
        can_batched = self.hard_negatives is not None
        if self.hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
            # all tensors have the same batch size
            can_batched = True
        if not self.cross_batch:
            # only calculate loss inside one sample
            if can_batched:
                loss = self._calculate_batched_loss(split_tensors)
            else:
                loss = self._calculate_looped_loss(split_tensors)
        else:
            if can_batched:
                loss = self._calculate_batched_loss_across_device(split_tensors)
            else:
                loss = self._calculate_looped_loss_across_device(split_tensors)
        return loss
