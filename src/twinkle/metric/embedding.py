# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import List, Union

from twinkle.data_format import InputFeature, ModelOutput
from .base import Metric


class EmbeddingMetric(Metric):
    """Embedding similarity metric for InfoNCE training.

    Reports anchor-positive cosine similarity stats (mean/min/max) and
    average anchor-to-other-positives (in-batch negative) similarity.
    Performs an extra all_gather to compute cross-rank statistics.
    """

    def __init__(self, device_mesh, process_group, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.reset()

    def reset(self):
        self.pos_sim_sum = 0.0
        self.pos_sim_min = float('inf')
        self.pos_sim_max = float('-inf')
        self.pos_count = 0
        self.neg_sim_sum = 0.0
        self.neg_count = 0
        self.total_loss = 0.0
        self.total_count = 0
        self.grad_norm = 0.0

    def accumulate(self, inputs: Union[InputFeature, List[InputFeature]], outputs: ModelOutput, **kwargs):
        sentences = outputs.get('embeddings')
        if sentences is None:
            sentences = outputs.get('logits')
        if sentences is None:
            return
        if sentences.dim() == 3:
            sentences = sentences[:, 0]

        if not isinstance(inputs, list):
            inputs = [inputs]
        labels = torch.cat([inp['labels'].view(-1) for inp in inputs], dim=0)

        # Gather embeddings and labels across DP for in-batch stats
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            local_shape = sentences.new_tensor(sentences.shape, dtype=torch.long)
            shapes = [torch.empty_like(local_shape) for _ in range(world_size)]
            dist.all_gather(shapes, local_shape)
            all_sentences = [sentences.new_empty(s.tolist()) for s in shapes]
            dist.all_gather(all_sentences, sentences.contiguous())
            sentences = torch.cat(all_sentences, dim=0)

            local_lshape = labels.new_tensor(labels.shape, dtype=torch.long)
            lshapes = [torch.empty_like(local_lshape) for _ in range(world_size)]
            dist.all_gather(lshapes, local_lshape)
            all_labels = [labels.new_empty(s.tolist()) for s in lshapes]
            dist.all_gather(all_labels, labels.contiguous())
            labels = torch.cat(all_labels, dim=0)

        anchor_idx = torch.nonzero(labels, as_tuple=False).squeeze(-1)
        if anchor_idx.numel() == 0:
            return

        anchors = sentences[anchor_idx]
        positives = sentences[anchor_idx + 1]

        # Anchor-positive cosine similarity
        pos_cos = F.cosine_similarity(anchors, positives, dim=1)
        self.pos_sim_sum += pos_cos.sum().item()
        self.pos_sim_min = min(self.pos_sim_min, pos_cos.min().item())
        self.pos_sim_max = max(self.pos_sim_max, pos_cos.max().item())
        self.pos_count += pos_cos.numel()

        # Anchor vs all other positives (in-batch negatives)
        if anchors.size(0) > 1:
            sim_matrix = torch.matmul(anchors, positives.T)
            mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
            neg_sims = sim_matrix[mask]
            self.neg_sim_sum += neg_sims.sum().item()
            self.neg_count += neg_sims.numel()

        loss = outputs.get('loss')
        if loss is not None:
            self.total_loss += loss.item() if hasattr(loss, 'item') else loss
            self.total_count += 1
        grad_norm = kwargs.get('grad_norm')
        if grad_norm is not None:
            self.grad_norm = grad_norm

    def calculate(self):
        results = {}
        if self.pos_count > 0:
            results['pos_sim'] = f'{self.pos_sim_sum / self.pos_count:.4f}'
            results['pos_sim_min'] = f'{self.pos_sim_min:.4f}'
            results['pos_sim_max'] = f'{self.pos_sim_max:.4f}'
        if self.neg_count > 0:
            results['neg_sim'] = f'{self.neg_sim_sum / self.neg_count:.4f}'
        if self.total_count > 0:
            results['loss'] = f'{self.total_loss / self.total_count:.4f}'
        if self.grad_norm > 0:
            results['grad_norm'] = f'{self.grad_norm:.6f}'
        self.reset()
        return results
