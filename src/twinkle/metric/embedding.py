# Copyright (c) ModelScope Contributors. All rights reserved.
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
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F
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

        # Gather embeddings and labels across DP for in-batch stats.
        # NCCL ``all_gather`` requires every rank to send the same tensor size,
        # but ``slice_dp`` dispatch (``divmod`` split) can leave per-rank dim-0
        # uneven. Pad to the global max along dim-0, gather, then strip padding.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            assert sentences.shape[0] == labels.shape[0], (
                f'sentences/labels dim-0 mismatch: {sentences.shape[0]} vs {labels.shape[0]}')
            local_n = torch.tensor([sentences.shape[0]], device=sentences.device, dtype=torch.long)
            sizes = [torch.empty_like(local_n) for _ in range(world_size)]
            dist.all_gather(sizes, local_n, group=self.process_group)
            sizes_int = [int(s.item()) for s in sizes]
            max_n = max(sizes_int)

            def _pad_gather(tensor: 'torch.Tensor') -> 'List[torch.Tensor]':
                if tensor.shape[0] < max_n:
                    pad_shape = (max_n - tensor.shape[0], ) + tuple(tensor.shape[1:])
                    padded = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)
                else:
                    padded = tensor
                buffers = [torch.empty_like(padded) for _ in range(world_size)]
                dist.all_gather(buffers, padded.contiguous(), group=self.process_group)
                return buffers

            sent_buffers = _pad_gather(sentences)
            label_buffers = _pad_gather(labels)
            sentences = torch.cat([sent_buffers[i][:sizes_int[i]] for i in range(world_size)], dim=0)
            labels = torch.cat([label_buffers[i][:sizes_int[i]] for i in range(world_size)], dim=0)

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
