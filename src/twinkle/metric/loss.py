# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Union

import torch.distributed as dist

from transformers.utils import is_torch_npu_available

from twinkle import Platform
from twinkle.data_format import InputFeature, ModelOutput
from .base import Metric


class LossMetric(Metric):
    """The loss metric.

    Args:
        device_mesh: The device mesh
        process_group: The process group to collect data from
    """

    def __init__(self, device_mesh, process_group, loss_reduction='mean', **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.total_loss = 0
        self.total_count = 0
        self.grad_norm = 0
        self.num_tokens = 0
        self.loss_reduction = loss_reduction

    def accumulate(self, inputs: Union[InputFeature, List[InputFeature]], outputs: ModelOutput, **kwargs):
        if 'loss' not in outputs:
            return
        loss = outputs['loss']
        if self.loss_reduction == 'sum':
            if not isinstance(inputs, list):
                inputs = [inputs]
            for input in inputs:
                # `Transformers` models may use reduction=sum, to average grads before step
                labels = input['labels']
                self.num_tokens += (labels >= 0).sum().item()
        grad_norm = kwargs.get('grad_norm')
        if grad_norm is not None:
            self.grad_norm = grad_norm

        self.total_loss += loss.item() if hasattr(loss, 'item') else loss
        self.total_count += 1

    def reset(self):
        self.total_loss = 0
        self.total_count = 0
        self.grad_norm = 0
        self.num_tokens = 0

    def calculate(self):
        total_loss = float(self.total_loss)
        total_count = float(self.total_count)
        grad_norm = float(self.grad_norm)
        num_tokens = float(self.num_tokens)

        if self.device_mesh is not None and self.process_group is not None and dist.is_initialized():
            is_npu = is_torch_npu_available()
            if is_npu:
                # On NPU/HCCL, tensor all_reduce is more stable than all_gather_object.
                import torch
                device = Platform.get_local_device()
                stats = torch.tensor([total_loss, total_count, num_tokens], dtype=torch.float64, device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.process_group)
                total_loss, total_count, num_tokens = stats.tolist()

                grad_tensor = torch.tensor([grad_norm], dtype=torch.float64, device=device)
                dist.all_reduce(grad_tensor, op=dist.ReduceOp.MAX, group=self.process_group)
                grad_norm = grad_tensor.item()
            else:
                local_results = [{
                    'loss': total_loss,
                    'count': total_count,
                    'grad_norm': grad_norm,
                    'num_tokens': num_tokens
                }]
                all_results = self.gather_results(local_results)
                total_loss = sum(r['loss'] for r in all_results)
                total_count = sum(r['count'] for r in all_results)
                grad_norm = max(r['grad_norm'] for r in all_results)
                num_tokens = sum(r['num_tokens'] for r in all_results)
        if num_tokens > 0:
            avg_loss = total_loss / num_tokens
        else:
            avg_loss = total_loss / total_count
        self.reset()
        results = {}
        if avg_loss is not None:
            results['loss'] = f'{avg_loss:.4f}'
        if grad_norm > 0:
            results['grad_norm'] = f'{grad_norm:.6f}'
        return results
