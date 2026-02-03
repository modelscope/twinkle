# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np

from .base import Metric
from twinkle import torch_util


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

    def accumulate(self, inputs, outputs):
        if 'loss' not in outputs:
            return
        loss = outputs["loss"]
        if self.loss_reduction == 'sum':
            # `Transformers` models may use reduction=sum, to average grads before step
            labels = inputs["labels"]
            self.num_tokens += (labels >= 0).sum().item()
        grad_norm = outputs.get("grad_norm")
        if grad_norm is not None:
            self.grad_norm = grad_norm

        self.total_loss += loss.item() if hasattr(loss, "item") else loss
        self.total_count += 1

    def reset(self):
        self.total_loss = 0
        self.total_count = 0
        self.grad_norm = 0
        self.num_tokens = 0

    def calculate(self):
        local_results = [
            {"loss": self.total_loss, "count": self.total_count, "grad_norm": self.grad_norm, "num_tokens": self.num_tokens}
        ]

        all_results = self.gather_results(local_results)

        total_loss = sum(r["loss"] for r in all_results)
        total_count = sum(r["count"] for r in all_results)
        grad_norm = max(r["grad_norm"] for r in all_results)
        num_tokens = sum(r["num_tokens"] for r in all_results)
        if num_tokens > 0:
            avg_loss = total_loss / num_tokens
        else:
            avg_loss = total_loss / total_count
        self.reset()
        results = {}
        if avg_loss is not None:
            results['loss'] = f'{avg_loss:.4f}'
        if grad_norm > 0:
            results['grad_norm'] = f'{grad_norm:.2f}'
        return results