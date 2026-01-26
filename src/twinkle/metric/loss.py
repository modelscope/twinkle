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

    def __init__(self, device_mesh, process_group, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.total_loss = 0
        self.total_count = 0
        self.max_grad_norm = 0

    def accumulate(self, inputs, outputs):
        if 'loss' not in outputs:
            return
        loss = outputs["loss"]
        grad_norm = outputs.get("grad_norm")
        if grad_norm is not None and self.max_grad_norm < grad_norm:
            self.max_grad_norm = grad_norm

        self.total_loss += loss.item() if hasattr(loss, "item") else loss
        self.total_count += 1

    def reset(self):
        self.total_loss = 0
        self.total_count = 0
        self.max_grad_norm = 0

    def calculate(self):
        local_results = [
            {"loss": self.total_loss, "count": self.total_count, "max_grad_norm": self.max_grad_norm}
        ]

        all_results = torch_util.gather_object(local_results, self.device_mesh, self.process_group)

        total_loss = sum(r["loss"] for r in all_results)
        total_count = sum(r["count"] for r in all_results)
        max_grad_norm = max(r["max_grad_norm"] for r in all_results)
        avg_loss = total_loss / total_count if total_count > 0 else np.nan
        self.reset()
        results = {}
        if avg_loss is not None:
            results['loss'] = f'{avg_loss:.4f}'
        if max_grad_norm > 0:
            results['grad_norm'] = f'{max_grad_norm:.2f}'
        return results