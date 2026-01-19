# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np

from .base import Metric
from twinkle import torch_util


class LossMetric(Metric):

    def __init__(self, device_mesh, process_group, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.total_loss = 0
        self.total_count = 0

    def accumulate(self, inputs, outputs):
        if 'loss' not in outputs:
            return
        loss = outputs["loss"]

        self.total_loss += loss.item() if hasattr(loss, "item") else loss
        self.total_count += 1

    def calculate(self):
        local_results = [
            {"loss": self.total_loss, "count": self.total_count}
        ]

        all_results = torch_util.gather_object(local_results, self.device_mesh, self.process_group)

        total_loss = sum(r["loss"] for r in all_results)
        total_count = sum(r["count"] for r in all_results)
        avg_loss = total_loss / total_count if total_count > 0 else np.nan

        return {
            "loss": avg_loss,
        }