# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np

from .base import Metric
from twinkle import torch_util


class Accuracy(Metric):
    """The accuracy metric.

    Args:
        device_mesh: The device mesh
        process_group: The process group to collect data from
    """

    def __init__(self, device_mesh, process_group, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.total_correct = 0
        self.total_count = 0

    def accumulate(self, inputs, outputs):
        labels = inputs["labels"]
        logits = outputs["logits"]
        output_token_ids = logits.argmax(dim=-1)
        mask = labels != -100
        correct_mask = (output_token_ids == labels) & mask

        local_correct = correct_mask.sum().item()
        local_total = mask.sum().item()

        self.total_correct += local_correct
        self.total_count += local_total

    def calculate(self):
        local_results = [
            {"correct": self.total_correct, "total": self.total_count}
        ]

        all_results = torch_util.gather_object(local_results, self.device_mesh, self.process_group)

        total_correct = sum(r["correct"] for r in all_results)
        total_count = sum(r["total"] for r in all_results)
        accuracy = total_correct / total_count if total_count > 0 else np.nan

        return {
            "accuracy": accuracy,
            "correct_tokens": total_correct,
            "total_tokens": total_count,
        }
