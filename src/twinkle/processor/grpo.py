# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Optional

from twinkle import DeviceMesh, remote_class, remote_function
from twinkle.data_format import InputFeature
from twinkle.processor import InputProcessor


@remote_class()
class GRPOLossProcessor(InputProcessor):
    """
    Processor for preparing inputs required by GRPOLoss.
    
    This processor computes intermediate variables from raw data (input_ids, labels):
    - completion_mask: Boolean mask for completion tokens
    - logits_to_keep: Number of completion tokens to keep
    - num_items_in_batch: Total completion tokens (for DAPO/CISPO normalization)
    
    The processor infers completion positions from labels:
    - labels == -100: prompt tokens (ignored in loss)
    - labels != -100: completion tokens (used in loss)
    """
    
    def __init__(self, device_mesh: Optional[DeviceMesh] = None, ignore_index: int = -100, **kwargs):
        super(GRPOLossProcessor, self).__init__(device_mesh)
        self.ignore_index = ignore_index

    @remote_function()
    def prepare_inputs(self, inputs: InputFeature) -> InputFeature:
        """Compute GRPO-specific fields from labels."""
        labels = inputs.labels
        # Compute logits_to_keep: maximum completion length across batch
        non_ignored = (labels != self.ignore_index).int()
        first_non_ignored = non_ignored.argmax(dim=-1)
        seq_len = labels.shape[-1]
        logits_to_keep = (seq_len - first_non_ignored).max().item()

        # Create completion mask for the last logits_to_keep positions
        completion_mask = (labels[:, -logits_to_keep:] != self.ignore_index).float()

        # Compute num_items_in_batch: total completion tokens
        num_items_in_batch = completion_mask.sum().int().item()

        # Update inputs with GRPO-specific fields
        inputs.completion_mask = completion_mask
        inputs.logits_to_keep = logits_to_keep
        inputs.num_items_in_batch = num_items_in_batch

        return super().prepare_inputs(inputs)
