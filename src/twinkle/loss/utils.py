# Copyright (c) ModelScope Contributors. All rights reserved.


def count_loss_tokens(model_input, ignore_index: int, *, loss_mask_fn=None) -> int:
    """Count scored tokens in one sequence or packing unit, optionally using the loss's own mask."""
    # A packing unit stays intact until the processor expands it into sequences.
    if isinstance(model_input, (list, tuple)):
        return sum(count_loss_tokens(item, ignore_index, loss_mask_fn=loss_mask_fn) for item in model_input)
    labels = model_input["labels"]
    if loss_mask_fn is not None:
        import torch

        labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        return int(loss_mask_fn(model_input, labels).sum().item())
    if hasattr(labels, "ne"):
        return int(labels.ne(ignore_index).sum().item())
    return sum(int(token != ignore_index) for token in labels)
