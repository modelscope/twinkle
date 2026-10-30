# Copyright (c) ModelScope Contributors. All rights reserved.


def count_loss_tokens(model_input, ignore_index: int) -> int:
    """Count non-ignored labels in one sequence or a nested packing unit."""
    # A packing unit stays intact until the processor expands it into sequences.
    if isinstance(model_input, (list, tuple)):
        return sum(count_loss_tokens(item, ignore_index) for item in model_input)
    labels = model_input["labels"]
    if hasattr(labels, "ne"):
        return int(labels.ne(ignore_index).sum().item())
    return sum(int(token != ignore_index) for token in labels)
