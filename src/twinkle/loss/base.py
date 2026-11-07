# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Optional

from twinkle.data_format import InputFeature, LossOutput, ModelOutput


class Loss:

    require_logits = False
    require_entropy = False
    require_logps = True
    require_values = False
    # Consecutive samples that must stay together, in order, on one DP rank when the model reorders a
    # batch to balance DP workloads (2 for interleaved chosen/rejected pairs). None forbids reordering.
    dp_balance_group_size: Optional[int] = 1

    def __call__(self, inputs: InputFeature, outputs: ModelOutput, **kwargs) -> LossOutput:
        ...

    def validate_inputs(self, inputs, **kwargs) -> None:
        """Reject a batch before any rank starts its forward pass.

        Loss implementations whose alignment rules can fail per sample override
        this hook. The default accepts every batch.
        """
        return None

    def micro_batch_scale(self, inputs: list[InputFeature], indices: list[int]) -> float:
        if len(indices) == len(inputs):
            return 1.0
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support micro-batching, including dynamic batching')

    def micro_batch_kwargs(self, kwargs, indices, batch_size):
        """Keep sample-aligned loss inputs in the same order as the selected model inputs."""
        from twinkle.model.micro_batch import select_batch

        return {key: select_batch(value, indices, batch_size) for key, value in kwargs.items()}

    def batch_normalizer(self, inputs: list[InputFeature]):
        """Return this batch's additive reduction denominator, or ``None`` for mean-reduced losses."""
        return None


def dp_balance_group_size(loss) -> Optional[int]:
    """The ``dp_balance_group_size`` of ``loss``; a plain callable may depend on order, so it gets None."""
    if loss is None:
        return 1
    return loss.dp_balance_group_size if isinstance(loss, Loss) else None
