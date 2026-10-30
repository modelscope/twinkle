# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import InputFeature, LossOutput, ModelOutput


class Loss:

    require_logits = False
    require_entropy = False
    require_logps = True
    require_values = False
    # This loss needs all local inputs in one evaluation to preserve its batch context.
    requires_full_batch = False

    def __call__(self, inputs: InputFeature, outputs: ModelOutput, **kwargs) -> LossOutput:
        ...

    def validate_micro_batching(self, num_micro_batches: int) -> None:
        """Validate the actual local plan; one micro-batch preserves the full batch context."""
        if num_micro_batches <= 1 or not self.requires_full_batch:
            return
        raise NotImplementedError(
            f'{type(self).__name__} requires the full local batch for its loss computation, but the '
            f'current batch would be split into {num_micro_batches} micro-batches. This loss does not '
            'support micro-batching, including dynamic batching. Increase micro_batch_size or the '
            'dynamic token budget so the full local batch fits in a single micro-batch.')

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
