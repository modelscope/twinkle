# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import InputFeature, LossOutput, ModelOutput


class Loss:

    require_logits = False
    require_entropy = False
    require_logps = True
    # Sequence-aligned outputs this loss needs in full-sequence layout.
    # Keys must be declared in the model/loss output schema.
    sp_gather_keys = ()
    # A loss may declare additional sequence-aligned input/output contracts.
    # Keep the core loss API independent of the Transformers backend by
    # importing DataKeySpec only in concrete loss implementations.
    input_key_specs = ()
    output_key_specs = ()
    # Compatibility for existing loss plugins. New declarations use the common names.
    sp_input_key_specs = ()
    sp_output_key_specs = ()
    require_values = False

    def __call__(self, inputs: InputFeature, outputs: ModelOutput, **kwargs) -> LossOutput:
        ...

    def micro_batch_scale(self, inputs: list[InputFeature], indices: list[int]) -> float:
        if len(indices) == len(inputs):
            return 1.0
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support micro-batching, including dynamic batching')

    def batch_normalizer(self, inputs: list[InputFeature]):
        """Return this batch's additive reduction denominator, or ``None`` for mean-reduced losses."""
        return None
