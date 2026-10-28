# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import InputFeature, LossOutput, ModelOutput


class Loss:

    require_logits = False
    require_entropy = False
    require_logps = True
    # Full-sequence outputs required by this loss call; missing keys raise an
    # error in Transformers SP. output_key_specs defines their layout, while
    # this selector requests collection for the current objective only.
    sp_gather_keys = ()
    # Provider-owned layouts/routing for custom tensors, also consumed by the
    # processor when SP is disabled. Concrete losses import DataKeySpec from
    # data_format; the base class does not depend on an SP strategy.
    input_key_specs = ()
    output_key_specs = ()
    # Request per-token values from the Transformers wrapper (e.g. PPO critic).
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
