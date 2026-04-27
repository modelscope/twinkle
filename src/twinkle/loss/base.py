# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import InputFeature, LossOutput, ModelOutput


class Loss:

    require_logits = False

    def __call__(self, inputs: InputFeature, outputs: ModelOutput, **kwargs) -> LossOutput:
        ...
