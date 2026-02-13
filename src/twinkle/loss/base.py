# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import InputFeature, ModelOutput


class Loss:

    def __call__(self, inputs: InputFeature, outputs: ModelOutput, **kwargs):
        ...
