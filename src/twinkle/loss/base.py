# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import InputFeature, LossOutput, ModelOutput


class Loss:

    require_logits = False
    require_entropy = False
    require_logps = True
    # Sequence-aligned outputs this loss needs in full-sequence layout.
    # Keys must have an output DataKeySpec registered by SP or a model adapter.
    sp_gather_keys = ()
    # A loss may declare additional sequence-aligned input/output contracts.
    # Keep the core loss API independent of the Transformers backend by
    # importing DataKeySpec only in concrete loss implementations.
    sp_input_key_specs = ()
    sp_output_key_specs = ()

    def __call__(self, inputs: InputFeature, outputs: ModelOutput, **kwargs) -> LossOutput:
        ...
