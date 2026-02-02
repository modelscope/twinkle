# Copyright (c) ModelScope Contributors. All rights reserved.
from .datum import datum_to_input_feature
from .transformers_model import TwinkleCompatTransformersModel
from twinkle.utils import exists, requires

if exists('megatron_core'):
    from .megatron_model import TwinkleCompatMegatronModel
else:
    class TwinkleCompatMegatronModel:  # pragma: no cover - only used when megatron_core is missing
        def __init__(self, *args, **kwargs):
            requires('megatron_core')
