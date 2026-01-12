# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.model import TwinkleModel
from twinkle.sampler import Sampler
from .base import WeightLoader


class NativeLoader(WeightLoader):

    def __call__(self, model: TwinkleModel, sampler: Sampler, adapter_name=''):
        state_dict = model.get_state_dict(adapter_name=adapter_name)
        sampler.sync_weights(state_dict, adapter_name=adapter_name)
