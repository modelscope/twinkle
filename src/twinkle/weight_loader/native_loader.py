from twinkle.model.base import TwinkleModel
from twinkle.sampler import Sampler
from .base import WeightLoader


class NativeLoader(WeightLoader):

    def __call__(self, module: TwinkleModel, sampler: Sampler, adapter_name=''):
        state_dict = module.get_state_dict(adapter_name=adapter_name)
        sampler.sync_weights(state_dict, adapter_name=adapter_name)
