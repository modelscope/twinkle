from twinkle.model.base import TwinkleModel
from twinkle.sampler import Sampler
from twinkle.weight_syncronizer.base import WeightSynchronizer


class VanillaSynchronizer(WeightSynchronizer):

    def __call__(self, module: TwinkleModel, sampler: Sampler, adapter_name=''):
        state_dict = module.get_state_dict(adapter_name=adapter_name)
        sampler.sync_weights(state_dict, adapter_name=adapter_name)
