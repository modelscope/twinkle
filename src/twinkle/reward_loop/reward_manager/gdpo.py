from .base import RewardManagerBase
from .registry import register


@register("gdpo")
class GDPORewardManager(RewardManagerBase):
    def __init__(self, *args, experiment_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_name = experiment_name

    async def run_single(self, item):
        result = await super().run_single(item)
        if self.experiment_name:
            result.reward_extra_info.setdefault("experiment_name", self.experiment_name)
        return result
