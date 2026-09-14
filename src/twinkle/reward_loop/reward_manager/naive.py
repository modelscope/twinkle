from .base import RewardManagerBase
from .registry import register


@register("naive")
class NaiveRewardManager(RewardManagerBase):
    pass
