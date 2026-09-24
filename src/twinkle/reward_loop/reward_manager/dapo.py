from .base import RewardManagerBase
from .registry import register


@register("dapo")
class DAPORewardManager(RewardManagerBase):
    def __init__(self, *args, reward_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_kwargs = reward_kwargs or {}

    async def run_single(self, item):
        result = await super().run_single(item)
        limit = self.reward_kwargs.get("max_response_length")
        penalty = self.reward_kwargs.get("overlong_penalty", 0.0)
        length = len(item.response_ids) if item.response_ids is not None else len(item.solution_str)
        if limit is not None and length > limit:
            result.reward_score -= penalty
            result.reward_extra_info.update({"overlong": True, "response_length": length})
        return result
