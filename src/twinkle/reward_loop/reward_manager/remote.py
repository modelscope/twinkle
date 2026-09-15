from .naive import NaiveRewardManager
from .registry import register


@register("remote")
class RemoteRewardManager(NaiveRewardManager):
    """Manager hook for CPU-isolated execution."""
    pass
