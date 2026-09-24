from dataclasses import dataclass


@dataclass
class RewardLoopMetrics:
    submitted_batches: int = 0
    collected_batches: int = 0
    submit_time: float = 0.0
    collect_wait_time: float = 0.0
    reward_time: float = 0.0
    max_backlog: int = 0

    @property
    def overlap_ratio(self):
        denominator = self.submit_time + self.reward_time
        return 1.0 - self.collect_wait_time / denominator if denominator else 0.0

    def snapshot(self):
        return dict(self.__dict__, overlap_ratio=self.overlap_ratio)

    def reset(self):
        for field in self.__dataclass_fields__:
            setattr(self, field, 0)
