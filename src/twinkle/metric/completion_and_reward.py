# Copyright (c) ModelScope Contributors. All rights reserved.
import statistics
from typing import Any, Dict, List

from .base import Metric


class CompletionRewardMetric(Metric):

    def __init__(self, device_mesh=None, process_group=None, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.generate_time: List[float] = []
        self.weight_sync_time: List[float] = []
        self.rewards: Dict[str, List[float]] = {}
        self.completion_lengths: List[int] = []

    def reset(self):
        self.generate_time = []
        self.weight_sync_time = []
        self.rewards = {}
        self.completion_lengths = []

    def accumulate(
            self,
            inputs=None,  # ignore
            outputs=None,  # ignore
            *,
            rewards=None,
            completion_lengths=None,
            generate_time: float = None,
            weight_sync_time: float = None,
            **kwargs):
        if completion_lengths is None:
            completion_lengths = []
        if rewards is None:
            rewards = {}
        for key, value in rewards.items():
            if key not in self.rewards:
                self.rewards[key] = []
            self.rewards[key].extend(value)

        self.completion_lengths.extend(completion_lengths)
        if generate_time is not None:
            self.generate_time.append(generate_time)
        if weight_sync_time is not None:
            self.weight_sync_time.append(weight_sync_time)

    @staticmethod
    def _mean(statistic_list: List[float]) -> float:
        return sum(statistic_list) / len(statistic_list) if len(statistic_list) > 0 else -1.0

    @staticmethod
    def _std(statistic_list: List[float]) -> float:
        if len(statistic_list) > 1:
            return statistics.stdev(statistic_list)
        return 0.0

    def calculate(self) -> Dict[str, Any]:
        metric_dict = {}
        if self.weight_sync_time:
            metric_dict['profiling/Time taken: move_model_to_sampler'] = self._mean(self.weight_sync_time)
        if self.generate_time:
            metric_dict['profiling/Time taken: generate'] = self._mean(self.generate_time)
        for key, values in self.rewards.items():
            metric_dict[f'train/{key}_reward'] = self._mean(values)
            metric_dict[f'train/{key}_reward_std'] = self._std(values)

        if self.completion_lengths:
            metric_dict['train/completion_length'] = self._mean(self.completion_lengths)
        return metric_dict
