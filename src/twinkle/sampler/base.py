# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Dict, Any

from twinkle.data_format import Trajectory

class Sampler:

    def __init__(self):
        pass

    def sample(self, trajectories: List[Trajectory], adapter_name = '')-> List[Trajectory]:
        ...

    def add_adapter_to_sampler(self, adapter_name: str, config):
        ...

    def sync_weights(self, state_dict: Dict[str, Any], adapter_name=''):
        pass