# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, Any, List

from twinkle import torch_util


class Metric:

    def __init__(self, device_mesh, process_group, **kwargs):
        self.process_group = process_group
        self.device_mesh = device_mesh

    def accumulate(self, inputs, outputs):
        ...

    def calculate(self):
        ...

    def reset(self):
        ...

    def gather_results(self, local_results: List[Dict[str, Any]]):
        if self.device_mesh is not None and self.process_group is not None:
            all_results = torch_util.gather_object(local_results, self.device_mesh, self.process_group)
        else:
            all_results = local_results
        return all_results
