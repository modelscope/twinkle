# Copyright (c) ModelScope Contributors. All rights reserved.

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