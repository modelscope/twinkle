from typing import Optional

from twinkle import DeviceMesh


class InputProcessor:

    def __init__(self, device_mesh: Optional[DeviceMesh] = None):
        self.device_mesh = device_mesh

    def __call__(self, inputs):
        return self.prepare_task_inputs(self.redistribute(inputs))

    def redistribute(self, inputs):
        return inputs

    def prepare_task_inputs(self, inputs):
        return inputs

    def collate_fn(self, inputs):
        return inputs
