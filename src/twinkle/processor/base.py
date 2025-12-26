from typing import Optional, Union
from twinkle import DeviceMesh
from twinkle.template import Template


class InputProcessor:

    def __init__(self, device_mesh: Optional[DeviceMesh] = None, template: Optional[Union[Template, str]] = None):
        self.device_mesh = device_mesh
        self.template = template()

    def __call__(self, inputs):
        inputs = self.dispatch(inputs)
        if self.template is not None:
            inputs = self.template.encode(inputs)
        return self.prepare_inputs(inputs)

    def dispatch(self, inputs):
        if self.device_mesh is None:
            return inputs

        dp_rank = self.device_mesh.dp_rank
        dp_world_size = self.device_mesh.dp_world_size

        total_items = len(inputs)
        if total_items < dp_world_size:
            last_item = inputs[-1] if inputs else None
            inputs = inputs + [last_item] * (dp_world_size - total_items)
            total_items = dp_world_size

        # Calculate data slice for current rank
        items_per_rank = total_items // dp_world_size
        remainder = total_items % dp_world_size

        # Distribute remainder items to first 'remainder' ranks
        if dp_rank < remainder:
            start_idx = dp_rank * (items_per_rank + 1)
            end_idx = start_idx + items_per_rank + 1
        else:
            start_idx = remainder * (items_per_rank + 1) + (dp_rank - remainder) * items_per_rank
            end_idx = start_idx + items_per_rank

        return inputs[start_idx:end_idx]
        

    def prepare_inputs(self, inputs):
        ...

    def collate_fn(self, inputs):
        ...
