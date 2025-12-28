from dataclasses import dataclass
from typing import List, Union

import numpy as np

InputType = Union[List[List[int]], List[int], np.ndarray, 'torch.Tensor']


@dataclass
class InputFeature:

    input_ids: InputType

    attention_mask: InputType

    position_ids: InputType

    def to_transformers_dict(self):
        return {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'position_ids': self.position_ids,
        }

    def to_megatron_dict(self):
        ...
