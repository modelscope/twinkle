from dataclasses import dataclass
from typing import List, Union, Optional

import numpy as np

InputType = Union[List[List[int]], List[int], np.ndarray, 'torch.Tensor']


@dataclass
class InputFeature:

    input_ids: InputType

    attention_mask: InputType = None

    position_ids: InputType = None

    labels: InputType = None

    completion_mask: InputType = None

    logits_to_keep: Optional[int] = None

    num_items_in_batch: Optional[int] = None

    def to_transformers_dict(self):
        return {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'position_ids': self.position_ids,
        }

    def to_megatron_dict(self):
        raise NotImplementedError()
