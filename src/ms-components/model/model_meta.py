from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelMeta:

    library: Literal['transformers', 'megatron']

    framework: Literal['torch'] = 'torch'

