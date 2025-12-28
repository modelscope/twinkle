from typing import Union, List

from twinkle.data_format import InputFeature
from twinkle.processor import InputProcessor


class GRPOInputProcessor(InputProcessor):

    def __call__(self, inputs: Union[InputFeature, List[InputFeature]]):
        return super().prepare_inputs(inputs)
