from typing import Dict, Any

from twinkle.processor import InputProcessor


class GRPOInputProcessor(InputProcessor):

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
