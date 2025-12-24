from transformers import AutoTokenizer, PreTrainedTokenizer

from twinkle.trajectory import Trajectory


class Template:

    def __init__(self, model_id: str):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    def encode(self, trajectory: Trajectory):
        ...