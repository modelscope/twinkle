from transformers import AutoTokenizer, PreTrainedTokenizer

from twinkle.trajectory import Trajectory


class Template:

    def __init__(self, model_id: str, **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    def encode(self, trajectory: Trajectory):
        ...

    def check(self, trajectory: Trajectory):
        encoded = None
        try:
            encoded = self.encode(trajectory)
            if not encoded:
                return None
            else:
                return trajectory
        except Exception as e: # noqa
            return None
        finally:
            if encoded:
                del encoded
