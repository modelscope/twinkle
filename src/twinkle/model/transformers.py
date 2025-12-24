from typing import Literal

from transformers import PreTrainedModel

from twinkle import remote_class


class Transformers:


@remote_class()
class AutoModelForCausalLM:

    def __init__(self, *, task_type: Literal['causal_lm'] = 'causal_lm', pretrained_model_name_or_path, **kwargs):
        from transformers import AutoModelForCausalLM as AutoModelForCausalLMTF
        self.model = AutoModelForCausalLMTF.from_pretrained(**kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> PreTrainedModel:
        raise NotImplementedError('Transformers does not implement from_pretrained')