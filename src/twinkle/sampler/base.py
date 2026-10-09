# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from peft import PeftConfig
from typing import Any, List, Optional, Type, Union

import twinkle
from twinkle import remote_function
from twinkle.data_format import InputFeature, SampleResponse, SamplingParams, Trajectory, is_encoded
from twinkle.patch import Patch
from twinkle.template import Template
from twinkle.utils import construct_class


class Sampler(ABC):

    def __init__(self):
        self.engine = None
        self.template = None

    @abstractmethod
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[SamplingParams] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        return_encoded: bool = False,
        use_base_model: bool = False,
    ) -> List[SampleResponse]:
        """Sample responses for given inputs.

        Args:
            inputs: Either InputFeature(s) or Trajectory(s).
                - InputFeature: Must contain 'input_ids'. For multimodal, include 'images'/'videos'.
                - Trajectory: Must contain 'messages'. Requires template to be set.
            sampling_params: Sampling parameters.
            adapter_name: Optional LoRA adapter name.
            adapter_path: Optional saved LoRA checkpoint path.
            return_encoded: Include training input features in sampled sequences.
            use_base_model: Sample without a LoRA adapter.
            Set sampling_params.num_samples for multiple completions per prompt.

        Returns:
            SampleResponse containing sampled sequences.
            Total sequences = len(inputs) * sampling_params.num_samples.
        """
        pass

    @abstractmethod
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        ...

    @staticmethod
    def _not_encoded(inputs: Any) -> bool:
        """Check if inputs are not yet encoded (i.e., is Trajectory, not InputFeature).

        Delegates to the single shared predicate so the three backends and the wire
        schema cannot drift apart.
        """
        assert isinstance(inputs, dict), f'Expected dict, got {type(inputs)}'
        return not is_encoded(inputs)

    def _is_trajectory(self, inputs: Any) -> bool:
        """Check if inputs are Trajectory type (not encoded)."""
        if isinstance(inputs, list):
            if not inputs:
                return False
            inputs = inputs[0]
        if isinstance(inputs, dict):
            return self._not_encoded(inputs)
        return False

    def _normalize_inputs(self, inputs) -> List:
        if isinstance(inputs, dict):
            return [inputs]
        return list(inputs)

    def encode_trajectory(self,
                          trajectory: Trajectory,
                          adapter_name: str = '',
                          add_generation_prompt: bool = True) -> InputFeature:
        template = self.template
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")

        encoded = template.encode(trajectory, add_generation_prompt=add_generation_prompt)

        input_ids = encoded.get('input_ids')
        if input_ids is None:
            raise ValueError("Template.encode() must return 'input_ids'")
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()

        result = InputFeature(input_ids=input_ids)

        for key, value in encoded.items():
            if key not in ('input_ids', 'labels'):
                result[key] = value

        return result

    def decode_response(self, token_ids: List[int], adapter_name: str = '') -> str:
        """Decode token ids to text."""
        template = self.template
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        return template.decode(token_ids)

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        if 'model_id' not in kwargs and hasattr(self, 'model_id'):
            kwargs['model_id'] = self.model_id
        template = construct_class(template_cls, Template, twinkle.template, **kwargs)
        self.template = template

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig) -> None:
        raise NotImplementedError
