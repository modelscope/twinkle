# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Type
from dataclasses import dataclass

from peft import PeftConfig

from twinkle.data_format import Trajectory, InputFeature
from twinkle.template import Template
from twinkle.processor import InputProcessor
from twinkle.sampler.types import SamplingParams, SampleResponse
import twinkle

from twinkle.utils import construct_class


@dataclass
class SampleGroup:
    adapter_name: str = None
    adapter_config: PeftConfig = None
    template: Template = None
    processor: InputProcessor = None
    # LoRA info for vLLM weight sync
    lora_int_id: int = None
    lora_ready: bool = False


class Sampler(ABC):
    _default_adapter_name = ''

    def __init__(self):
        self.engine = None
        self.template = None
        self.sample_group: Dict[str, SampleGroup] = {
            self._default_adapter_name: SampleGroup()
        }

    @abstractmethod
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[SamplingParams] = None,
        adapter_name: str = '',
    ) -> SampleResponse:
        """Sample responses for given inputs.
        
        Args:
            inputs: Either InputFeature(s) or Trajectory(s).
                - InputFeature: Must contain 'input_ids'. For multimodal, include 'images'/'videos'.
                - Trajectory: Must contain 'messages'. Requires template to be set.
            sampling_params: Sampling parameters.
            adapter_name: Optional LoRA adapter name.
            
        Returns:
            SampleResponse containing sampled sequences.
        """
        pass

    @staticmethod
    def _not_encoded(inputs: Any) -> bool:
        """Check if inputs are not yet encoded (i.e., is Trajectory, not InputFeature).
        
        Aligned with TransformersModel._not_encoded for consistency.
        """
        assert isinstance(inputs, dict), f"Expected dict, got {type(inputs)}"
        return 'input_ids' not in inputs and 'input_embedding' not in inputs

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

    def _check_adapter_valid(self, adapter_name: str):
        assert adapter_name in self.sample_group, \
            f'Invalid adapter_name: {adapter_name}. Available: {list(self.sample_group.keys())}'

    def _get_template(self, adapter_name: str = '') -> Optional[Template]:
        if adapter_name and adapter_name in self.sample_group:
            template = self.sample_group[adapter_name].template
            if template is not None:
                return template
        return self.template

    def encode_trajectory(self, trajectory: Trajectory, adapter_name: str = '', 
                          add_generation_prompt: bool = True) -> InputFeature:
        template = self._get_template(adapter_name)
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
        template = self._get_template(adapter_name)
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        return template.decode(token_ids)

    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        template = construct_class(template_cls, Template, twinkle.template, **kwargs)
        self.sample_group[adapter_name].template = template
        if adapter_name == '' or self.template is None:
            self.template = template

    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        processor = construct_class(processor_cls, InputProcessor, twinkle.processor, **kwargs)
        self.sample_group[adapter_name].processor = processor

    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig) -> None:
        if adapter_name in self.sample_group and adapter_name != self._default_adapter_name:
            return
        
        if adapter_name not in self.sample_group:
            self.sample_group[adapter_name] = SampleGroup()
        
        group = self.sample_group[adapter_name]
        group.adapter_name = adapter_name
        group.adapter_config = config
        
        used_ids = [g.lora_int_id for g in self.sample_group.values() if g.lora_int_id is not None]
        group.lora_int_id = (max(used_ids) + 1) if used_ids else 1
        group.lora_ready = False

    def sync_weights(self, state_dict: Dict[str, Any], adapter_name: str = '') -> None:
        pass

    def remove_adapter(self, adapter_name: str) -> None:
        if adapter_name and adapter_name in self.sample_group:
            self.sample_group.pop(adapter_name)
