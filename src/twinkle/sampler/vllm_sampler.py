# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import uuid
from dataclasses import dataclass, asdict
from typing import List, Type, Dict, Any, Union

from peft import PeftConfig

from .base import Sampler
import twinkle
from twinkle import remote_function, remote_class, DeviceMesh, Plugin, requires
from twinkle.data_format import Trajectory, Message
from twinkle.patch.vllm_lora_weights import VLLMLoraWeights, TensorLoRARequest
from twinkle.processor import InputProcessor
from twinkle.template import Template
from ..utils import construct_class


@dataclass
class SampleGroup:
    adapter_name: str = None
    adapter_config: PeftConfig = None
    template: Template = None
    processor: InputProcessor = None
    lora_int_id: int = None


@remote_class()
class VLLMSampler(Sampler):
    """A vLLM sampler.

    Args:
        model_id: The model id for inference.
        engine_args: Engine args in dict, which is needed by `vllm.EngineArgs`.
        device_mesh: vLLM device mesh
    """

    _default_adapter_name = ''

    def __init__(self, model_id: str, engine_args: Dict[str, Any], device_mesh: DeviceMesh=None, **kwargs):
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'
        super().__init__()
        requires('vllm')
        from vllm import LLMEngine, EngineArgs
        engine_args = EngineArgs(**engine_args)
        vllm_config = engine_args.create_engine_config()
        self.engine = LLMEngine.from_vllm_config(
            vllm_config=vllm_config,
        )
        self.model_id = model_id
        self.device_mesh = device_mesh
        self.sample_group: Dict[str, SampleGroup] = {self._default_adapter_name: SampleGroup()}
        VLLMLoraWeights()(self)

    def _check_adapter_valid(self, adapter_name: str):
        assert adapter_name in self.sample_group, f'Use a valid {adapter_name} first, current is: {adapter_name}'

    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        template = construct_class(template_cls, Template, twinkle.template, **kwargs)
        self.sample_group[adapter_name].template = template

    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        processor = construct_class(processor_cls, InputProcessor, twinkle.processor, **kwargs)
        self.sample_group[adapter_name].processor = processor

    @remote_function()
    def sample(self, trajectories: List[Trajectory], adapter_name = '') -> List[Trajectory]:
        self._check_adapter_valid(adapter_name)
        if adapter_name:
            group = self.sample_group[adapter_name]
            from vllm.lora.request import LoRARequest
            adapter_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=group.lora_int_id,
                lora_path='dummy_lora_path',
            )
        else:
            adapter_request = None

        request_ids = []
        template_ins = self.sample_group[adapter_name].template
        for trajectory in trajectories:
            inputs = template_ins.encode(trajectory)
            request_id = str(uuid.uuid4().hex)
            request_ids.append(request_id)
            llm_inputs = {'prompt_token_ids': inputs.input_ids}
            self.engine.add_request(request_id, llm_inputs, generation_config=trajectory.generation_config, adapter_request=adapter_request)
        outputs = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs[output.request_id] = output
        outputs = [outputs[request_id] for request_id in request_ids]
        for trajectory, output in zip(trajectories, outputs):
            response = template_ins.decode(output.token_ids)
            trajectory.messages.append(Message(role='assistant', content=response))
        return trajectories

    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig):
        assert adapter_name not in self.sample_group, f'{adapter_name} already exists.'
        self.sample_group[adapter_name] = SampleGroup()
        self.sample_group[adapter_name].adapter_name = adapter_name
        self.sample_group[adapter_name].adapter_config = config

    def sync_weights(self, state_dict: Dict[str, Any], adapter_name='') -> None:
        if not adapter_name:
            llm_model = self.engine.inner_model
            llm_model.load_weights(state_dict.items())
        else:
            self._check_adapter_valid(adapter_name)
            group = self.sample_group[adapter_name]
            lora_request = TensorLoRARequest(
                lora_name=adapter_name,
                lora_int_id=group.lora_int_id,
                lora_path='dummy_lora_path',
                peft_config=asdict(group.adapter_config),
                lora_tensors=state_dict,
            )
            # TODO Replace lora
            self.engine.engine.add_lora(lora_request)

    def remove_adapter(self, adapter_name: str):
        if adapter_name and adapter_name in self.sample_group:
            self.sample_group.pop(adapter_name)
        # TODO Remove lora
