import os
import uuid
from dataclasses import dataclass, asdict
from typing import List, Type, Dict, Any, Union, Callable

from peft import PeftConfig

from .base import Sampler
from twinkle import remote_function, remote_class
from twinkle.utils.plugin import Plugin
from twinkle.trajectory import Trajectory, Message
from twinkle import requires
from twinkle import template
from twinkle import processor
from twinkle.patch.vllm_lora_weights import VLLMLoraWeights, TensorLoRARequest
from ..processor import InputProcessor
from ..template import Template


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'


@dataclass
class SampleGroup:
    adapter_name: str = None
    adapter_config: PeftConfig = None
    template: Template = None
    processor: InputProcessor = None
    lora_int_id: int = None


@remote_class()
class VLLMSampler(Sampler):

    _default_adapter_name = ''

    def __init__(self, model_id: str, engine_args: Dict[str, Any], template: Type[template.Template], remote_group):
        super().__init__()
        requires('vllm')
        from vllm import LLMEngine, EngineArgs
        engine_args = EngineArgs(**engine_args)
        vllm_config = engine_args.create_engine_config()
        self.engine = LLMEngine.from_vllm_config(
            vllm_config=vllm_config,
        )
        self.model_id = model_id
        self.template = template(model_id)
        self.sample_group: Dict[str, SampleGroup] = {self._default_adapter_name: SampleGroup()}
        VLLMLoraWeights()(self)

    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.sample_group, f'Add {adapter_name} first before training.'
        if isinstance(template_cls, str):
            if hasattr(template, template_cls):
                template_cls = getattr(template, template_cls)
            else:
                template_cls = Plugin.load_plugin(template_cls, template.Template)
        self.sample_group[adapter_name].template = template_cls(self.model_id, **kwargs)

    def set_processor(self, processor_cls: Union[Type[processor.InputProcessor], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.sample_group, f'Add {adapter_name} first before training.'
        if isinstance(processor_cls, str):
            if hasattr(__file__.__module__, processor_cls):
                processor_cls = getattr(__file__.__module__, processor_cls)
            else:
                processor_cls = Plugin.load_plugin(processor_cls, processor.InputProcessor)
        self.sample_group[adapter_name].processor = processor_cls(self.model_id, **kwargs)

    @remote_function()
    def sample(self, trajectories: List[Trajectory], adapter_name = '') -> List[Trajectory]:
        if adapter_name:
            assert adapter_name in self.sample_group, f'Add {adapter_name} first before sampling.'
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
        for trajectory in trajectories:
            input_ids = self.template.encode(trajectory)
            request_id = str(uuid.uuid4().hex)
            request_ids.append(request_id)
            llm_inputs = {'prompt_token_ids': input_ids}
            self.engine.add_request(request_id, llm_inputs, generation_config=trajectory.generation_config, adapter_request=adapter_request)
        outputs = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs[output.request_id] = output
        outputs = [outputs[request_id] for request_id in request_ids]
        for trajectory, output in zip(trajectories, outputs):
            trajectory.messages.append(Message(role='assistant', content=output))
        return trajectories

    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig):
        assert adapter_name not in self.sample_group, f'{adapter_name} already exists.'
        assert adapter_name, 'Use a different adapter_name, current is empty.'
        self.sample_group[adapter_name] = SampleGroup()
        self.sample_group[adapter_name].adapter_name = adapter_name
        self.sample_group[adapter_name].adapter_config = config

    def sync_weights(self, state_dict: Dict[str, Any], adapter_name='') -> None:
        if not adapter_name:
            llm_model = self.engine.inner_model
            llm_model.load_weights(state_dict.items())
        else:
            assert adapter_name in self.sample_group, f'Add {adapter_name} first before sampling.'
            group = self.sample_group[adapter_name]
            lora_request = TensorLoRARequest(
                lora_name=adapter_name,
                lora_int_id=group.lora_int_id,
                lora_path='dummy_lora_path',
                peft_config=asdict(group.adapter_config),
                lora_tensors=state_dict,
            )
            self.engine.engine.add_lora(lora_request)
