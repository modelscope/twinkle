import uuid
from typing import List, Type, Dict, Any, Union

import twinkle
from .base import Sampler
from .. import remote_function, remote_class, InputProcessor
from ..plugin.plugin import Plugin
from ..trajectory import Trajectory, Message
from ..utils import requires
from ..template import Template

@remote_class()
class VLLMSampler(Sampler):

    def __init__(self, model_id: str, engine_args: Dict[str, Any], template: Type[Template], remote_group):
        super().__init__()
        requires('vllm')
        from vllm import LLMEngine, EngineArgs
        engine_args = EngineArgs(**engine_args)
        vllm_config = engine_args.create_engine_config()
        self.engine = LLMEngine.from_vllm_config(
            vllm_config=vllm_config,
        )
        self.template = template(model_id)

    def set_template(self, template: Union[Type[Template], str]):
        if isinstance(template, str):
            if hasattr(twinkle.template, template):
                template = getattr(twinkle.template, template)
            else:
                template = Plugin.load_plugin(template, Template)
        self.template = template(self.model_id)

    @remote_function()
    def set_input_processor(self, processor: Union[Type[InputProcessor], str]):
        if isinstance(processor, str):
            if hasattr(twinkle.processor, processor):
                processor = getattr(twinkle.processor, processor)
            else:
                processor = Plugin.load_plugin(processor, InputProcessor)
        self.processor = processor()

    @remote_function()
    def sample(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        request_ids = []
        for trajectory in trajectories:
            input_ids = self.template.encode(trajectory)
            request_id = str(uuid.uuid4().hex)
            request_ids.append(request_id)
            llm_inputs = {'prompt_token_ids': input_ids}
            self.engine.add_request(request_id, llm_inputs, generation_config=trajectory.generation_config)
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