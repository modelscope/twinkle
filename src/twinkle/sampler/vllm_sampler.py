import uuid
from typing import List, Type

from .base import Sampler
from ..trajectory import Trajectory, Message
from ..utils import requires
from ..template import Template

class VLLMSampler(Sampler):

    def __init__(self, engine_args: 'vllm.EngineArgs', template: Type[Template], remote_group):
        super().__init__()
        requires('vllm')
        from vllm import LLMEngine
        vllm_config = engine_args.create_engine_config()
        self.engine = LLMEngine.from_vllm_config(
            vllm_config=vllm_config,
        )
        self.template = template_type()


    def sample(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        request_ids = []
        for trajectory in trajectories:
            input_ids = self.template(trajectory)
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