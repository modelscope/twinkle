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
        import inspect

        allowed = set(inspect.signature(EngineArgs.__init__).parameters.keys())

        allowed.discard('self')

        engine_args = {k: v for k, v in engine_args.items() if k in allowed}

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
    def _kv_list_to_dict(self, kv_list):
        """Trajectory.generation_config is List[Tuple[str, Any]]; normalize to dict."""
        if kv_list is None:
          return {}
        if isinstance(kv_list, dict):
          return kv_list
        if isinstance(kv_list, list):
          out = {}
        for item in kv_list:
            if isinstance(item, tuple) and len(item) == 2:
                k, v = item
                out[str(k)] = v
            return out
        return {}

    def _build_sampling_params(self, traj: Trajectory):
        """WIP: build vLLM SamplingParams from trajectory.generation_config (best-effort)."""
        from vllm import SamplingParams

        cfg = self._kv_list_to_dict(traj.get("generation_config", None))
        temperature = float(cfg.get("temperature", os.environ.get("TWINKLE_TEMPERATURE", 0.8)))
        top_p = float(cfg.get("top_p", os.environ.get("TWINKLE_TOP_P", 0.95)))
        max_tokens = int(cfg.get("max_new_tokens", cfg.get("max_tokens", os.environ.get("TWINKLE_MAX_NEW_TOKENS", 128))))
        n = int(cfg.get("n", cfg.get("num_return_sequences", os.environ.get("TWINKLE_NUM_GENERATIONS", 1))))

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
        )

    @remote_function()
    def sample(self, trajectories: List[Trajectory], adapter_name: str = "") -> List[Trajectory]:
        self._check_adapter_valid(adapter_name)

        group = self.sample_group[adapter_name]
        template_ins = group.template
        if template_ins is None:
            raise RuntimeError(
                "VLLMSampler requires template to be set before sampling. "
                "Call sampler.set_template(...) first."
            )

        # Ensure messages exists (Trajectory is total=False)
        for t in trajectories:
            t.setdefault("messages", [])

        # LoRA request (WIP: keep current behavior)
        if adapter_name:
            from vllm.lora.request import LoRARequest
            adapter_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=group.lora_int_id,
                lora_path="dummy_lora_path",
            )
        else:
            adapter_request = None

        request_ids: List[str] = []
        outputs_by_id: Dict[str, Any] = {}

        # enqueue
        for trajectory in trajectories:
            inputs = template_ins.encode(trajectory)
            request_id = uuid.uuid4().hex
            request_ids.append(request_id)

            sampling_params = self._build_sampling_params(trajectory)

            # vLLM LLMEngine.add_request signature varies by version; try both
            try:
                self.engine.add_request(
                    request_id=request_id,
                    prompt=None,
                    sampling_params=sampling_params,
                    prompt_token_ids=inputs.input_ids,
                    lora_request=adapter_request,
                )
            except TypeError:
                # fallback older naming
                self.engine.add_request(
                    request_id,
                    prompt=None,
                    sampling_params=sampling_params,
                    prompt_token_ids=inputs.input_ids,
                    adapter_request=adapter_request,
                )

        # collect until done
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for out in step_outputs:
                if getattr(out, "finished", False):
                    outputs_by_id[out.request_id] = out

        # decode and append assistant message (take first candidate)
        for trajectory, rid in zip(trajectories, request_ids):
            out = outputs_by_id.get(rid)
            if out is None:
                raise RuntimeError(f"Missing vLLM output for request_id={rid}")

            candidates = getattr(out, "outputs", None)
            if candidates:
                token_ids = candidates[0].token_ids
            else:
                token_ids = getattr(out, "token_ids", None)
                if token_ids is None:
                    raise RuntimeError(f"vLLM output has no token_ids for request_id={rid}")

            response = template_ins.decode(token_ids)
            trajectory["messages"].append(Message(role="assistant", content=response))

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
