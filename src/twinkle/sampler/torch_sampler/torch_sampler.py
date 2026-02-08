# Copyright (c) ModelScope Contributors. All rights reserved.
"""PyTorch native sampler using TransformersEngine."""
from typing import List, Dict, Any, Optional, Type, Union

import torch
from peft import PeftConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from twinkle.sampler.base import Sampler
from twinkle.data_format.sampling import SamplingParams, SampleResponse, SampledSequence
from twinkle import remote_class, remote_function, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation


@remote_class()
class TorchSampler(Sampler):
    """A PyTorch native sampler using TransformersEngine."""

    def __init__(
        self,
        model_id: str,
        device_mesh: DeviceMesh = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        model_cls: Optional[Union[Type[PreTrainedModel], str, Type[_BaseAutoModelClass]]] = AutoModelForCausalLM,
        **kwargs
    ):
        super().__init__()
        model_id = HubOperation.download_model(model_id)
        self.model_id = model_id
        self.device_mesh = device_mesh
        
        if device_mesh is not None and getattr(device_mesh, 'device_type', None):
            self.device = torch.device(device_mesh.device_type)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            self.device = torch.device('npu')
        else:
            self.device = torch.device('cpu')
        
        from .transformers_engine import TransformersEngine
        self.engine = TransformersEngine(
            model_id=model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            model_cls=model_cls,
            **kwargs
        )
        self.model = self.engine.model
        self.tokenizer = self.engine.tokenizer

    @remote_function()
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
    ) -> SampleResponse:
        """Sample responses for given inputs.
        
        Args:
            inputs: Either InputFeature(s) or Trajectory(s).
                - InputFeature: Must contain 'input_ids'.
                - Trajectory: Must contain 'messages'. Requires template to be set.
            sampling_params: Sampling parameters.
            adapter_name: Optional LoRA adapter name.
            
        Returns:
            SampleResponse containing sampled sequences.
        """
        self._check_adapter_valid(adapter_name)
        
        if sampling_params is None:
            sampling_params = SamplingParams()
        elif isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)
        
        inputs_list = self._normalize_inputs(inputs)
        
        # Check if inputs are Trajectory (not encoded) - aligned with Model.forward logic
        is_trajectory = self._is_trajectory(inputs)
        
        if is_trajectory:
            template = self._get_template(adapter_name)
            assert template is not None, \
                'Use set_template to add a template when trying to input Trajectory'
            encoded_inputs = [self.encode_trajectory(traj, adapter_name) for traj in inputs_list]
        else:
            encoded_inputs = inputs_list
        
        gen_kwargs = sampling_params.to_transformers(self.tokenizer)
        gen_kwargs["return_dict_in_generate"] = True
        gen_kwargs["output_scores"] = True
        
        all_sequences = []
        device = next(self.model.parameters()).device
        
        for feat in encoded_inputs:
            input_ids = feat['input_ids']
            if hasattr(input_ids, 'tolist'):
                input_ids = input_ids.tolist()
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_tensor)
            
            # Build model inputs including multimodal data
            model_inputs = {
                'input_ids': input_tensor,
                'attention_mask': attention_mask,
            }
            
            # Add extra inputs for multimodal models (pixel_values, image_grid_thw, etc.)
            # These are typically produced by template.encode() for VLM models
            extra_keys = ['pixel_values', 'image_grid_thw', 'video_grid_thw', 
                         'pixel_values_videos', 'second_per_grid_ts']
            for key in extra_keys:
                if key in feat:
                    value = feat[key]
                    if hasattr(value, 'to'):
                        model_inputs[key] = value.to(device)
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        # Handle list of tensors
                        if hasattr(value[0], 'to'):
                            model_inputs[key] = [v.to(device) for v in value]
                        else:
                            model_inputs[key] = value
                    else:
                        model_inputs[key] = value
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    **gen_kwargs
                )
            
            generated_ids = outputs.sequences
            prompt_len = len(input_ids)
            
            gen_tokens = generated_ids[0][prompt_len:].tolist()
            
            seq_logprobs = None
            # TODO: fix logprobs
            if hasattr(outputs, 'scores') and outputs.scores:
                seq_logprobs = []
                for k, score in enumerate(outputs.scores):
                    if k >= len(gen_tokens):
                        break
                    log_probs = torch.log_softmax(score[0], dim=-1)
                    seq_logprobs.append(log_probs[gen_tokens[k]].item())
            
            stop_reason = "length"
            if gen_tokens and gen_tokens[-1] == self.tokenizer.eos_token_id:
                stop_reason = "stop"

            all_sequences.append(SampledSequence(
                stop_reason=stop_reason,
                tokens=gen_tokens,
                logprobs=seq_logprobs,
            ))
        
        return SampleResponse(sequences=all_sequences)

    @remote_function()
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig):
        super().add_adapter_to_sampler(adapter_name, config)
        
        if config is not None:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                self.model.add_adapter(adapter_name, config)
            else:
                self.model = get_peft_model(self.model, config, adapter_name=adapter_name)
            self.model.eval()

    @remote_function()
    def sync_weights(self, state_dict: Dict[str, Any], adapter_name: str = '') -> None:
        if not adapter_name:
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self._check_adapter_valid(adapter_name)
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                adapter_state_dict = {k: v for k, v in state_dict.items() if adapter_name in k}
                if adapter_state_dict:
                    self.model.load_state_dict(adapter_state_dict, strict=False)

    @remote_function()
    def remove_adapter(self, adapter_name: str):
        if adapter_name and adapter_name in self.sample_group:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                self.model.base_model.delete_adapter(adapter_name=adapter_name)
            super().remove_adapter(adapter_name)
