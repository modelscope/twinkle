# Copyright (c) ModelScope Contributors. All rights reserved.
"""PyTorch native sampler using transformers model.generate()"""
from typing import List, Dict, Any, Type, Union

import torch
from peft import PeftConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import twinkle
from twinkle import remote_class, remote_function, DeviceMesh, Plugin
from twinkle.data_format import Trajectory, Message
from twinkle.hub import HubOperation
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils import construct_class

from .base import Sampler


@remote_class()
class TorchSampler(Sampler):
    """A PyTorch native sampler using transformers model.generate().

    Args:
        model_id: The model id for inference.
        device_mesh: Device mesh
        torch_dtype: The torch dtype to use, default bf16.
        trust_remote_code: Whether to trust remote code.
    """

    _default_adapter_name = ''

    def __init__(
        self,
        model_id: str,
        device_mesh: DeviceMesh = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        **kwargs
    ):
        super().__init__()
        model_id = HubOperation.download_model(model_id)
        self.model_id = model_id
        self.device_mesh = device_mesh
        
        # Determine device (for reference, actual device placement handled by device_map='auto')
        if device_mesh is not None and getattr(device_mesh, 'device_type', None):
            self.device = torch.device(device_mesh.device_type)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            self.device = torch.device('npu')
        else:
            self.device = torch.device('cpu')
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map='auto',
            **kwargs
        )
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store adapter and template info
        self.sample_group: Dict[str, Dict[str, Any]] = {
            self._default_adapter_name: {
                'adapter_name': None,
                'adapter_config': None,
                'template': None,
                'processor': None,
            }
        }

    def _check_adapter_valid(self, adapter_name: str):
        assert adapter_name in self.sample_group, f'Use a valid adapter_name first, current is: {adapter_name}'

    def set_template(self, template_cls: Union[Type[Template], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        template_ins = construct_class(template_cls, Template, twinkle.template, **kwargs)
        self.sample_group[adapter_name]['template'] = template_ins
        if adapter_name == '' or not hasattr(self, 'template'):
            self.template = template_ins

    def set_processor(self, processor_cls: Union[Type[InputProcessor], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        if isinstance(processor_cls, str):
            if hasattr(twinkle.processor, processor_cls):
                processor_cls = getattr(twinkle.processor, processor_cls)
            else:
                processor_cls = Plugin.load_plugin(processor_cls, InputProcessor)
        self.sample_group[adapter_name]['processor'] = processor_cls(**kwargs)

    @remote_function()
    def sample(self, trajectories: List[Trajectory], adapter_name: str = '') -> List[Trajectory]:
        """Sample responses for given trajectories using model.generate()."""
        self._check_adapter_valid(adapter_name)
        template_ins = self.sample_group[adapter_name]['template']
        
        results = []
        for trajectory in trajectories:
            # Encode the trajectory to get input_ids
            inputs = template_ins.encode(trajectory)
            input_ids = inputs['input_ids']
            
            if hasattr(input_ids, 'tolist'):
                input_ids = input_ids.tolist()
            
            # Fix: create tensor directly on model device to avoid "npu:0 vs cpu" device mismatch error
            device = next(self.model.parameters()).device
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_tensor)
            
            # Get generation config from trajectory if provided
            generation_config = trajectory.get('generation_config')
            if isinstance(generation_config, list):
                generation_config = dict(generation_config)
            
            # Default generation parameters
            gen_kwargs = {
                'max_new_tokens': 2048,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }

            # Override with trajectory-specific config
            if generation_config:
                # Convert VLLM-style parameters to transformers-style
                converted_config = {}
                for key, value in generation_config.items():
                    if key == 'max_tokens':
                        converted_config['max_new_tokens'] = value
                    elif key == 'min_tokens':
                        converted_config['min_new_tokens'] = value
                    elif key in ['stop', 'ignore_eos', 'logit_bias']:
                        # Skip VLLM-specific parameters not supported by transformers
                        continue
                    else:
                        converted_config[key] = value
                gen_kwargs.update(converted_config)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
            
            # Extract only the generated part (excluding input)
            generated_ids = outputs[0][len(input_ids):].tolist()
            
            # Decode the response
            response = template_ins.decode(generated_ids, skip_special_tokens=True)
            
            # Append assistant message to trajectory
            if isinstance(trajectory, dict):
                trajectory.setdefault('messages', []).append(
                    Message(role='assistant', content=response)
                )
            else:
                trajectory.messages.append(Message(role='assistant', content=response))
            results.append(trajectory)
        
        return results

    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig):
        """Add a LoRA adapter to the sampler's model."""
        if adapter_name in self.sample_group:
            return  # Already exists
        
        self.sample_group[adapter_name] = {
            'adapter_name': adapter_name,
            'adapter_config': config,
            'template': None,
            'processor': None,
        }
        
        # Apply LoRA to model if config provided
        if config is not None:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                # Add another adapter
                self.model.add_adapter(adapter_name, config)
            else:
                # First adapter, wrap the model
                self.model = get_peft_model(self.model, config, adapter_name=adapter_name)
            self.model.eval()

    def sync_weights(self, state_dict: Dict[str, Any], adapter_name: str = '') -> None:
        """Sync weights from training model to sampler model."""
        if not adapter_name:
            # Full model weights
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # LoRA adapter weights
            self._check_adapter_valid(adapter_name)
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                # Load only adapter-specific weights to avoid modifying base model weights
                adapter_state_dict: Dict[str, Any] = {}
                for key, value in state_dict.items():
                    # Keep only parameters that belong to the specified adapter
                    if adapter_name in key:
                        adapter_state_dict[key] = value
                if adapter_state_dict:
                    self.model.load_state_dict(adapter_state_dict, strict=False)

    def remove_adapter(self, adapter_name: str):
        """Remove an adapter from the sampler."""
        if adapter_name and adapter_name in self.sample_group:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                self.model.base_model.delete_adapter(adapter_name=adapter_name)
            self.sample_group.pop(adapter_name)
