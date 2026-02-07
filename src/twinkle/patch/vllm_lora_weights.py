# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import field
from typing import Optional, Dict

from .base import Patch
from twinkle import requires
import torch

try:
    from vllm.lora.request import LoRARequest
except (ModuleNotFoundError, ImportError):
    LoRARequest = object


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)
    lora_embeddings: Optional[Dict[str, torch.Tensor]] = None

    @property
    def config(self):
        return self.peft_config

    @property
    def embeddings(self):
        return self.lora_embeddings



class VLLMLoraWeights(Patch):

    def __call__(self, sampler, **kwargs):
        requires('vllm>=0.11')
        
        _sampler_ref = sampler
        
        def _get_tokenizer():
            """Get tokenizer lazily from sampler's template."""
            if _sampler_ref.template is not None:
                return _sampler_ref.template.tokenizer
            return None
        
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        try:
            from vllm.lora.models import LoRAModel
        except ImportError:
            # vllm >= 0.13 https://github.com/vllm-project/vllm/pull/30253
            from vllm.lora.lora_model import LoRAModel
        from vllm.lora.utils import get_adapter_absolute_path

        try:
            from vllm.transformers_utils.tokenizer_group import TokenizerGroup
        except ImportError:
            # removed in https://github.com/vllm-project/vllm/pull/24078
            TokenizerGroup = None

        def patched_load_adapter(self: LRUCacheWorkerLoRAManager, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            code borrowed from verl.utils.vllm.utils.py
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors
            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
            load memory-based LoRA tensors.
            """
            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_modules: list[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_modules.extend(packed_modules_mapping[module])
                else:
                    expected_lora_modules.append(module)
            expected_lora_modules = list(set(expected_lora_modules))
            # this is the patch
            lora_tensors = None
            from vllm.lora.peft_helper import PEFTHelper
            if isinstance(lora_request, TensorLoRARequest):
                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)
            else:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)
                peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)
            # Validates the LoRA configuration against requirements before
            # loading weights, throwing an exception if validation fails.
            peft_helper.validate_legal(self.lora_config)
            # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
            # to ensure correct loading of lora weights.
            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, 'hf_to_vllm_mapper', None)
            if isinstance(lora_request, TensorLoRARequest):  # this is the patch
                lora = self._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    device='cpu',
                    dtype=self.lora_config.lora_dtype,
                    embeddings=None,
                    target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )
            else:
                lora = self._lora_model_cls.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device='cpu',
                    dtype=self.lora_config.lora_dtype,
                    target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )
            if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                raise ValueError(f'LoRA added vocab size {lora.extra_vocab_size} is greater than '
                                 f'lora_extra_vocab_size {self.lora_config.lora_extra_vocab_size}.')
            return lora

        def patched_get_lora_tokenizer(self: TokenizerGroup, lora_request: LoRARequest):
            # since we pass dummy path, skip get tokenizer from path
            # Use lazy tokenizer access
            tokenizer = _get_tokenizer()
            if tokenizer is None:
                # Fallback to the original method if tokenizer not available
                return self._old_get_lora_tokenizer(lora_request)
            return tokenizer

        if not hasattr(LRUCacheWorkerLoRAManager, '_old_load_adapter'):
            _old_load_adapter = LRUCacheWorkerLoRAManager._load_adapter
            LRUCacheWorkerLoRAManager._load_adapter = patched_load_adapter
            LRUCacheWorkerLoRAManager._old_load_adapter = _old_load_adapter
            if TokenizerGroup is not None:
                TokenizerGroup._old_get_lora_tokenizer = TokenizerGroup.get_lora_tokenizer
                TokenizerGroup.get_lora_tokenizer = patched_get_lora_tokenizer
