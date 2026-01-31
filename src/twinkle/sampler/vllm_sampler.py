# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM-based sampler using VLLMEngine (AsyncLLM)."""
import asyncio
import os
from dataclasses import asdict
from typing import List, Dict, Any, Union, Optional

from .base import Sampler
from .types import SamplingParams, SampleResponse, SampledSequence
from twinkle import remote_function, remote_class, DeviceMesh, requires
from twinkle.data_format import InputFeature, Trajectory
from twinkle.patch.vllm_lora_weights import VLLMLoraWeights, TensorLoRARequest


@remote_class()
class VLLMSampler(Sampler):
    """A vLLM-based sampler using VLLMEngine (AsyncLLM)."""

    def __init__(
        self,
        model_id: str,
        engine_args: Dict[str, Any] = None,
        device_mesh: DeviceMesh = None,
        **kwargs
    ):
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'
        super().__init__()
        requires('vllm')
        
        self.model_id = model_id
        self.device_mesh = device_mesh
        
        from .vllm_engine import VLLMEngine
        engine_kwargs = engine_args or {}
        self.engine = VLLMEngine(model_id=model_id, **engine_kwargs)
        
        VLLMLoraWeights().patch(self)

    def encode_trajectory_for_vllm(self, trajectory: Trajectory, adapter_name: str = '') -> InputFeature:
        """Encode trajectory for vLLM - does not expand image tokens.

        Args:
            trajectory: The trajectory to encode.
            adapter_name: Optional LoRA adapter name.
            
        Returns:
            InputFeature with input_ids suitable for vLLM (unexpanded image tokens).
        """
        template = self._get_template(adapter_name)
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        
        # For vLLM: tokenize without passing images to the processor
        # This gives us the text with placeholder tokens, which vLLM will expand
        messages = [dict(msg) for msg in trajectory['messages']]
        
        # Preprocess images for vLLM (load as PIL Images)
        # vLLM expects PIL Images, not URLs
        images = []
        if trajectory.get('images'):
            images = template.preprocess_images(trajectory['images'])
        videos = []
        if trajectory.get('videos'):
            videos = template.preprocess_videos(trajectory['videos'])
        
        # Apply chat template without images (to get unexpanded tokens)
        # We need to convert <image> placeholders to the model's native format
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, str) and template.is_mm:
                # Convert placeholders to standard format for tokenization
                if template.image_placeholder in content:
                    # Split content by image placeholder and rebuild with proper format
                    parts = content.split(template.image_placeholder)
                    new_content = []
                    for i, part in enumerate(parts):
                        if i > 0:
                            # Add image token structure (vLLM will expand this)
                            new_content.append({'type': 'image'})
                        if part.strip():
                            new_content.append({'type': 'text', 'text': part})
                    msg['content'] = new_content if new_content else [{'type': 'text', 'text': ''}]
        
        encoded = template.processor.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors='pt',
        )
        
        input_ids = encoded['input_ids']
        if hasattr(input_ids, 'squeeze'):
            input_ids = input_ids.squeeze(0)
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        
        result = InputFeature(input_ids=input_ids)
        
        # Attach preprocessed images/videos for vLLM
        if images:
            result['images'] = images
        if videos:
            result['videos'] = videos
        
        return result

    async def _sample_single(
        self,
        feat: Dict[str, Any],
        sampling_params: SamplingParams,
        adapter_uri: Optional[str] = None,
    ) -> List[SampledSequence]:
        """Sample a single input asynchronously."""
        input_ids = feat['input_ids']
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        
        images = feat.get('images')
        videos = feat.get('videos')
        
        response = await self.engine.sample(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params,
            adapter_uri=adapter_uri,
            images=images,
            videos=videos,
        )
        
        return [
            SampledSequence(
                stop_reason=seq.stop_reason,
                tokens=seq.tokens,
                logprobs=seq.logprobs,
            )
            for seq in response.sequences
        ]

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
                - InputFeature: Must contain 'input_ids'. For multimodal, include 'images'/'videos'.
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
            encoded_inputs = [
                self.encode_trajectory_for_vllm(traj, adapter_name) 
                for traj in inputs_list
            ]
        else:
            encoded_inputs = inputs_list
        
        # Get adapter URI if using LoRA
        adapter_uri = None
        if adapter_name:
            group = self.sample_group[adapter_name]
            if group.lora_ready and self.engine.adapter_manager:
                adapter_uri = self.engine.adapter_manager.get_uri(adapter_name)
        
        # Sample all inputs in parallel using asyncio.gather
        async def _sample_all():
            tasks = [
                self._sample_single(feat, sampling_params, adapter_uri)
                for feat in encoded_inputs
            ]
            return await asyncio.gather(*tasks)
        
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(_sample_all())
        finally:
            loop.close()
        
        # Flatten results
        all_sequences = []
        for seqs in results:
            all_sequences.extend(seqs)
        
        return SampleResponse(sequences=all_sequences)

    def sync_weights(self, state_dict: Dict[str, Any], adapter_name: str = '') -> None:
        if not adapter_name:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.engine.update_weights(state_dict))
            finally:
                loop.close()
        else:
            self._check_adapter_valid(adapter_name)
            group = self.sample_group[adapter_name]
            
            lora_request = TensorLoRARequest(
                lora_name=adapter_name,
                lora_int_id=group.lora_int_id,
                lora_path='dummy_lora_path',
                peft_config=asdict(group.adapter_config) if group.adapter_config else {},
                lora_tensors=state_dict,
            )
            
            if group.lora_ready:
                remove_lora = getattr(self.engine, 'remove_lora', None)
                if remove_lora is not None:
                    try:
                        remove_lora(adapter_name)
                    except TypeError:
                        remove_lora(group.lora_int_id)
                group.lora_ready = False
            
            if self.engine.add_lora(lora_request):
                group.lora_ready = True

    def remove_adapter(self, adapter_name: str):
        if adapter_name and adapter_name in self.sample_group:
            group = self.sample_group[adapter_name]
            if group.lora_ready:
                remove_lora = getattr(self.engine, 'remove_lora', None)
                if remove_lora is not None:
                    try:
                        remove_lora(adapter_name)
                    except TypeError:
                        remove_lora(group.lora_int_id)
            self.sample_group.pop(adapter_name)
