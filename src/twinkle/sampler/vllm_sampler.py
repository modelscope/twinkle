# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM-based sampler using VLLMEngine (AsyncLLM).

Device Configuration:
    VLLMSampler automatically detects the number of available GPUs from
    CUDA_VISIBLE_DEVICES environment variable (set by twinkle's ResourceManager)
    and configures vLLM's tensor_parallel_size accordingly.
    
    To use tensor parallelism, configure DeviceGroup with gpus_per_worker > 1:
    
        # DP2 with TP2 (4 GPUs total, 2 workers, each with 2 GPUs)
        DeviceGroup(name='sampler', ranks=[0,1,2,3], gpus_per_worker=2)
        
        # TP4 (4 GPUs, 1 worker with all 4 GPUs)
        DeviceGroup(name='sampler', ranks=[0,1,2,3], gpus_per_worker=4)

Data Flow:
    When multiple VLLMSampler workers exist (DP > 1):
    - Data is dispatched via dispatch='slice_dp' (each worker gets a slice)
    - Results are collected via collect='flatten' (merged into single list)
"""
import asyncio
import atexit
import logging
import os
import threading
from dataclasses import asdict
from typing import List, Dict, Any, Union, Optional

from .base import Sampler
from .types import SamplingParams, SampleResponse, SampledSequence
from twinkle import remote_function, remote_class, DeviceMesh, requires
from twinkle.utils.platform import Platform
from twinkle.data_format import InputFeature, Trajectory
from twinkle.patch.vllm_lora_weights import VLLMLoraWeights, TensorLoRARequest

logger = logging.getLogger(__name__)


def _collect_sample_responses(results: List[SampleResponse]) -> SampleResponse:
    """Custom collect function to merge multiple SampleResponse objects.
    
    Args:
        results: List of SampleResponse from each DP worker.
        
    Returns:
        Merged SampleResponse with all sequences combined.
    """
    if not results:
        return SampleResponse(sequences=[])
    
    if len(results) == 1:
        return results[0]
    
    all_sequences = []
    for resp in results:
        if resp is not None and hasattr(resp, 'sequences'):
            all_sequences.extend(resp.sequences)
    
    return SampleResponse(sequences=all_sequences)


@remote_class()
class VLLMSampler(Sampler):
    """A vLLM-based sampler using VLLMEngine (AsyncLLM).
    
    This sampler automatically configures vLLM based on available GPUs.
    When gpus_per_worker > 1 is set in DeviceGroup, tensor parallelism is used.
    """

    def __init__(
        self,
        model_id: str,
        engine_args: Dict[str, Any] = None,
        device_mesh: DeviceMesh = None,
        **kwargs
    ):
        """Initialize VLLMSampler.
        
        Args:
            model_id: HuggingFace model ID or local path.
            engine_args: Arguments passed to VLLMEngine. If tensor_parallel_size
                is not specified, it will be automatically set based on the
                number of visible GPUs (from CUDA_VISIBLE_DEVICES).
            device_mesh: Parallel configuration for data parallelism.
            **kwargs: Additional arguments.
        """
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'
        super().__init__()
        requires('vllm')
        
        self.model_id = model_id
        self.device_mesh = device_mesh
        
        # Create a dedicated background event loop for vLLM async operations.
        # This is necessary because:
        # 1. vLLM's AsyncLLM requires its async methods to run in the same event loop
        #    where the engine was created (due to background output_handler task)
        # 2. Ray workers use uvloop which is already running, so we can't use
        #    run_until_complete() or asyncio.run() directly
        # 3. By creating engine in the background thread's event loop, all async
        #    operations stay in the same loop context
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="VLLMSampler-EventLoop"
        )
        self._async_thread.start()
        
        from .vllm_engine import VLLMEngine
        engine_kwargs = engine_args.copy() if engine_args else {}
        
        # Auto-detect tensor_parallel_size from CUDA_VISIBLE_DEVICES
        if 'tensor_parallel_size' not in engine_kwargs:
            tp_size = 1
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if visible_devices:
                num_gpus = len([d for d in visible_devices.split(',') if d.strip()])
                if num_gpus > 0:
                    tp_size = num_gpus
            engine_kwargs['tensor_parallel_size'] = tp_size
        
        # Set unique seed per engine based on rank for diverse sampling across DP workers
        # User can override by passing 'seed' in engine_args
        engine_seed = engine_kwargs.get('seed', None)
        if engine_seed is None:
            rank = Platform.get_rank()
            engine_seed = 42 + rank
            # set different seed to get different results
            engine_kwargs['seed'] = engine_seed
        
        # Create engine in the background event loop so all async operations
        # (including vLLM's internal background tasks) run in the same loop
        self.engine = self._run_in_loop(
            self._create_engine_async(VLLMEngine, model_id, engine_kwargs)
        )
        
        VLLMLoraWeights().patch(self)

        self._shutdown_called = False
        atexit.register(self.shutdown)

    def _run_event_loop(self):
        """Run the event loop in background thread."""
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()
    
    def _run_in_loop(self, coro):
        """Run a coroutine in the background event loop and wait for result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        return future.result()
    
    async def _create_engine_async(self, engine_cls, model_id, engine_kwargs):
        """Create engine in async context to ensure output_handler starts correctly."""
        return engine_cls(model_id=model_id, **engine_kwargs)

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
            messages,
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
        request_seed: Optional[int] = None,
        *,
        num_samples: int = 1,
    ) -> List[SampledSequence]:
        """Sample a single input asynchronously.
        
        Args:
            feat: Encoded input features containing 'input_ids' and optionally 'images'/'videos'.
            sampling_params: Sampling parameters.
            adapter_uri: Optional LoRA adapter URI.
            request_seed: Optional seed for reproducibility.
            num_samples: Number of completions to generate for this prompt.
            
        Returns:
            List of num_samples SampledSequence objects.
        """
        input_ids = feat['input_ids']
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        
        images = feat.get('images')
        videos = feat.get('videos')
        
        response = await self.engine.sample(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params,
            num_samples=num_samples,
            adapter_uri=adapter_uri,
            images=images,
            videos=videos,
        )
        
        # response.sequences contains num_samples sequences for this prompt
        return [
            SampledSequence(
                stop_reason=seq.stop_reason,
                tokens=seq.tokens,
                logprobs=seq.logprobs,
            )
            for seq in response.sequences
        ]

    @remote_function(dispatch='slice_dp', collect=_collect_sample_responses, lazy_collect=False)
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        *,
        num_samples: int = 1,
    ) -> SampleResponse:
        """Sample responses for given inputs.
        
        Args:
            inputs: Either InputFeature(s) or Trajectory(s).
                - InputFeature: Must contain 'input_ids'. For multimodal, include 'images'/'videos'.
                - Trajectory: Must contain 'messages'. Requires template to be set.
            sampling_params: Sampling parameters.
            adapter_name: Optional LoRA adapter name.
            num_samples: Number of completions to generate per input prompt.
                        When > 1, returns num_samples sequences for each input.
            
        Returns:
            SampleResponse containing sampled sequences.
            Total sequences = len(inputs) * num_samples.
            
        Note:
            In Ray mode with multiple workers (DP > 1):
            - Data is automatically sliced by DP rank (dispatch='slice_dp')
            - Results are merged using _collect_sample_responses
            - Each worker receives already-sliced inputs (e.g., DP4 with 8 inputs -> 2 per worker)
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
        
        # Sample all inputs in parallel using background event loop
        async def _sample_all():
            tasks = [
                self._sample_single(feat, sampling_params, adapter_uri, num_samples=num_samples)
                for feat in encoded_inputs
            ]
            return await asyncio.gather(*tasks)
        
        results = self._run_in_loop(_sample_all())
        
        # Flatten results (each result contains num_samples sequences)
        all_sequences = []
        for seqs in results:
            all_sequences.extend(seqs)
        
        return SampleResponse(sequences=all_sequences)

    @remote_function(dispatch='all', collect='first')
    def sync_weights(self, state_dict: Dict[str, Any], adapter_name: str = '') -> None:
        """Sync weights from training to vLLM engine.
        
        Args:
            state_dict: Model state dict to sync.
            adapter_name: If provided, sync as LoRA adapter. Otherwise sync base model.
        """
        if not adapter_name:
            self._run_in_loop(self.engine.update_weights(state_dict))
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
    
    @remote_function(dispatch='all', collect='first')
    def sleep(self, level: int = 2) -> None:
        """Release GPU memory for colocate mode.
        
        Call this before training to free up GPU memory used by vLLM.
        
        Args:
            level: Sleep level (1=light, 2=deep). Default 2 releases most memory.
        """
        self._run_in_loop(self.engine.sleep(level))
    
    @remote_function(dispatch='all', collect='first')
    def wake_up(self, tags: List[str] = None, reload_weights: bool = False) -> None:
        """Resume GPU memory for colocate mode.
        
        Call this before sampling to reload weights/KV cache into GPU.
        
        Args:
            tags: Optional list of memory types to resume (e.g., ['weights', 'kv_cache']).
                  If None, resumes all.
            reload_weights: If True, reload weights from disk after wake_up.
                  Required after level 2 sleep which discards weights.
        """
        self._run_in_loop(self.engine.wake_up(tags=tags, reload_weights=reload_weights))

    def shutdown(self):
        """Gracefully shutdown the vLLM engine and background event loop.

        Registered via atexit so it runs automatically on process exit,
        before GC destroys objects in unpredictable order. Safe to call
        multiple times (idempotent).
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True

        # 1. Shutdown vLLM engine (stops EngineCore process and output_handler)
        try:
            if hasattr(self, 'engine') and self.engine is not None:
                self.engine.shutdown()
        except Exception as e:
            logger.warning(f"VLLMSampler engine shutdown error: {e}")

        # 2. Stop the background event loop and join thread
        try:
            if hasattr(self, '_async_loop') and self._async_loop.is_running():
                self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            if hasattr(self, '_async_thread') and self._async_thread.is_alive():
                self._async_thread.join(timeout=5)
        except Exception as e:
            logger.warning(f"VLLMSampler event loop shutdown error: {e}")
