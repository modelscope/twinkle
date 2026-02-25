# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM-based sampler using VLLMEngine (AsyncLLM).

Device Configuration:
    vLLMSampler automatically detects the number of available GPUs from
    CUDA_VISIBLE_DEVICES environment variable (set by twinkle's ResourceManager)
    and configures vLLM's tensor_parallel_size accordingly.

    To use tensor parallelism, configure DeviceGroup with gpus_per_worker > 1:

        # DP2 with TP2 (4 GPUs total, 2 workers, each with 2 GPUs)
        DeviceGroup(name='sampler', ranks=[0,1,2,3], gpus_per_worker=2)

        # TP4 (4 GPUs, 1 worker with all 4 GPUs)
        DeviceGroup(name='sampler', ranks=[0,1,2,3], gpus_per_worker=4)

Data Flow:
    When multiple vLLMSampler workers exist (DP > 1):
    - Data is dispatched via dispatch='slice_dp' (each worker gets a slice)
    - Results are collected via collect='flatten' (merged into single list)
"""
import asyncio
import atexit
import os
import threading
from typing import Any, Dict, List, Optional, Union

from twinkle import DeviceMesh, get_logger, remote_class, remote_function, requires
from twinkle.checkpoint_engine import CheckpointEngineMixin
from twinkle.data_format import InputFeature, SampledSequence, SampleResponse, SamplingParams, Trajectory
from twinkle.patch.vllm_lora_weights import VLLMLoraWeights
from twinkle.sampler.base import Sampler
from twinkle.utils import Platform

logger = get_logger()


def _collect_sample_responses(results: List[SampleResponse], **kwargs) -> SampleResponse:
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
class vLLMSampler(Sampler, CheckpointEngineMixin):
    """A vLLM-based sampler using VLLMEngine (AsyncLLM).

    This sampler automatically configures vLLM based on available GPUs.
    When gpus_per_worker > 1 is set in DeviceGroup, tensor parallelism is used.
    """

    def __init__(self, model_id: str, engine_args: Dict[str, Any] = None, device_mesh: DeviceMesh = None, **kwargs):
        """Initialize vLLMSampler.

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
        self._async_thread = threading.Thread(target=self._run_event_loop, daemon=True, name='vLLMSampler-EventLoop')
        self._async_thread.start()

        from .vllm_engine import VLLMEngine
        engine_kwargs = engine_args.copy() if engine_args else {}

        # Auto-detect tensor_parallel_size from CUDA_VISIBLE_DEVICES
        if 'tensor_parallel_size' not in engine_kwargs:
            tp_size = 1
            visible_devices = os.environ.get(Platform.visible_device_env(), '')
            if visible_devices:
                num_gpus = len([d for d in visible_devices.split(',') if d.strip()])
                if num_gpus > 0:
                    tp_size = num_gpus
            logger.info(f'vLLM TP size: {tp_size}')
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
        self.engine: VLLMEngine = self._run_in_loop(self._create_engine_async(VLLMEngine, model_id, engine_kwargs))
        # fix: On NPU, monkey_patch_model can trigger Triton compatibility errors and abort sampler init.
        # fix: Explicitly skip this patch on NPU and keep it for non-NPU paths only.
        # NPU platform may trigger triton errors with monkey_patch_model
        if Platform.get_platform().device_prefix() != 'npu':
            self._run_in_loop(self.engine.engine.collective_rpc('monkey_patch_model'))

        VLLMLoraWeights()(self)

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
        template = self.template
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
        lora_request: Optional[Any] = None,
        *,
        logprobs: bool = True,
        num_samples: int = 1,
    ) -> List[SampledSequence]:
        """Sample a single input asynchronously.

        Args:
            feat: Encoded input features containing 'input_ids' and optionally 'images'/'videos'.
            sampling_params: Sampling parameters.
            adapter_path: Optional LoRA adapter path (legacy, prefer lora_request).
            lora_request: Pre-built LoRARequest to attach to the sampling request.
                Avoids repeated ``_get_or_load_lora`` calls per input.
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
            logprobs=logprobs,
            num_samples=num_samples,
            lora_request=lora_request,
            images=images,
            videos=videos,
        )

        # response.sequences contains num_samples sequences for this prompt
        return [
            SampledSequence(
                stop_reason=seq.stop_reason,
                tokens=seq.tokens,
                logprobs=seq.logprobs,
                decoded=self.template.decode(seq.tokens),
                new_input_feature=self.template.concat_input_feature(feat, seq.tokens),
            ) for seq in response.sequences
        ]

    @remote_function(dispatch='slice_dp', collect=_collect_sample_responses, lazy_collect=False)
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        logprobs: bool = True,
        num_samples: int = 1,
        return_encoded: bool = False,
    ) -> SampleResponse:
        """Sample responses for given inputs.

        Args:
            inputs: Either InputFeature(s) or Trajectory(s).
                - InputFeature: Must contain 'input_ids'. For multimodal, include 'images'/'videos'.
                - Trajectory: Must contain 'messages'. Requires template to be set.

            sampling_params: Sampling parameters.

            adapter_name: Optional LoRA adapter name.

            adapter_path: Optional LoRA adapter path.

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
        if sampling_params is None:
            sampling_params = SamplingParams()
        elif isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)

        inputs_list = self._normalize_inputs(inputs)

        # Check if inputs are Trajectory (not encoded) - aligned with Model.forward logic
        is_trajectory = self._is_trajectory(inputs)

        if is_trajectory:
            template = self.template
            assert template is not None, \
                'Use set_template to add a template when trying to input Trajectory'
            encoded_inputs = [self.encode_trajectory_for_vllm(traj, adapter_name) for traj in inputs_list]
        else:
            encoded_inputs = inputs_list

        lora_request = None
        if adapter_path is not None:
            lora_request = self._run_in_loop(self.engine._get_or_load_lora(adapter_path))
            if lora_request is None:
                logger.warning(f'Failed to pre-load LoRA from {adapter_path}, '
                               'sampling will proceed without LoRA')

        # Sample all inputs in parallel using background event loop
        async def _sample_all():
            tasks = [
                self._sample_single(
                    feat,
                    sampling_params,
                    lora_request=lora_request,
                    logprobs=logprobs,
                    num_samples=num_samples,
                ) for feat in encoded_inputs
            ]
            return await asyncio.gather(*tasks)

        results = self._run_in_loop(_sample_all())
        # Flatten results (each result contains num_samples sequences)
        all_sequences = []
        for seqs in results:
            all_sequences.extend(seqs)
        return SampleResponse(sequences=all_sequences)

    @remote_function(dispatch='all', collect='first')
    def sleep(self, level: int = 1) -> None:
        """
        Release GPU memory for colocate mode.
        """
        self._run_in_loop(self.engine.sleep(level))

    @remote_function(dispatch='all', collect='first')
    def wake_up(self, tags: List[str] = None) -> None:
        self._run_in_loop(self.engine.wake_up(tags=tags))

    @remote_function(dispatch='all', collect='first')
    def reset_prefix_cache(self):
        self._run_in_loop(self.engine.reset_prefix_cache())

    @remote_function(dispatch='all', lazy_collect=True)
    def receive_weights(
        self,
        base_sync_done: bool = False,
        peft_config: dict = None,
    ):
        """Receive weights via NCCL broadcast and stream into vLLM.

        Uses a **streaming pipeline** to avoid accumulating a
        full model-weight copy on GPU:

        1. ``CheckpointEngine.receive_weights()`` yields tensors from
           double-buffered NCCL buckets (async generator, GPU tensors).
        2. The async generator is passed **directly** to
           ``VLLMEngine.update_weights()`` which consumes it one tensor at
           a time, copying each into a GPU IPC bucket and flushing to the
           vLLM worker subprocess when the bucket is full.

        Peak GPU overhead is only ~1 IPC bucket (~2 GB) instead of a full
        model copy.

        Args:
            base_sync_done: If True, this is a LoRA-only sync.
            peft_config: PEFT config dict for LoRA adapter loading.

        Returns:
            Number of weights loaded (approximate, from engine log).
        """
        engine = self._get_or_create_checkpoint_engine()

        async def _receive_and_load():
            # Stream NCCL-received tensors directly into vLLM via IPC.
            # VLLMEngine.update_weights accepts an async generator and
            # handles bucket packing + ZMQ transfer internally.
            await self.engine.update_weights(
                engine.receive_weights(),  # async generator — not materialised
                peft_config=peft_config,
                base_sync_done=base_sync_done,
            )

            # After a LoRA sync, refresh the cached LoRARequest in engine
            # so that sample() can use it without per-request list_loras RPC.
            if base_sync_done and peft_config:
                await self.engine.refresh_synced_lora()
            elif not base_sync_done:
                # Base-model sync invalidates any previously synced LoRA.
                self.engine.invalidate_synced_lora()

        self._run_in_loop(_receive_and_load())

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
                self._run_in_loop(self.engine.shutdown())
        except Exception as e:
            logger.warning(f'vLLMSampler engine shutdown error: {e}')

        # 2. Stop the background event loop and join thread
        try:
            if hasattr(self, '_async_loop') and self._async_loop.is_running():
                self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            if hasattr(self, '_async_thread') and self._async_thread.is_alive():
                self._async_thread.join(timeout=5)
        except Exception as e:
            logger.warning(f'vLLMSampler event loop shutdown error: {e}')
