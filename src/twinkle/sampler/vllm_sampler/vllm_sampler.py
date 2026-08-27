# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import atexit
import numpy as np
import os
import threading
from copy import copy
from typing import Any, Dict, List, Optional, Type, Union

from twinkle import DeviceMesh, get_logger, remote_class, remote_function, requires
from twinkle.checkpoint_engine import CheckpointEngineMixin
from twinkle.data_format import InputFeature, SampledSequence, SampleResponse, SamplingParams, Trajectory
from twinkle.hub import HubOperation
from twinkle.patch import Patch, apply_patch
from twinkle.patch.vllm_lora_weights import VLLMLoraWeights
from twinkle.sampler.base import Sampler
from twinkle.utils import Platform

logger = get_logger()


def _convert_ndarray_to_list(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [_convert_ndarray_to_list(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


# max_concurrency: how many sample() calls one worker serves at once. Without it
# Ray runs one method per actor at a time, so concurrent callers queue at the actor
# and never share a batch inside AsyncLLM. 24 is what vLLM reports as the maximum
# concurrency its KV cache holds for this context length; past it vLLM preempts and
# recomputes, which costs more than it gains.
@remote_class(max_concurrency=24)
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

    def _iter_in_loop(self, async_gen_coro):
        """Iterate an async generator running in the background event loop.

        Counterpart to ``_run_in_loop`` for streaming results: schedules the
        async generator on ``self._async_loop`` and yields items to the caller
        via a thread-safe queue.
        """
        import queue as stdlib_queue
        q: stdlib_queue.Queue = stdlib_queue.Queue()
        _SENTINEL = object()

        async def _drain():
            try:
                async for item in async_gen_coro:
                    q.put(item)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_SENTINEL)

        asyncio.run_coroutine_threadsafe(_drain(), self._async_loop)

        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def _create_engine_async(self, engine_cls, model_id, engine_kwargs):
        """Create engine in async context to ensure output_handler starts correctly."""
        return engine_cls(model_id=model_id, **engine_kwargs)

    def encode_trajectory_for_vllm(self,
                                   trajectory: Trajectory,
                                   adapter_name: str = '',
                                   add_generation_prompt=True) -> Dict[str, Any]:
        """Encode trajectory for vLLM.

        Messages should already use transformers standard format (content is List[Dict]).
        ``encode`` preprocesses media refs in-place (to PIL objects).
        """
        template = self.template
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        encoded = template.encode(
            trajectory,
            add_generation_prompt=add_generation_prompt,
        )
        for key in encoded:
            if isinstance(encoded[key], np.ndarray):
                encoded[key] = encoded[key].tolist()
        return encoded

    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        apply_patch(self, patch_cls, **kwargs)

    @staticmethod
    def _extract_multi_modal_data(feat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build vLLM ``multi_modal_data`` dict from feat.

        Checks top-level 'images'/'videos' first, then falls back to
        extracting PIL objects from transformers-standard message content blocks.
        """
        images = feat.get('images')
        videos = feat.get('videos')

        if not images and not videos:
            for msg in feat.get('messages', []):
                content = msg.get('content')
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get('type')
                    if btype == 'image':
                        for key in ('image', 'url', 'path'):
                            if key in block and block[key] is not None:
                                if images is None:
                                    images = []
                                images.append(block[key])
                                break
                    elif btype == 'video':
                        for key in ('video', 'url', 'path'):
                            if key in block and block[key] is not None:
                                if videos is None:
                                    videos = []
                                videos.append(block[key])
                                break

        mm_data = {}
        if images:
            mm_data['image'] = images
        if videos:
            mm_data['video'] = videos
        return mm_data or None

    async def _sample_single(
        self,
        feat: Dict[str, Any],
        sampling_params: SamplingParams,
        lora_request: Optional[Any] = None,
        *,
        multi_modal_data: Optional[Dict[str, Any]] = None,
        logprobs_only: bool = False,
        disable_lora: bool = False,
    ) -> SampleResponse:
        """Sample a single input asynchronously."""
        response = await self.engine.sample(
            prompt=self.template.get_vllm_input_ids(feat['input_ids']) if self.template else feat['input_ids'],
            sampling_params=sampling_params,
            lora_request=lora_request,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=feat.get('mm_processor_kwargs'),
            disable_lora=disable_lora,
        )

        if 'input_ids' not in feat or multi_modal_data:
            if 'input_ids' in feat:
                if len(feat['input_ids']) != len(response.prompt_token_ids):
                    raise RuntimeError(f'Input ids length {len(feat["input_ids"])} does not '
                                       f'match prompt_token_ids length {len(response.prompt_token_ids)}')
            else:
                feat['input_ids'] = response.prompt_token_ids
                feat['labels'] = [-100] * len(response.prompt_token_ids)
        sequences = []
        for seq in response.sequences:
            if logprobs_only:
                new_input_feature = _convert_ndarray_to_list(feat)
                if seq.routed_experts is not None:
                    new_input_feature['routed_experts'] = _convert_ndarray_to_list(seq.routed_experts)
                sampled_seq = SampledSequence(
                    tokens=[],
                    stop_reason=seq.stop_reason,
                    new_input_feature=new_input_feature,
                )
            else:
                new_input_feature = _convert_ndarray_to_list(self.template.concat_input_feature(feat, seq.tokens))
                if seq.routed_experts is not None:
                    new_input_feature['routed_experts'] = _convert_ndarray_to_list(seq.routed_experts)
                sampled_seq = SampledSequence(
                    stop_reason=seq.stop_reason,
                    tokens=seq.tokens,
                    logprobs=seq.logprobs,
                    decoded=self.template.decode(seq.tokens),
                    new_input_feature=new_input_feature,
                )
            sequences.append(sampled_seq)
        return SampleResponse(
            prompt_token_ids=response.prompt_token_ids,
            sequences=sequences,
            prompt_logprobs=response.prompt_logprobs,
            topk_prompt_logprobs=response.topk_prompt_logprobs)

    @remote_function(dispatch='slice_dp', collect='flatten', lazy_collect=False, enable_continous_work=True)
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        return_encoded: bool = False,
        use_base_model: bool = False,
    ) -> List[SampleResponse]:
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
            - Each worker receives already-sliced inputs (e.g., DP4 with 8 inputs -> 2 per worker)
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        elif isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)

        inputs_list = self._normalize_inputs(inputs)

        # Check if inputs are Trajectory (not encoded) - aligned with Model.forward logic
        is_trajectory = 'input_ids' not in inputs_list[0]
        logprobs_only = False
        if sampling_params.max_tokens == 0:
            sampling_params = copy(sampling_params)
            sampling_params.max_tokens = 1
            logprobs_only = True

        multi_modal_data_list = []
        for feat in inputs_list:
            multi_modal_data_list.append(self._extract_multi_modal_data(feat))

        if is_trajectory:
            template = self.template
            assert template is not None, \
                'Use set_template to add a template when trying to input Trajectory'
            encoded_inputs = [
                self.encode_trajectory_for_vllm(traj, adapter_name, not logprobs_only) for traj in inputs_list
            ]
        else:
            encoded_inputs = inputs_list

        lora_request = None
        if adapter_path is not None:
            logger.info(f'Loading LoRA from {adapter_path}')
            adapter_path = HubOperation.download_model(model_id_or_path=adapter_path)
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
                    multi_modal_data=multi_modal_data,
                    logprobs_only=logprobs_only,
                    disable_lora=use_base_model,
                ) for feat, multi_modal_data in zip(encoded_inputs, multi_modal_data_list)
            ]
            return await asyncio.gather(*tasks)

        sample_results = self._run_in_loop(_sample_all())
        return sample_results

    def sample_stream(
        self,
        inputs: Union[InputFeature, Trajectory, Dict[str, Any]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
    ):
        """Stream token deltas as they are generated by the vLLM engine.

        Yields:
            (delta_text: str, finish_reason: str | None) tuples.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        elif isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)

        feat = inputs
        is_trajectory = 'input_ids' not in feat
        if is_trajectory:
            feat = self.encode_trajectory_for_vllm(feat, adapter_name)

        lora_request = None
        if adapter_path is not None:
            adapter_path = HubOperation.download_model(model_id_or_path=adapter_path)
            lora_request = self._run_in_loop(self.engine._get_or_load_lora(adapter_path))

        prompt = self.template.get_vllm_input_ids(feat['input_ids']) if self.template else feat['input_ids']

        yield from self._iter_in_loop(
            self.engine.generate_stream(
                prompt=prompt,
                sampling_params=sampling_params,
                lora_request=lora_request,
            ))

    def sample_stream_to_queue(self, queue, inputs, sampling_params=None, adapter_name='', adapter_path=None):
        """Push streaming deltas to a cross-process Ray queue."""
        from twinkle.server.sampler.backends import stream_to_queue
        stream_to_queue(self, queue, inputs, sampling_params, adapter_name, adapter_path)

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

    @remote_function(dispatch='all', collect='first')
    def reset_mm_cache(self):
        self._run_in_loop(self.engine.reset_mm_cache())

    @remote_function(dispatch='all', collect='first')
    def reset_encoder_cache(self):
        self._run_in_loop(self.engine.reset_encoder_cache())

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def get_state_keys(self):
        return self._run_in_loop(self.engine.get_state_keys())

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

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def load_full_weights_from_path(self, path: Optional[str] = None) -> int:
        """Load a full (non-LoRA) HF checkpoint into the engine's base model.

        Unlike :meth:`receive_weights`, this does **not** involve the training model:
        weights are read from disk and streamed straight into vLLM. That is what lets
        a sampler be restored to a known checkpoint without a trainer round-trip --
        no ``save``/``load`` on the training model, so training weights and optimizer
        state are never touched. Full-parameter training uses the same entry point:
        its checkpoint is a plain HF directory (no ``adapter_config.json``), so it
        replaces the base weights instead of loading as a LoRA adapter.

        Weights are yielded **lazily** one tensor at a time (never materialising a full
        state dict) because ``VLLMEngine.update_weights`` accepts a generator and packs
        tensors into fixed-size transfer buckets itself. Tensors stay on CPU, so the
        engine takes its shared-memory path rather than CUDA IPC.

        Names are passed through untouched: safetensors files already store canonical
        HF names, which is exactly what the worker's ``model.load_weights()`` expects
        (it does the q/k/v -> qkv and gate/up -> gate_up stacking internally).

        Idempotent: repeated calls with the same resolved path are skipped.

        Args:
            path: Local checkpoint dir or a hub model id. Defaults to the ``model_id``
                the sampler was constructed with, i.e. the original pretrained weights.

        Returns:
            1 if weights were (re)loaded, 0 if that path was already loaded.
        """
        import glob
        import json
        from safetensors import safe_open

        path = path or self.model_id
        resolved = path if os.path.exists(path) else HubOperation.download_model(path)
        if getattr(self, '_loaded_full_weights_path', None) == resolved:
            return 0

        # Resolve the shard list eagerly so a bad path fails here rather than
        # part-way through streaming tensors into a live engine.
        index_path = os.path.join(resolved, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            with open(index_path, encoding='utf-8') as f:
                weight_map = json.load(f)['weight_map']
            shards = [os.path.join(resolved, s) for s in sorted(set(weight_map.values()))]
        else:
            shards = sorted(glob.glob(os.path.join(resolved, '*.safetensors')))
        if not shards:
            raise FileNotFoundError(f'No .safetensors weights found under {resolved}')

        def _iter_weights():
            # safe_open + get_tensor reads one tensor at a time (mmap-backed), so peak
            # host memory is a single tensor rather than the whole shard.
            for shard in shards:
                with safe_open(shard, framework='pt', device='cpu') as f:
                    for name in f.keys():
                        yield name, f.get_tensor(name)

        async def _load():
            await self.engine.update_weights(_iter_weights(), peft_config=None, base_sync_done=False)
            # A base-model swap invalidates any previously synced LoRA adapter,
            # mirroring the `not base_sync_done` branch of receive_weights().
            self.engine.invalidate_synced_lora()

        logger.info(f'Loading full-parameter weights into sampler base model from {resolved}')
        self._run_in_loop(_load())
        self._loaded_full_weights_path = resolved
        # Prefixes cached under the previous weights would decode against a model
        # that no longer exists; drop them before the next sample().
        self.reset_prefix_cache()
        logger.info(f'Reloaded base weights from {resolved} ({len(shards)} shard(s))')
        return 1

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
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
