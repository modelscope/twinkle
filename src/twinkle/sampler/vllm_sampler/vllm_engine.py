# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import torch
import uuid
from typing import Any, Dict, List, Optional, Union

from twinkle import get_logger
from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingParams, StopReason
from twinkle.sampler.base_engine import BaseSamplerEngine
from twinkle.utils import Platform

logger = get_logger()


def get_vllm_max_lora_rank(lora_rank: int) -> int:
    """Get the nearest allowed vLLM LoRA rank."""
    from typing import get_args
    try:
        from vllm.config.lora import MaxLoRARanks
        allowed_ranks = sorted(get_args(MaxLoRARanks))
        for rank in allowed_ranks:
            if lora_rank <= rank:
                return rank
        return allowed_ranks[-1]
    except ImportError:
        # Fallback for older vLLM versions
        return lora_rank


class VLLMEngine(BaseSamplerEngine):
    """
    A vLLM-based inference engine for RL training.

    This engine uses vLLM v1's AsyncLLM and supports:
    - Tinker-compatible sample() API with logprobs
    - Multi-tenant LoRA adapters for client-server mode
    - Weight synchronization via load_weights (colocated) or CUDA IPC
    - Sleep/wake_up for GPU memory management in colocated training

    Deployment scenarios:
    1. Standalone server (client-server): Multi-tenant, LoRA adapters indexed by URI
    2. Colocated with training (Ray): Single-tenant, weight sync via load_weights
    """

    def __init__(
        self,
        model_id: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        enable_lora: bool = True,
        max_loras: int = 1,
        max_lora_rank: int = 32,
        enable_sleep_mode: bool = False,
        enable_prefix_caching: bool = False,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        dtype: str = 'auto',
        quantization: Optional[str] = None,
        load_format: str = 'auto',
        logprobs_mode: Optional[str] = None,
        **kwargs,
    ):
        from twinkle.hub import HubOperation
        model_id = HubOperation.download_model(model_id)
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.enable_lora = enable_lora
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.enable_sleep_mode = enable_sleep_mode
        self.enable_prefix_caching = enable_prefix_caching
        self.enforce_eager = enforce_eager
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.quantization = quantization
        self.load_format = load_format
        self.logprobs_mode = logprobs_mode or 'processed_logprobs'
        self.engine_kwargs = kwargs or {}

        self._lora_request_cache: Dict[str, Any] = {}
        self._next_lora_id = 1

        # Cached LoRARequest for the RL-training synced LoRA.
        # Built lazily by ``refresh_synced_lora()`` after CheckpointEngine
        # finishes a LoRA sync, so ``sample()`` never needs to call
        # ``list_loras()`` per request.
        self._synced_lora_request: Optional[Any] = None

        # Initialize engine
        self.engine = self._create_engine()

        # Tokenizer is lazy loaded via get_tokenizer()
        self._tokenizer = None

    def _create_engine(self):
        """Create and return the vLLM engine."""
        os.environ['VLLM_USE_V1'] = '1'
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM

        # Build engine config
        engine_config = {
            'model': self.model_id,
            'tensor_parallel_size': self.tensor_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'max_num_seqs': self.max_num_seqs,
            'trust_remote_code': self.trust_remote_code,
            'enforce_eager': self.enforce_eager,
            'dtype': self.dtype,
            'load_format': self.load_format,
            'disable_log_stats': True,
        }

        if self.tensor_parallel_size > 1:
            engine_config['distributed_executor_backend'] = 'mp'

        if self.max_model_len is not None:
            engine_config['max_model_len'] = self.max_model_len

        if self.quantization is not None:
            engine_config['quantization'] = self.quantization

        if self.enable_prefix_caching:
            engine_config['enable_prefix_caching'] = True

        if self.enable_sleep_mode:
            engine_config['enable_sleep_mode'] = True

        if self.logprobs_mode is not None:
            engine_config['logprobs_mode'] = self.logprobs_mode

        if self.enable_lora:
            engine_config['enable_lora'] = True
            engine_config['max_loras'] = self.max_loras
            engine_config['max_lora_rank'] = get_vllm_max_lora_rank(self.max_lora_rank)

        # Enable worker extension for weight synchronization
        engine_config['worker_extension_cls'] = (
            'twinkle.sampler.vllm_sampler.vllm_worker_extension.TwinkleWorkerExtension')

        engine_config.update(self.engine_kwargs)
        valid_args = inspect.signature(AsyncEngineArgs).parameters.keys()
        filtered_engine_config = {k: v for k, v in engine_config.items() if k in valid_args}
        invalid_args = set(engine_config.keys()) - set(valid_args)
        if invalid_args:
            logger.warning(f'VLLMEngine: Filtered out invalid arguments: {invalid_args}')
        # Create engine using vLLM v1 API
        engine_args = AsyncEngineArgs(**filtered_engine_config)
        vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)

        engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )

        logger.info(f'VLLMEngine initialized: model={self.model_id}')
        return engine

    async def get_tokenizer(self):
        """Get the tokenizer asynchronously."""
        if self._tokenizer is None:
            self._tokenizer = await self.engine.get_tokenizer()
        return self._tokenizer

    # =========================================================================
    # Core Sampling API
    # =========================================================================

    async def sample(
        self,
        prompt_token_ids: List[int],
        sampling_params: Union[SamplingParams, Dict[str, Any]],
        num_samples: int = 1,
        logprobs: bool = True,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_request: Optional[Any] = None,
        request_id: Optional[str] = None,
        priority: int = 0,
        *,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
    ) -> SampleResponse:
        """
        Sample completions from the model.

        This is the core API aligned with tinker's sampling interface.

        Args:
            prompt_token_ids: Input token IDs.
            sampling_params: Sampling parameters (tinker.types.SamplingParams or dict).
            num_samples: Number of samples to generate.
            logprobs: Whether to return log probabilities for generated tokens.
            include_prompt_logprobs: Whether to compute logprobs on prompt tokens.
            topk_prompt_logprobs: If > 0, returns top-k logprobs for each prompt token.
            lora_request: LoRARequest for sampling.
            request_id: Optional request ID for tracking.
            priority: Request priority (higher = more urgent).
            images: Optional list of images for multimodal models.
                    Can be PIL.Image, file paths, URLs, or bytes.
            videos: Optional list of videos for multimodal models.
                    Can be file paths or list of frames.

        Returns:
            tinker.types.SampleResponse containing sequences and optionally prompt_logprobs.
        """
        from vllm.inputs import TokensPrompt

        # Convert to vLLM params
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)
        prompt_logprobs_k = topk_prompt_logprobs if topk_prompt_logprobs > 0 else (1 if include_prompt_logprobs else 0)
        vllm_params = sampling_params.to_vllm(
            num_samples=num_samples,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs_k,
        )

        # Build request
        if request_id is None:
            request_id = uuid.uuid4().hex

        # Build multi_modal_data if images or videos provided
        multi_modal_data = {}
        if images:
            multi_modal_data['image'] = images
        if videos:
            multi_modal_data['video'] = videos

        # Build prompt (with or without multimodal data)
        if multi_modal_data:
            prompt = TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data=multi_modal_data,
            )
        else:
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        if lora_request is not None and not self.enable_lora:
            logger.warning('lora_request provided but enable_lora is '
                           'False — LoRA will be ignored for this request')
            lora_request = None

        if lora_request is None and self._synced_lora_request is not None:
            # RL training path: use the LoRA synced via CheckpointEngine.
            # The request object is cached after the first ``list_loras``
            # check to avoid per-request RPC overhead.
            lora_request = self._synced_lora_request

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=vllm_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        # Get final result
        result = None
        async for output in generator:
            result = output

        if result is None:
            raise RuntimeError('Sampling did not produce a result')

        # Extract sequences
        sequences = []
        for output in result.outputs:
            token_ids = list(output.token_ids)

            # Extract logprobs
            seq_logprobs = None
            if output.logprobs is not None:
                seq_logprobs = []
                for i, lp in enumerate(output.logprobs):
                    if i < len(token_ids) and token_ids[i] in lp:
                        seq_logprobs.append(lp[token_ids[i]].logprob)

            # Map finish_reason to StopReason
            stop_reason: StopReason = 'length'
            if output.finish_reason in ('stop', 'eos_token'):
                stop_reason = 'stop'

            sequences.append(SampledSequence(
                stop_reason=stop_reason,
                tokens=token_ids,
                logprobs=seq_logprobs,
            ))

        # Extract prompt logprobs if requested
        result_prompt_logprobs = None
        result_topk_prompt_logprobs = None
        if prompt_logprobs_k > 0 and result.prompt_logprobs is not None:
            result_prompt_logprobs = []
            result_topk_prompt_logprobs = []

            for i, lp_dict in enumerate(result.prompt_logprobs):
                if lp_dict is None:
                    result_prompt_logprobs.append(None)
                    result_topk_prompt_logprobs.append(None)
                    continue

                # Get logprob for the actual token
                if i < len(prompt_token_ids):
                    token_id = prompt_token_ids[i]
                    if token_id in lp_dict:
                        lp_obj = lp_dict[token_id]
                        result_prompt_logprobs.append(lp_obj.logprob if hasattr(lp_obj, 'logprob') else lp_obj)
                    else:
                        result_prompt_logprobs.append(None)
                else:
                    result_prompt_logprobs.append(None)

                # Get top-k logprobs
                sorted_items = sorted(
                    lp_dict.items(), key=lambda x: -(x[1].logprob
                                                     if hasattr(x[1], 'logprob') else x[1]))[:prompt_logprobs_k]
                result_topk_prompt_logprobs.append([(tid, lp_obj.logprob if hasattr(lp_obj, 'logprob') else lp_obj)
                                                    for tid, lp_obj in sorted_items])

        return SampleResponse(
            sequences=sequences,
            prompt_logprobs=result_prompt_logprobs,
            topk_prompt_logprobs=result_topk_prompt_logprobs,
        )

    # -----------------------------------------------------------------
    # RL-training synced LoRA helpers
    # -----------------------------------------------------------------

    async def refresh_synced_lora(self) -> None:
        """Refresh the cached LoRARequest for the RL-training synced LoRA.

        Called by ``vLLMSampler.receive_weights`` after a successful LoRA
        sync via CheckpointEngine.  Subsequent ``sample()`` calls will use
        the cached request object without any ``list_loras()`` RPC.
        """
        from vllm.lora.request import LoRARequest

        from twinkle.sampler.vllm_sampler.vllm_worker_extension import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH
        loaded = await self.engine.list_loras()
        if VLLM_LORA_INT_ID in loaded:
            self._synced_lora_request = LoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
            )
        else:
            self._synced_lora_request = None

    def invalidate_synced_lora(self) -> None:
        """Clear the cached synced LoRA request.

        Called before a new base-model weight sync that replaces the model
        weights (invalidating any previously loaded LoRA).
        """
        self._synced_lora_request = None

    def _generate_lora_id(self) -> int:
        """Generate a unique LoRA int ID."""
        lora_id = self._next_lora_id
        self._next_lora_id += 1
        return lora_id

    async def _get_or_load_lora(
        self,
        lora_path: str,
    ):
        """Get or load a LoRA adapter from *lora_path*.

        Args:
            lora_path: Resolved filesystem path to the LoRA adapter directory.

        Returns:
            ``LoRARequest`` or ``None`` if loading fails.
        """
        from vllm.lora.request import LoRARequest

        # Fast path: return cached request for this path.
        if lora_path in self._lora_request_cache:
            return self._lora_request_cache[lora_path]

        if not os.path.exists(lora_path):
            logger.error(f'LoRA path does not exist: {lora_path}')
            return None

        config_path = os.path.join(lora_path, 'adapter_config.json')
        if not os.path.exists(config_path):
            logger.error(f'adapter_config.json not found in {lora_path}')
            return None

        lora_int_id = self._generate_lora_id()
        lora_name = str(lora_int_id)

        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=lora_path,
        )

        try:
            await self.engine.add_lora(lora_request)
            self._lora_request_cache[lora_path] = lora_request
            return lora_request
        except Exception as e:
            logger.error(f'Failed to load LoRA from {lora_path}: {e}')
            return None

    async def sleep(self, level: int = 2) -> None:
        """
        Offload weights and/or KV cache from GPU memory.

        Used in colocated mode to free GPU memory for training.

        Args:
            level: Sleep level.
                1 = offload KV cache only
                2 = offload KV cache and weights
        """
        if not self.enable_sleep_mode:
            logger.warning('sleep_mode not enabled, skipping sleep')
            return

        await self.engine.sleep(level=level)
        logger.debug(f'Engine sleeping at level {level}')

    async def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """
        Resume weights and/or KV cache to GPU memory.

        Used in colocated mode before inference.

        Args:
            tags: What to resume. Options: ['weights', 'kv_cache'].
                  If None, resumes both.
            reload_weights: If True and level 2 sleep was used (weights discarded),
                  reload weights from disk via collective_rpc("reload_weights").

        """
        if not self.enable_sleep_mode:
            logger.warning('sleep_mode not enabled, skipping wake_up')
            return

        if tags is None:
            tags = ['weights', 'kv_cache']

        await self.engine.wake_up(tags=tags)

        logger.debug(f'Engine waking up with tags: {tags}')

    async def reset_prefix_cache(self) -> None:
        await self.engine.reset_prefix_cache()

    async def update_weights(
        self,
        weights,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        bucket_size_mb: int = 2048,
        **kwargs,
    ) -> None:
        """Update model weights via ZMQ + CUDA IPC to worker extension.

        Accepts **either** a ``dict[str, Tensor]`` (legacy) **or** an async
        generator / sync generator of ``(name, tensor)`` pairs (streaming).

        The streaming path avoids accumulating a full model copy on GPU:
        tensors are consumed one-by-one from the generator, copied into a
        GPU IPC bucket, and flushed to the vLLM worker subprocess when the
        bucket is full.

        Args:
            weights: Weights to transfer.  ``dict[str, Tensor]`` or
                ``(Async)Generator[tuple[str, Tensor], ...]``.
            peft_config: PEFT config dict for LoRA adapter loading.
            base_sync_done: If True with peft_config, load as LoRA adapter.
            bucket_size_mb: Size of transfer bucket in MB.
        """
        import asyncio
        import gc
        import time
        import zmq
        from vllm.platforms import current_platform

        start_time = time.time()

        # Normalise *weights* into an async iterator regardless of input type.
        if isinstance(weights, dict):

            async def _dict_iter():
                for item in weights.items():
                    yield item

            weight_aiter = _dict_iter()
        elif hasattr(weights, '__aiter__'):
            weight_aiter = weights.__aiter__()
        else:
            # sync generator / iterable
            async def _sync_iter():
                for item in weights:
                    yield item

            weight_aiter = _sync_iter()

        # Peek first tensor to detect device (GPU → IPC, CPU → SHM).
        try:
            first_name, first_tensor = await weight_aiter.__anext__()
        except StopAsyncIteration:
            logger.warning('update_weights called with empty weights')
            return

        use_gpu_ipc = first_tensor.is_cuda
        use_shm = not use_gpu_ipc

        # fix: On NPU, current_platform.get_device_uuid may be unimplemented and break receive_weights flow.
        # fix: Route through platform-level fallback so IPC socket name remains stable.
        # Get device UUID for ZMQ handle.
        # For NPU, this is resolved from `npu-smi info` Bus-Id when needed.
        device_uuid = Platform.get_vllm_device_uuid(0)
        zmq_handle = f'ipc:///tmp/twinkle-ipc-{device_uuid}.sock'

        bucket_size = bucket_size_mb << 20

        # Create transfer buffer
        buffer = None
        shm = None

        if use_gpu_ipc:
            from torch.multiprocessing.reductions import reduce_tensor
            buffer = torch.empty(bucket_size, dtype=torch.uint8, device=first_tensor.device)
            ipc_handle = reduce_tensor(buffer)
        else:
            from multiprocessing import shared_memory
            shm_name = f'twinkle_weights_{uuid.uuid4().hex}'
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=bucket_size)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

        # Setup ZMQ socket FIRST (bind before worker connects)
        zmq_ctx = zmq.Context()
        socket = zmq_ctx.socket(zmq.REQ)
        socket.bind(zmq_handle)

        loop = asyncio.get_running_loop()

        # Non-blocking ZMQ helpers — run blocking socket ops in the
        # default executor so the event loop stays responsive.  This is
        # critical when TP > 1: collective_rpc is an async task on the
        # same loop, and blocking socket.recv() would prevent it from
        # being scheduled, causing a deadlock.
        def _zmq_send_recv(payload):
            socket.send_pyobj(payload)
            return socket.recv()

        # Launch worker side concurrently
        worker_task = asyncio.ensure_future(
            self.engine.collective_rpc(
                'update_weights_from_ipc',
                kwargs={
                    'peft_config': peft_config,
                    'base_sync_done': base_sync_done,
                    'use_shm': use_shm,
                },
            ))

        # Send IPC/SHM handle, wait for worker ready (non-blocking)
        handle_payload = ipc_handle if use_gpu_ipc else {'name': shm_name, 'size': bucket_size}
        await loop.run_in_executor(None, _zmq_send_recv, handle_payload)

        # Stream weights into buckets and send to worker
        async def _chain_first():
            """Re-inject the peeked first tensor, then yield the rest."""
            yield first_name, first_tensor
            async for item in weight_aiter:
                yield item

        offset = 0
        bucket_meta: dict = {}
        n_weights = 0

        async for name, weight in _chain_first():
            if use_shm and weight.is_cuda:
                weight = weight.cpu()

            if weight.nbytes > bucket_size:
                raise ValueError(f'Weight {name} ({weight.nbytes / (1 << 20):.1f} MB) exceeds '
                                 f'bucket size ({bucket_size_mb} MB). Increase bucket_size_mb.')

            # Flush current bucket if it would overflow
            if offset + weight.nbytes > bucket_size:
                if use_gpu_ipc:
                    torch.cuda.synchronize()
                await loop.run_in_executor(
                    None,
                    _zmq_send_recv,
                    {
                        'bucket_meta': bucket_meta,
                        'is_last': False
                    },
                )
                bucket_meta = {}
                offset = 0

            bucket_meta[name] = {
                'name': name,
                'shape': weight.shape,
                'dtype': weight.dtype,
                'offset': offset,
            }
            buffer[offset:offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += weight.nbytes
            n_weights += 1

        # Send last bucket
        if use_gpu_ipc:
            torch.cuda.synchronize()
        await loop.run_in_executor(
            None,
            _zmq_send_recv,
            {
                'bucket_meta': bucket_meta,
                'is_last': True
            },
        )

        # Wait for worker to finish loading
        await worker_task

        # Clean up
        socket.close()
        zmq_ctx.term()
        del buffer
        if shm is not None:
            shm.close()
            shm.unlink()
            del shm
        gc.collect()

        elapsed = time.time() - start_time
        mode = 'LoRA' if base_sync_done and peft_config else 'base'
        logger.info(f'Updated {n_weights} {mode} weights via '
                    f"{'IPC' if use_gpu_ipc else 'SHM'} in {elapsed:.2f}s")

    async def shutdown(self) -> None:
        """Shutdown the vLLM engine and release all resources.

        This method should be called before the process exits to ensure
        proper cleanup of the vLLM AsyncLLM engine and its subprocesses.
        """
        import gc

        logger.info('Shutting down VLLMEngine...')

        if self.engine is not None:
            try:
                # vLLM v1 AsyncLLM has shutdown() method
                if hasattr(self.engine, 'shutdown'):
                    await self.engine.shutdown()
                elif hasattr(self.engine, 'engine_core'):
                    # For older versions, try to stop engine core
                    if hasattr(self.engine.engine_core, 'shutdown'):
                        await self.engine.engine_core.shutdown()
            except Exception as e:
                logger.warning(f'Error during engine shutdown: {e}')
            finally:
                self.engine = None

        # Clear LoRA state
        self._lora_request_cache.clear()

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()

        logger.info('VLLMEngine shutdown complete')
