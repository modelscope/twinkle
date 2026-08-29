# Copyright (c) ModelScope Contributors. All rights reserved.
"""SGLang sampler, the sglang counterpart of `vllm_sampler.vLLMSampler`."""
import asyncio
import atexit
import os
import threading
from copy import copy
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from twinkle import DeviceMesh, get_logger, remote_class, remote_function, requires
from twinkle.checkpoint_engine import CheckpointEngineMixin
from twinkle.data_format import InputFeature, SampledSequence, SampleResponse, SamplingParams, Trajectory
from twinkle.hub import HubOperation
from twinkle.patch import Patch, apply_patch
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


@remote_class()
class SGLangSampler(Sampler, CheckpointEngineMixin):
    """An sglang-based sampler using :class:`SGLangEngine`.

    Mirrors :class:`vLLMSampler`: tensor parallelism is taken from the visible devices unless given
    explicitly, sampling is dispatched across data-parallel actors, and weights arrive over the
    checkpoint engine's NCCL/HCCL broadcast.

    Two things differ from the vLLM sampler, both following from sglang's design:

    - Weight transport is simpler. sglang's ``update_weights_from_tensor`` serialises to its own TP
      workers, so there is no ZMQ-plus-CUDA-IPC layer to maintain here; :class:`SGLangEngine` streams
      the received tensors into it a bucket at a time.
    - :meth:`sleep`/:meth:`wake_up` need ``enable_memory_saver=True`` in ``engine_args``, and release by
      region rather than by level. Without it they log a warning and do nothing.

    Not supported:
        LoRA-only weight sync. sglang registers adapters from a path, not from tensors, so
        ``sync_weights(merge_and_sync=False)`` cannot push adapter deltas incrementally -- use the
        default ``sync_weights(merge_and_sync=True)``, which sends merged base weights.
    """

    def __init__(self, model_id: str, engine_args: Dict[str, Any] = None, device_mesh: DeviceMesh = None, **kwargs):
        """Initialize SGLangSampler.

        Args:
            model_id: HuggingFace/ModelScope model ID or local path.
            engine_args: Arguments passed to :class:`SGLangEngine`. If ``tensor_parallel_size`` is not
                specified it is set from the number of visible devices.
            device_mesh: Parallel configuration for data parallelism.
            **kwargs: Additional arguments.
        """
        super().__init__()
        requires('sglang')

        self.model_id = model_id
        self.device_mesh = device_mesh
        # adapter path -> the name sglang knows it by. Registration is not idempotent, so this is what
        # keeps per-input adapters from re-registering the same adapter on every request.
        self._registered_loras: Dict[str, str] = {}

        # A dedicated background event loop, for the same reason vLLMSampler has one -- Ray workers run
        # uvloop, so run_until_complete/asyncio.run are unavailable on the calling thread -- plus one
        # reason specific to sglang: the engine must be *constructed* on the loop that will drive it, so
        # that sglang pins its tokenizer_manager handle loop to that loop. See SGLangEngine's class
        # docstring for what breaks otherwise.
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_event_loop, daemon=True, name='SGLangSampler-EventLoop')
        self._async_thread.start()

        from .sglang_engine import SGLangEngine
        engine_kwargs = engine_args.copy() if engine_args else {}

        if 'tensor_parallel_size' not in engine_kwargs:
            tp_size = 1
            visible_devices = os.environ.get(Platform.visible_device_env(), '')
            if visible_devices:
                num_devices = len([d for d in visible_devices.split(',') if d.strip()])
                if num_devices > 0:
                    tp_size = num_devices
            logger.info(f'sglang TP size: {tp_size}')
            engine_kwargs['tensor_parallel_size'] = tp_size

        # Distinct seed per DP rank, so replicas do not all sample the same continuation.
        # sglang spells this `random_seed`; user-supplied values win.
        if 'random_seed' not in engine_kwargs:
            engine_kwargs['random_seed'] = 42 + Platform.get_rank()

        self.engine: SGLangEngine = self._run_in_loop(self._create_engine_async(SGLangEngine, model_id, engine_kwargs))

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
        """Create the engine on the background loop, which sglang then binds its handle loop to."""
        return engine_cls(model_id=model_id, **engine_kwargs)

    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        apply_patch(self, patch_cls, **kwargs)

    def encode_trajectory_for_sglang(self,
                                     trajectory: Trajectory,
                                     adapter_name: str = '',
                                     add_generation_prompt=True) -> Dict[str, Any]:
        """Encode trajectory for sglang.

        Messages should already use transformers standard format (content is List[Dict]).
        ``encode`` preprocesses media refs in-place (to PIL objects).
        """
        template = self.template
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        encoded = template.encode(trajectory, add_generation_prompt=add_generation_prompt)
        for key in encoded:
            if isinstance(encoded[key], np.ndarray):
                encoded[key] = encoded[key].tolist()
        return encoded

    @staticmethod
    def _extract_image_data(feat: Dict[str, Any]) -> Optional[List[Any]]:
        """Build sglang's ``image_data`` list from feat.

        sglang takes a flat list of images per request -- paths, URLs, base64 or PIL objects -- rather
        than vLLM's ``{'image': [...], 'video': [...]}`` mapping. Videos have no equivalent here, so
        they are not collected.
        """
        images = feat.get('images')

        if not images:
            images = []
            for msg in feat.get('messages', []):
                content = msg.get('content')
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get('type') != 'image':
                        continue
                    for key in ('image', 'url', 'path'):
                        if key in block and block[key] is not None:
                            images.append(block[key])
                            break

        return list(images) or None

    async def _sample_single(
        self,
        feat: Dict[str, Any],
        sampling_params: SamplingParams,
        *,
        image_data: Optional[List[Any]] = None,
        lora_name: Optional[str] = None,
        logprobs_only: bool = False,
    ) -> SampleResponse:
        """Sample a single input asynchronously."""
        response = await self.engine.sample(
            prompt=feat['input_ids'],
            sampling_params=sampling_params,
            image_data=image_data,
            lora_name=lora_name,
        )

        if 'input_ids' not in feat:
            feat['input_ids'] = response.prompt_token_ids
            feat['labels'] = [-100] * len(response.prompt_token_ids)

        sequences = []
        for seq in response.sequences:
            if logprobs_only:
                sampled_seq = SampledSequence(
                    tokens=[],
                    stop_reason=seq.stop_reason,
                    new_input_feature=_convert_ndarray_to_list(feat),
                )
            else:
                sampled_seq = SampledSequence(
                    stop_reason=seq.stop_reason,
                    tokens=seq.tokens,
                    logprobs=seq.logprobs,
                    decoded=self.template.decode(seq.tokens),
                    new_input_feature=_convert_ndarray_to_list(self.template.concat_input_feature(feat, seq.tokens)),
                )
            sequences.append(sampled_seq)
        return SampleResponse(
            prompt_token_ids=response.prompt_token_ids,
            sequences=sequences,
            prompt_logprobs=response.prompt_logprobs,
            topk_prompt_logprobs=response.topk_prompt_logprobs)

    @remote_function(dispatch='slice_dp', collect='flatten', lazy_collect=False)
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        adapter_paths: Optional[List[Optional[str]]] = None,
    ) -> List[SampleResponse]:
        """Sample responses for given inputs.

        Args:
            inputs: Either InputFeature(s) or Trajectory(s).
                - InputFeature: Must contain 'input_ids'. For multimodal, include 'images'.
                - Trajectory: Must contain 'messages'. Requires template to be set.

            sampling_params: Sampling parameters. ``num_samples`` on these controls how many
                completions are generated per prompt.

            adapter_name: LoRA adapter name to sample with; must already be registered.

            adapter_path: Optional LoRA adapter path, registered under ``adapter_name`` before sampling.

            adapter_paths: Per-input LoRA paths, one entry per input (``None`` entries sample from the
                base model). sglang carries the adapter per request (``lora_path``) and admits up to
                ``max_loras_per_batch`` of them in one running batch, so several adapters can be served
                concurrently without serializing. Mutually exclusive with ``adapter_path``.

        Returns:
            One SampleResponse per input, each holding ``sampling_params.num_samples`` sequences.

        Note:
            In Ray mode with multiple workers (DP > 1) data is sliced by DP rank
            (dispatch='slice_dp'), so each worker receives its own shard of ``inputs``.
            ``adapter_paths`` is a list too and is sliced in lockstep, staying aligned with ``inputs``.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        elif isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)

        inputs_list = self._normalize_inputs(inputs)
        if adapter_paths is not None:
            if adapter_path is not None:
                raise ValueError('Pass either adapter_path (one adapter for the whole call) or adapter_paths '
                                 '(one per input), not both.')
            if len(adapter_paths) != len(inputs_list):
                raise ValueError(f'adapter_paths has {len(adapter_paths)} entries but there are '
                                 f'{len(inputs_list)} inputs; they must correspond one-to-one so that DP '
                                 'slicing keeps them aligned.')

        is_trajectory = 'input_ids' not in inputs_list[0]
        logprobs_only = False
        if sampling_params.max_tokens == 0:
            # sglang has no zero-token request, so ask for one token and drop it.
            sampling_params = copy(sampling_params)
            sampling_params.max_tokens = 1
            logprobs_only = True

        image_data_list = [self._extract_image_data(feat) for feat in inputs_list]

        if is_trajectory:
            assert self.template is not None, \
                'Use set_template to add a template when trying to input Trajectory'
            encoded_inputs = [
                self.encode_trajectory_for_sglang(traj, adapter_name, not logprobs_only) for traj in inputs_list
            ]
        else:
            encoded_inputs = inputs_list

        if adapter_paths is not None:
            lora_names = [self._register_lora(path) for path in adapter_paths]
        else:
            registered = self._register_lora(adapter_path, adapter_name)
            lora_names = [registered or (adapter_name or None)] * len(encoded_inputs)

        async def _sample_all():
            tasks = [
                self._sample_single(
                    feat,
                    sampling_params,
                    image_data=image_data,
                    lora_name=lora_name,
                    logprobs_only=logprobs_only,
                ) for feat, image_data, lora_name in zip(encoded_inputs, image_data_list, lora_names)
            ]
            return await asyncio.gather(*tasks)

        return self._run_in_loop(_sample_all())

    def _register_lora(self, adapter_path: Optional[str], adapter_name: Optional[str] = None) -> Optional[str]:
        """Register an adapter with sglang and return the name to send as ``lora_path``.

        Cached by path: ``load_lora_adapter`` on an already-registered name is an error, and per-input
        adapters would otherwise re-register the same adapter once per request. When no name is given
        one is generated, because several distinct adapters cannot all be called 'default'.
        """
        if adapter_path is None:
            return None
        if adapter_path in self._registered_loras:
            return self._registered_loras[adapter_path]
        lora_name = adapter_name or f'lora_{len(self._registered_loras)}'
        logger.info(f'Loading LoRA {lora_name!r} from {adapter_path}')
        local_path = HubOperation.download_model(model_id_or_path=adapter_path)
        self._run_in_loop(self.engine.load_lora_adapter(lora_name=lora_name, lora_path=local_path))
        self._registered_loras[adapter_path] = lora_name
        return lora_name

    def sample_stream(
        self,
        inputs: Union[InputFeature, Trajectory, Dict[str, Any]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
    ):
        """Yield ``(delta_text, finish_reason)`` for a single input, mirroring ``vLLMSampler``.

        Deliberately undecorated: streaming is inherently one request on one engine, so there is
        nothing for ``slice_dp`` to slice, and a generator cannot cross Ray's result boundary.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        elif isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)

        feat = inputs
        if 'input_ids' not in feat:
            feat = self.encode_trajectory_for_sglang(feat, adapter_name)
        lora_name = self._register_lora(adapter_path, adapter_name)

        yield from self._iter_in_loop(
            self.engine.generate_stream(
                prompt=feat['input_ids'],
                sampling_params=sampling_params,
                lora_name=lora_name,
            ))

    def _iter_in_loop(self, async_gen):
        """Drain an async generator running on the background loop into a plain iterator."""
        import queue as stdlib_queue
        q: stdlib_queue.Queue = stdlib_queue.Queue()
        sentinel = object()

        async def _drain():
            try:
                async for item in async_gen:
                    q.put(item)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(sentinel)

        asyncio.run_coroutine_threadsafe(_drain(), self._async_loop)
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    @remote_function(dispatch='all', collect='first')
    def sleep(self, level: int = 1) -> None:
        """Release device memory for colocate mode. Needs ``enable_memory_saver=True``."""
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
        weights=None,
    ):
        """Receive weights from the trainer and stream them into sglang.

        With no ``weights`` argument, the checkpoint engine supplies an async generator from its
        NCCL/HCCL/CUDA IPC transport. A local naive caller can instead provide the model's synchronous
        generator directly. Either iterator is handed straight to :meth:`SGLangEngine.update_weights`,
        which consumes it one tensor at a time into a bucket and forwards each full bucket to sglang.
        Peak extra memory is one bucket rather than a second copy of the model.

        Args:
            base_sync_done: If True, this would be a LoRA-only sync.
            peft_config: PEFT config dict for LoRA adapter loading.
            weights: Optional synchronous/asynchronous weight iterator. If
                omitted, weights are received from the checkpoint engine.

        Raises:
            NotImplementedError: For a LoRA-only sync; see the class docstring.
        """
        if weights is None:
            engine = self._get_or_create_checkpoint_engine()
            weights = engine.receive_weights()

        async def _receive_and_load():
            await self.engine.update_weights(
                weights,  # async/sync generator — not materialised
                peft_config=peft_config,
                base_sync_done=base_sync_done,
            )

        self._run_in_loop(_receive_and_load())

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def shutdown(self):
        """Shut down the sglang engine and the background event loop.

        Registered via atexit so it runs on process exit before GC tears objects down in an
        unpredictable order. Idempotent.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True

        try:
            if getattr(self, 'engine', None) is not None:
                self._run_in_loop(self.engine.shutdown())
        except Exception as e:
            logger.warning(f'SGLangSampler engine shutdown error: {e}')

        try:
            if hasattr(self, '_async_loop') and self._async_loop.is_running():
                self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            if hasattr(self, '_async_thread') and self._async_thread.is_alive():
                self._async_thread.join(timeout=5)
        except Exception as e:
            logger.warning(f'SGLangSampler event loop shutdown error: {e}')
