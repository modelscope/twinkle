# Copyright (c) ModelScope Contributors. All rights reserved.
"""SGLang inference engine, the sglang counterpart of `vllm_engine.VLLMEngine`."""
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from twinkle import get_logger, requires
from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingParams, StopReason
from twinkle.sampler.base_engine import BaseSamplerEngine

logger = get_logger()

# What sglang reports as a finish reason, in twinkle's vocabulary. sglang gives either a bare string or
# a dict whose `type` holds one of these.
_FINISH_REASONS: Dict[str, StopReason] = {
    'stop': 'stop',
    'eos_token': 'stop',
    'stop_str': 'stop',
    'length': 'length',
    'abort': 'abort',
}


class SGLangEngine(BaseSamplerEngine):
    """Token-level inference on sglang's offline engine.

    Same role as :class:`VLLMEngine`: take token ids, return a
    :class:`~twinkle.data_format.SampleResponse`. The differences that matter are all in how sglang
    states things, not in what it can do:

    - Sampling parameters are a plain dict, and the token budget is called ``max_new_tokens``. See
      :meth:`SamplingParams.to_sglang`.
    - Logprobs are requested per request (``return_logprob``/``top_logprobs_num``/``logprob_start_len``)
      rather than through the sampling parameters, and come back as ``(logprob, token_id, text)`` --
      logprob first, the opposite of vLLM's mapping.
    - Weight updates land through ``update_weights_from_tensor``, which does its own serialisation out to
      the TP workers. The whole ZMQ-and-CUDA-IPC layer :class:`VLLMEngine` has to implement by hand is
      internal to sglang here.
    - Releasing memory is by region (``kv_cache``/``weights``) rather than by level, and needs the engine
      to have been started with ``enable_memory_saver=True``.

    Event loop contract:
        Every method here goes through ``engine.tokenizer_manager`` rather than the same-named methods on
        ``sglang.Engine``. This is not stylistic. ``sglang.Engine``'s non-generate methods are sync
        wrappers shaped like ``self.loop.run_until_complete(self.tokenizer_manager.foo(...))``, and
        ``Engine.__init__`` binds ``self.loop`` to the running loop if there is one. So once this engine
        is built inside the loop that drives it -- which is what :class:`SGLangSampler` does, and what
        sglang needs in order to pin its tokenizer_manager handle loop to that same loop -- every one of
        those wrappers raises ``RuntimeError: This event loop is already running``. Awaiting the
        tokenizer_manager coroutines directly keeps all traffic on the one loop, which is also the only
        arrangement sglang's handle loop tolerates.

        The corollary is a construction requirement: **build this engine inside the event loop that will
        drive it.** Building it outside gives sglang a second, idle loop, and requests awaited on the
        driving loop then deadlock against a handle loop attached elsewhere.
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        enable_memory_saver: bool = False,
        **engine_kwargs,
    ):
        """Start the sglang engine.

        Args:
            model_id: Model path or hub id, passed as sglang's ``model_path``.
            tensor_parallel_size: sglang's ``tp_size``.
            data_parallel_size: sglang's ``dp_size``. Note this is sglang-internal data parallelism,
                distinct from the data parallelism twinkle does across sampler actors.
            enable_memory_saver: Required for :meth:`sleep`/:meth:`wake_up` to do anything. Off by
                default because it costs a memory-saver allocator on every allocation.
            **engine_kwargs: Passed through to ``sglang.Engine``.
        """
        requires('sglang')
        import sglang as sgl

        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self._tokenizer = None
        self._memory_saver_enabled = enable_memory_saver
        # Which memory regions sleep() gave up, so wake_up() can ask for exactly those back.
        self._offloaded_tags: List[str] = []

        kwargs = dict(
            model_path=model_id,
            tp_size=tensor_parallel_size,
            dp_size=data_parallel_size,
            enable_memory_saver=enable_memory_saver,
            **engine_kwargs,
        )
        logger.info(f'Creating sglang Engine: tp_size={tensor_parallel_size}, dp_size={data_parallel_size}')
        self.engine = sgl.Engine(**kwargs)

    @property
    def _manager(self):
        """sglang's tokenizer_manager, which is where the awaitable form of every request lives.

        See the event loop contract in the class docstring for why this is used in place of the
        same-named ``sglang.Engine`` methods.
        """
        return self.engine.tokenizer_manager

    async def get_tokenizer(self):
        if self._tokenizer is None:
            tokenizer = getattr(self.engine, 'tokenizer_manager', None)
            tokenizer = getattr(tokenizer, 'tokenizer', None)
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self._tokenizer = tokenizer
        return self._tokenizer

    # =========================================================================
    # Core Sampling API
    # =========================================================================

    async def sample(
        self,
        prompt: Union[List[int], str],
        sampling_params: Union[SamplingParams, Dict[str, Any]],
        request_id: Optional[str] = None,
        *,
        image_data: Optional[Any] = None,
        lora_name: Optional[str] = None,
        **kwargs,
    ) -> SampleResponse:
        """Sample completions for one prompt.

        Args:
            prompt: Token ids, or text. Token ids are the path the sampler uses; text is tokenised
                afterwards only to fill in ``prompt_token_ids``, which sglang does not return.
            sampling_params: Twinkle sampling parameters, or a dict of them.
            request_id: Passed to sglang as ``rid``.
            image_data: Images for multimodal models, in any form sglang accepts (path, URL, base64,
                PIL object), one entry per image.
            lora_name: Name of a LoRA adapter already registered by ``load_lora_adapter``.
            **kwargs: Extra sampling parameters, merged into the dict sglang receives.
        """
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)
        logprobs_k = sampling_params.logprobs or 0
        prompt_logprobs_k = sampling_params.prompt_logprobs or 0

        request: Dict[str, Any] = {'sampling_params': sampling_params.to_sglang(**kwargs)}
        if isinstance(prompt, str):
            request['prompt'] = prompt
        else:
            request['input_ids'] = list(prompt)

        if logprobs_k or prompt_logprobs_k:
            request['return_logprob'] = True
            top_k = max(logprobs_k, prompt_logprobs_k)
            if top_k > 1:
                request['top_logprobs_num'] = top_k
            # -1 keeps sglang on the generated tokens only; 0 makes it score the prompt as well, which
            # is the more expensive path and only worth asking for when prompt logprobs were requested.
            request['logprob_start_len'] = 0 if prompt_logprobs_k else -1
        if image_data is not None:
            request['image_data'] = image_data
        if lora_name:
            request['lora_path'] = lora_name
        if request_id is not None:
            request['rid'] = request_id

        outputs = await self.engine.async_generate(**request)
        # With `n > 1` sglang returns one dict per sample; with n == 1, a single dict.
        outputs = outputs if isinstance(outputs, list) else [outputs]

        prompt_token_ids = request.get('input_ids')
        if prompt_token_ids is None:
            tokenizer = await self.get_tokenizer()
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

        sequences = []
        for output in outputs:
            meta = output.get('meta_info') or {}
            tokens = self._output_token_ids(meta, output, await self.get_tokenizer())
            sequences.append(
                SampledSequence(
                    stop_reason=self._map_finish_reason(meta.get('finish_reason')),
                    tokens=tokens,
                    logprobs=self._sequence_logprobs(meta, tokens, logprobs_k),
                ))

        # Prompt logprobs describe the shared prompt, so they are the same on every sample.
        first_meta = (outputs[0].get('meta_info') or {}) if outputs else {}
        prompt_logprobs, topk_prompt_logprobs = self._prompt_logprobs(first_meta, prompt_logprobs_k)
        return SampleResponse(
            sequences=sequences,
            prompt_token_ids=list(prompt_token_ids),
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )

    async def generate_stream(self,
                              prompt: Union[List[int], str],
                              sampling_params: Union[SamplingParams, Dict[str, Any]],
                              lora_name: Optional[str] = None,
                              **kwargs):
        """Yield ``(delta_text, finish_reason)`` as sglang produces tokens.

        sglang's streaming sends the *cumulative* text on every chunk, not the increment, so the delta
        has to be derived by remembering how much was already emitted. Forwarding the raw chunk instead
        would replay the whole completion on each event.

        Yields:
            ``(delta_text, None)`` per chunk, then a final ``('', reason)`` once sglang reports one.
        """
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)

        request: Dict[str, Any] = {'sampling_params': sampling_params.to_sglang(**kwargs), 'stream': True}
        if isinstance(prompt, str):
            request['prompt'] = prompt
        else:
            request['input_ids'] = list(prompt)
        if lora_name:
            request['lora_path'] = lora_name

        emitted = 0
        finish_reason = None
        async for chunk in await self.engine.async_generate(**request):
            text = chunk.get('text') or ''
            if len(text) > emitted:
                yield text[emitted:], None
                emitted = len(text)
            reason = (chunk.get('meta_info') or {}).get('finish_reason')
            if reason:
                finish_reason = self._map_finish_reason(reason)
        yield '', finish_reason or 'stop'

    @staticmethod
    def _map_finish_reason(reason: Any) -> StopReason:
        """Twinkle's stop reason for whatever sglang reported.

        sglang gives a dict (``{'type': 'stop', 'matched': ...}``) in most versions and a bare string in
        others. Anything unrecognised becomes ``error`` rather than being quietly called a normal stop.
        """
        if isinstance(reason, dict):
            reason = reason.get('type')
        if reason is None:
            return 'stop'
        return _FINISH_REASONS.get(str(reason), 'error')

    @staticmethod
    def _output_token_ids(meta: Dict[str, Any], output: Dict[str, Any], tokenizer) -> List[int]:
        """The generated token ids.

        sglang returns text, and the ids only as a side effect of asking for logprobs, so without
        logprobs they have to be recovered by tokenising the completion.
        """
        token_logprobs = meta.get('output_token_logprobs')
        if token_logprobs:
            return [int(entry[1]) for entry in token_logprobs]
        text = output.get('text') or ''
        return tokenizer.encode(text, add_special_tokens=False) if text else []

    @classmethod
    def _sequence_logprobs(cls, meta: Dict[str, Any], tokens: List[int],
                           logprobs_k: int) -> Optional[List[List[Tuple[int, float]]]]:
        """Per-position logprobs for the generated tokens, as ``[(token_id, logprob), ...]`` per position.

        Two shapes, matching what was asked for: the sampled token alone when ``logprobs == 1``, and
        sglang's top-k list otherwise -- the same distinction :class:`VLLMEngine` makes.
        """
        if not logprobs_k:
            return None
        if logprobs_k > 1:
            top_logprobs = meta.get('output_top_logprobs')
            if top_logprobs:
                return [cls._to_token_logprobs(position)[:logprobs_k] for position in top_logprobs]
        token_logprobs = meta.get('output_token_logprobs')
        if not token_logprobs:
            return None
        return [[entry] for entry in cls._to_token_logprobs(token_logprobs)]

    @classmethod
    def _prompt_logprobs(cls, meta: Dict[str, Any], prompt_logprobs_k: int):
        """Prompt-token logprobs, in the two forms :class:`SampleResponse` carries them.

        The first prompt token has no logprob -- nothing preceded it -- and sglang reports that as
        ``None``, which is passed through rather than filled in.
        """
        if not prompt_logprobs_k:
            return None, None
        token_logprobs = meta.get('input_token_logprobs')
        if not token_logprobs:
            return None, None
        flat = [None if entry[0] is None else float(entry[0]) for entry in token_logprobs]

        topk = None
        top_logprobs = meta.get('input_top_logprobs')
        if top_logprobs:
            topk = [
                None if position is None else cls._to_token_logprobs(position)[:prompt_logprobs_k]
                for position in top_logprobs
            ]
        return flat, topk

    @staticmethod
    def _to_token_logprobs(entries: Sequence[Any]) -> List[Tuple[int, float]]:
        """sglang's ``(logprob, token_id, text)`` entries as twinkle's ``(token_id, logprob)`` pairs.

        The order is reversed between the two, so the token id is asserted to be an integer: a silent
        swap here would put logprobs where token ids are expected and corrupt every downstream ratio.
        """
        result = []
        for entry in entries:
            logprob, token_id = entry[0], entry[1]
            assert isinstance(token_id,
                              int), (f'Expected sglang logprob entries as (logprob, token_id, ...), got {entry!r}. '
                                     'The tuple order may have changed between sglang versions.')
            result.append((token_id, float(logprob)))
        return result

    # =========================================================================
    # Weight synchronisation
    # =========================================================================

    async def update_weights(
        self,
        weights,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        bucket_size_mb: int = 2048,
        **kwargs,
    ) -> None:
        """Load new weights into the running engine, in buckets.

        Accepts a ``dict[str, Tensor]`` or an (async) generator of ``(name, tensor)``. The generator is
        consumed a tensor at a time into a bucket and handed to sglang once the bucket is full, so peak
        extra memory is one bucket rather than a second copy of the model -- the same reason
        :class:`VLLMEngine` streams.

        The KV cache is flushed once, after the last bucket. Flushing per bucket would be correct but
        pays the cost N times for a model that is only consistent again at the end anyway.

        Args:
            weights: The new weights.
            peft_config: PEFT config, for a LoRA-only sync. Not supported here; see below.
            base_sync_done: Whether the base model has already been synced.
            bucket_size_mb: How much to accumulate before handing a bucket to sglang.

        Raises:
            NotImplementedError: For a LoRA-only sync. sglang's tensor path updates base weights only;
                its adapters are registered by ``load_lora_adapter``, which reads from a path, so an
                incremental LoRA sync would have to write the adapter to disk first. Use
                ``sync_weights(merge_and_sync=True)``, which sends merged base weights.
        """
        if base_sync_done and peft_config:
            raise NotImplementedError(
                'SGLangEngine cannot apply a LoRA-only weight sync: sglang loads adapters from a path '
                'via load_lora_adapter, not from tensors. Call sync_weights(merge_and_sync=True) to '
                'send merged base weights instead.')

        weight_aiter = self._as_async_iter(weights)
        bucket_limit = int(bucket_size_mb) << 20
        bucket: List[Tuple[str, torch.Tensor]] = []
        bucket_bytes = 0
        total = 0

        async for name, tensor in weight_aiter:
            bucket.append((name, tensor))
            bucket_bytes += tensor.numel() * tensor.element_size()
            total += 1
            if bucket_bytes >= bucket_limit:
                await self._flush_bucket(bucket, flush_cache=False)
                bucket, bucket_bytes = [], 0

        if bucket:
            await self._flush_bucket(bucket, flush_cache=True)
        elif total:
            # Everything went out in full buckets, so the cache still has to be dropped.
            await self._manager.flush_cache()

        if total == 0:
            logger.warning('update_weights called with no weights')
        else:
            logger.info(f'Updated {total} weights into the sglang engine')

    async def _flush_bucket(self, bucket: List[Tuple[str, torch.Tensor]], flush_cache: bool) -> None:
        """Hand one bucket to sglang, which serialises it out to the TP workers itself.

        Each TP worker deserialises its own copy of the payload, hence one serialisation per rank.
        """
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
        from sglang.srt.utils import MultiprocessingSerializer

        request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(bucket) for _ in range(self.tensor_parallel_size)
            ],
            load_format=None,
            flush_cache=flush_cache,
        )
        success, message = await self._manager.update_weights_from_tensor(request, None)
        if not success:
            raise RuntimeError(f'sglang rejected a weight update bucket of {len(bucket)} tensors: {message}')

    @staticmethod
    async def _as_async_iter(weights):
        """``weights`` as an async iterator, whichever of the three forms it arrived in."""
        if isinstance(weights, dict):
            for item in weights.items():
                yield item
        elif hasattr(weights, '__aiter__'):
            async for item in weights:
                yield item
        else:
            for item in weights:
                yield item

    # =========================================================================
    # Memory and lifecycle
    # =========================================================================

    async def sleep(self, level: int = 1, **kwargs) -> None:
        """Give back device memory while training runs.

        sglang releases by region rather than by level, so the vLLM-style ``level`` is read as how much
        to give up: level 1 drops the KV cache and keeps the weights resident, level 2 and above drops
        both.
        """
        if not self._memory_saver_enabled:
            logger.warning('sleep() does nothing unless the engine was created with '
                           'enable_memory_saver=True; skipping.')
            return
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

        tags = [GPU_MEMORY_TYPE_KV_CACHE]
        if level >= 2:
            tags.append(GPU_MEMORY_TYPE_WEIGHTS)
        await self._manager.release_memory_occupation(ReleaseMemoryOccupationReqInput(tags=tags), None)
        self._offloaded_tags = tags

    async def wake_up(self, tags: Optional[List[str]] = None, **kwargs) -> None:
        """Take the memory back.

        Defaults to resuming exactly what :meth:`sleep` gave up. sglang reads ``tags=None`` as every
        region and then removes each from its offload set unconditionally, so passing None after a
        level-1 sleep raises ``KeyError: 'weights'`` -- the weights were never offloaded.
        """
        if not self._memory_saver_enabled:
            logger.warning('wake_up() does nothing unless the engine was created with '
                           'enable_memory_saver=True; skipping.')
            return
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        tags = tags or self._offloaded_tags
        if not tags:
            logger.warning('wake_up() called without a preceding sleep(); skipping.')
            return
        await self._manager.resume_memory_occupation(ResumeMemoryOccupationReqInput(tags=tags), None)
        self._offloaded_tags = []

    async def reset_prefix_cache(self) -> None:
        await self._manager.flush_cache()

    async def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False) -> None:
        await self.engine.async_load_lora_adapter(lora_name=lora_name, lora_path=lora_path, pinned=pinned)

    async def unload_lora_adapter(self, lora_name: str) -> None:
        await self.engine.async_unload_lora_adapter(lora_name=lora_name)

    async def shutdown(self) -> None:
        try:
            # Unlike the rest, Engine.shutdown only tears down subprocesses and touches no event loop.
            self.engine.shutdown()
        except Exception as e:  # noqa: BLE001  -- shutdown runs on the way out; never raise from here
            logger.warning(f'sglang engine shutdown error: {e}')
