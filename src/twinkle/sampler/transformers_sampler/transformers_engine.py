# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import threading
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle import get_logger, requires
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.sampler.base_engine import BaseSamplerEngine
from twinkle.utils import torch_util

logger = get_logger()


def _resolve_model_cls(model_cls: Any):
    """The transformers class to ``from_pretrained`` with.

    Defaults to ``AutoModelForCausalLM`` (plain text generation). A vision-language checkpoint's
    config is not a causal-LM config, so ``AutoModelForCausalLM`` rejects it -- the caller supplies
    the family's real ``...ForConditionalGeneration`` class instead, either as the class object or as
    a string: ``'module:ClassName'`` (the form dev's model loaders declare, and safe to ship through
    Ray ``engine_args``) or a bare name resolved against ``transformers`` (twinkle's
    ``TransformersModel.model_cls`` convention).
    """
    from transformers import AutoModelForCausalLM
    if model_cls is None:
        return AutoModelForCausalLM
    if isinstance(model_cls, str):
        if ':' in model_cls:
            import importlib
            module_name, _, cls_name = model_cls.partition(':')
            return getattr(importlib.import_module(module_name), cls_name)
        import transformers
        return getattr(transformers, model_cls)
    return model_cls


class TransformersEngine(BaseSamplerEngine):
    """Wrap a HF causal LM (or a vision-language model, via ``model_cls``) so it satisfies the
    sampler-engine contract."""

    def __init__(
        self,
        model_id: str,
        *,
        dtype: Any = None,
        device_map: Any = None,
        max_model_len: Optional[int] = None,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
        model_cls: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        **model_kwargs,
    ):
        """
        Args:
            model_id: HuggingFace model id or local path.
            dtype: Torch dtype (or its string name) for the weights. Defaults to ``'auto'``.
            device_map: Passed to ``from_pretrained``. Defaults to the current accelerator, i.e. a
                single-device load -- ``'auto'`` would shard the model across every visible GPU,
                which is wrong under data parallelism where each rank owns one device.
            max_model_len: Prompt+completion budget. Prompts longer than this are rejected rather
                than silently truncated, because a truncated prompt produces a plausible-looking but
                meaningless completion.
            attn_implementation: e.g. ``'flash_attention_2'``, ``'sdpa'``.
            trust_remote_code: Allow custom modelling code from the checkpoint.
            model_cls: The transformers class to load with (class object, ``'module:ClassName'`` or a
                bare ``transformers`` name). Defaults to ``AutoModelForCausalLM``; a vision-language
                checkpoint needs its own ``...ForConditionalGeneration`` class (see
                :func:`_resolve_model_cls`).
            model: A pre-built model, for callers that already loaded one. Skips ``from_pretrained``.
            tokenizer: A pre-built tokenizer/processor, likewise.
            **model_kwargs: Forwarded to ``from_pretrained``.
        """
        requires('transformers')
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.max_model_len = max_model_len
        # Serializes generate() calls: a single nn.Module cannot run two generate loops at once, and
        # the lock is what turns concurrent callers into a queue instead of a corrupted forward.
        self._generate_lock = asyncio.Lock()
        self._adapters: Dict[str, str] = {}
        # Serializes adapter loading across caller threads: sample() dispatches one thread per
        # prompt, so without this lock N concurrent first-requests all miss the _adapters cache
        # and run PeftModel.from_pretrained on the SAME nn.Module simultaneously -- concurrent
        # parameter injection/.to() corrupts the allocator (glibc 'corrupted double-linked list')
        # or double-wraps the model. threading.Lock (not the asyncio generate lock): load_lora is
        # called from synchronous dispatch threads.
        self._lora_lock = threading.Lock()
        self._active_adapter: Optional[str] = None

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, 'pad_token_id', None) is None:
            # Padding is unavoidable for batched generation; eos is the conventional stand-in and is
            # masked out anyway, so it never contributes to the logits that matter.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if model is None:
            model_cls_resolved = _resolve_model_cls(model_cls)
            load_kwargs: Dict[str, Any] = {
                'dtype': dtype if dtype is not None else 'auto',
                'trust_remote_code': trust_remote_code,
                **model_kwargs,
            }
            if attn_implementation is not None:
                load_kwargs['attn_implementation'] = attn_implementation
            load_kwargs['device_map'] = device_map if device_map is not None else torch_util.get_device(None)
            logger.info(f'TransformersEngine loading {model_id} with {model_cls_resolved.__name__} and {load_kwargs}')
            model = model_cls_resolved.from_pretrained(model_id, **load_kwargs)
        self.model = model
        self.model.eval()
        # transformers' `_keep_in_fp32_modules_strict` contract (modeling_utils dtype plan): when a
        # model declares strict fp32 modules (e.g. DeepseekV4's norms/hyper-connections) and is
        # loaded in fp16/bf16, those weights stay fp32 and the model's forward is only dtype-safe
        # under AMP -- plain eager calls hit `expected mat1 and mat2 to have the same dtype`
        # (fp32 norm output into a bf16 linear). Training gets autocast from the trainer; inference
        # must supply it here. Models without the strict list keep the exact previous behavior.
        self._amp_dtype = None
        if getattr(model, '_keep_in_fp32_modules_strict', None):
            mdtype = getattr(getattr(model, 'config', None), 'dtype', None)
            if isinstance(mdtype, str):
                mdtype = getattr(torch, mdtype, None)
            if mdtype in (torch.float16, torch.bfloat16):
                self._amp_dtype = mdtype

    def _amp_context(self):
        """AMP context honoring the loaded model's strict-fp32 contract; no-op otherwise."""
        if self._amp_dtype is None:
            import contextlib
            return contextlib.nullcontext()
        device_type = next(self.model.parameters()).device.type
        return torch.autocast(device_type=device_type, dtype=self._amp_dtype)

    # ------------------------------------------------------------------ engine contract

    async def get_tokenizer(self):
        return self.tokenizer

    async def sample(
        self,
        prompt_token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
        *,
        num_samples: int = 1,
        logprobs: bool = True,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        adapter_uri: Optional[str] = None,
        request_id: Optional[str] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
        **kwargs,
    ) -> SampleResponse:
        """Single-prompt sampling. Prefer :meth:`batch_sample` -- see the class docstring on why."""
        responses = await self.batch_sample(
            [prompt_token_ids],
            sampling_params,
            num_samples=num_samples,
            logprobs=logprobs,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            adapter_uri=adapter_uri,
            **kwargs,
        )
        return responses[0]

    def generate_batch(
        self,
        prompt_token_ids: Sequence[List[int]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        num_samples: int = 1,
        logprobs: bool = True,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        adapter_uri: Optional[str] = None,
        extra_model_inputs: Optional[Dict[str, Any]] = None,
        use_base_model: bool = False,
        **kwargs,
    ) -> List[SampleResponse]:
        """Generate for several prompts in one padded forward, synchronously.

        The blocking core behind :meth:`batch_sample`. A caller with no event loop to keep responsive --
        a training model's worker generating inline (see ``TransformersModel.generate``) -- drives this
        directly rather than paying for a loop and a thread hop it has no use for.

        ``extra_model_inputs`` carries the already-processed multimodal tensors (``pixel_values`` and
        friends). Unlike vLLM, this engine cannot accept raw images -- the template's processor has to
        run first -- which is the contract note in ``BaseSamplerEngine.sample``.

        ``use_base_model`` generates from the base weights even when a peft adapter is loaded. It is
        explicit because a bare ``adapter_uri=None`` means "keep the current adapter" for in-place
        generation on a live training model (``TransformersModel.generate``), while a sampler request
        with no adapter must sample the base -- the two callers need opposite defaults.
        """
        if not prompt_token_ids:
            return []
        params = sampling_params or SamplingParams()
        self._check_lengths(prompt_token_ids, params)
        self._activate_adapter(adapter_uri)
        with self._base_model_context(use_base_model):
            return self._generate_batch(
                list(prompt_token_ids),
                params,
                num_samples,
                logprobs,
                include_prompt_logprobs,
                topk_prompt_logprobs,
                extra_model_inputs or {},
                kwargs,
            )

    async def batch_sample(
        self,
        prompt_token_ids: Sequence[List[int]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        num_samples: int = 1,
        logprobs: bool = True,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        adapter_uri: Optional[str] = None,
        extra_model_inputs: Optional[Dict[str, Any]] = None,
        use_base_model: bool = False,
        **kwargs,
    ) -> List[SampleResponse]:
        """Async :meth:`generate_batch`: serialized behind a lock and run off the caller's event loop.

        Not part of :class:`BaseSamplerEngine`, but the reason this engine is usable at all: routing
        every prompt through :meth:`sample` would serialize them behind :attr:`_generate_lock` and
        give up the only batching transformers has. The lock turns concurrent callers into a queue
        instead of a corrupted forward (a single nn.Module cannot run two generate loops at once), and
        off-loading the blocking generate keeps the caller's loop responsive (the deploy recipe streams
        other requests on it).
        """
        async with self._generate_lock:
            return await asyncio.to_thread(
                self.generate_batch,
                prompt_token_ids,
                sampling_params,
                num_samples=num_samples,
                logprobs=logprobs,
                include_prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
                adapter_uri=adapter_uri,
                extra_model_inputs=extra_model_inputs,
                use_base_model=use_base_model,
                **kwargs,
            )

    async def generate_stream(
        self,
        prompt: List[int],
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[str] = None,
        extra_model_inputs: Optional[Dict[str, Any]] = None,
        use_base_model: bool = False,
        **kwargs,
    ):
        """Yield ``(delta_text, finish_reason)`` as tokens are produced.

        ``TextIteratorStreamer`` only speaks to a blocking ``generate()`` running in another thread,
        so the bridge here is a thread + a thread-safe queue drained from the event loop. finish_reason
        is only known once generation ends, hence the final ``('', reason)`` tuple.
        """
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class _AbortOnEvent(StoppingCriteria):
            """Ends ``model.generate`` once ``aborted`` is set; checked between decoding steps."""

            def __init__(self, aborted: threading.Event):
                super().__init__()
                self.aborted = aborted

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return self.aborted.is_set()

        params = sampling_params or SamplingParams()
        self._check_lengths([prompt], params)

        async with self._generate_lock:
            self._activate_adapter(lora_request)
            with self._base_model_context(use_base_model):
                # Same seeded-request contract as _generate_batch: to_transformers spells the seed as
                # torch.manual_seed, which is process-global. Snapshot before building the gen kwargs
                # (that is where the reseed happens) and restore once model.generate has really ended,
                # so a seeded stream stays reproducible without clobbering the caller's RNG (training
                # dropout / data order share this process). The restore sits in the finally AFTER the
                # join, covering normal completion, errors and a consumer that closes the stream early.
                seeded = params.seed is not None
                if seeded:
                    cpu_rng_state = torch.get_rng_state()
                    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False if params.stop else None)
                gen_kwargs = self._build_gen_kwargs(params, num_samples=1, logprobs=False)
                gen_kwargs['streamer'] = streamer
                # Cooperative stop for an abandoned stream: checked between decoding steps. Appended, not
                # assigned: ``to_transformers`` already installs a ``StopStringCriteria`` when ``params.stop``
                # is a string list.
                aborted = threading.Event()
                criteria = gen_kwargs.get('stopping_criteria')
                if criteria is None:
                    criteria = gen_kwargs['stopping_criteria'] = StoppingCriteriaList()
                criteria.append(_AbortOnEvent(aborted))
                inputs = self._to_model_inputs([prompt], extra_model_inputs)
                produced: List[int] = []
                eos_ids = gen_kwargs.get('eos_token_id')
                eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]) - {None}

                def _run():
                    try:
                        # autocast is thread-local: enter it inside the streaming thread.
                        with self._amp_context():
                            out = self.model.generate(**inputs, **gen_kwargs)
                        sequences = out.sequences if hasattr(out, 'sequences') else out
                        produced.extend(_strip_padding(sequences[0, inputs['input_ids'].shape[-1]:], eos_ids))
                    except Exception as exc:  # surfaced to the consumer via the queue below
                        streamer.on_finalized_text('', stream_end=True)
                        raise exc

                thread = threading.Thread(target=_run, daemon=True, name='TransformersEngine-stream')
                thread.start()
                try:
                    loop = asyncio.get_running_loop()
                    sentinel = object()
                    stops = [params.stop] if isinstance(params.stop, str) else list(params.stop or ())
                    strings = [stop for stop in stops if isinstance(stop, str)]
                    # Hold potential stop prefixes so excluded text is never emitted and then retracted.
                    hold = max((len(stop) - 1 for stop in strings), default=0)
                    pending = ''
                    matched = False
                    while True:
                        delta = await loop.run_in_executor(None, lambda: next(streamer, sentinel))
                        if delta is sentinel:
                            break
                        if not delta or matched:
                            continue
                        pending, matched = self._trim_stop_text(pending + delta, params)
                        ready = len(pending) if matched else max(0, len(pending) - hold)
                        if ready:
                            yield pending[:ready], None
                            pending = pending[ready:]
                    if pending:
                        yield pending, None
                finally:
                    # Whatever ends the stream -- completion, a consumer that closed it early, an error --
                    # ``model.generate`` must have stopped before the lock is released, or the next
                    # generate / weight sync runs concurrently with this one on the same nn.Module.
                    # On completion the criteria is already satisfied and the join is immediate.
                    aborted.set()
                    thread.join()
                    if seeded:
                        torch.set_rng_state(cpu_rng_state)
                        if cuda_rng_states is not None:
                            torch.cuda.set_rng_state_all(cuda_rng_states)
                max_new = gen_kwargs.get('max_new_tokens')
                yield (
                    '',
                    'stop' if matched else (
                        'length'
                        if max_new and len(produced) >= max_new and (not produced or produced[-1] not in eos_ids)
                        else 'stop'
                    ),
                )

    # ------------------------------------------------------------------ weights / memory

    async def update_weights(self,
                             weights: Dict[str, torch.Tensor],
                             adapter_name: Optional[str] = None,
                             **kwargs) -> None:
        """In-place weight refresh for colocated training. Loads non-strictly: callers stream shards."""
        if adapter_name is not None:
            raise NotImplementedError('TransformersEngine cannot hot-swap adapter weights; pass adapter_uri '
                                      'to sample() with an on-disk adapter instead.')
        async with self._generate_lock:
            self.model.load_state_dict(weights, strict=False, assign=False)

    async def get_state_keys(self) -> List[str]:
        return list(self.model.state_dict().keys())

    async def sleep(self, **kwargs) -> None:
        """Park weights on CPU so a colocated trainer can have the GPU back."""
        async with self._generate_lock:
            self._device_before_sleep = next(self.model.parameters()).device
            self.model.to('cpu')
            torch_util.empty_cache()

    async def wake_up(self, **kwargs) -> None:
        async with self._generate_lock:
            self.model.to(getattr(self, '_device_before_sleep', torch_util.get_device(None)))

    async def shutdown(self) -> None:
        self.model = None
        torch_util.empty_cache()

    # ------------------------------------------------------------------ LoRA

    def load_lora(self, adapter_path: str, adapter_name: Optional[str] = None) -> str:
        """Load a peft adapter and return the name to pass as ``adapter_uri``.

        Cached by path: peft raises on a duplicate name, and reloading the same weights is pure waste.
        Note that peft can hold many adapters but activates one at a time, so mixing adapters within a
        single batch is impossible here -- callers must group requests by adapter.
        """
        with self._lora_lock:
            if adapter_path in self._adapters:
                return self._adapters[adapter_path]
            requires('peft')
            from peft import PeftModel

            name = adapter_name or f'adapter_{len(self._adapters)}'
            if isinstance(self.model, PeftModel):
                self.model.load_adapter(adapter_path, adapter_name=name)
            else:
                self.model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name=name)
            self.model.eval()
            self._adapters[adapter_path] = name
            logger.info(f'TransformersEngine loaded LoRA {adapter_path} as {name!r}')
            return name

    def _activate_adapter(self, adapter_uri: Optional[str]) -> None:
        """Switch the active peft adapter, or disable adapters entirely when None."""
        from peft import PeftModel

        if not isinstance(self.model, PeftModel):
            if adapter_uri:
                raise ValueError(f'adapter {adapter_uri!r} requested but no LoRA is loaded; call load_lora() first.')
            return
        if adapter_uri is None:
            # disable_adapter() is a context manager, so base-model sampling is expressed by the
            # caller passing use_base_model, handled in the sampler. Here None means "keep whatever
            # single adapter is loaded" only if exactly one exists.
            return
        if adapter_uri != self._active_adapter:
            self.model.set_adapter(adapter_uri)
            self._active_adapter = adapter_uri

    def _base_model_context(self, use_base_model: bool):
        """Disable any loaded peft adapter for the duration of a base-model request.

        peft activates one adapter at a time and ``_activate_adapter(None)`` deliberately keeps the
        current one (in-place generation on a live training model must keep its adapter), so a
        sampler request that explicitly asks for the base weights needs this separate switch.
        """
        import contextlib
        from peft import PeftModel
        if use_base_model and isinstance(self.model, PeftModel):
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ internals

    def _check_lengths(self, prompts: Sequence[List[int]], params: SamplingParams) -> None:
        if self.max_model_len is None:
            return
        budget = self.max_model_len - (params.max_tokens or 0)
        longest = max(len(p) for p in prompts)
        if longest > budget:
            raise ValueError(f'Prompt of {longest} tokens leaves no room for {params.max_tokens} new tokens '
                             f'within max_model_len={self.max_model_len}. Shorten the prompt, lower '
                             'max_tokens, or raise max_model_len -- this engine will not truncate, because a '
                             'truncated prompt yields a fluent but meaningless completion.')

    def _build_gen_kwargs(self, params: SamplingParams, *, num_samples: int, logprobs: bool) -> Dict[str, Any]:
        gen_kwargs = params.to_transformers(self.tokenizer)
        gen_kwargs['num_return_sequences'] = num_samples
        if logprobs:
            gen_kwargs['output_scores'] = True
            # Request dict output on a GenerationConfig copy, not as a call kwarg: generate()
            # wrappers that hardcode return_dict_in_generate when delegating to a submodel
            # (qwen-asr's Qwen3ASRForConditionalGeneration.generate -> thinker.generate) forward
            # the caller's same-named kwarg too and raise TypeError 'multiple values for keyword
            # argument'. GenerationMixin merges config values with call kwargs (kwargs win), so
            # standard models keep identical behavior; the copy never mutates the model's config.
            import copy
            generation_config = copy.copy(self.model.generation_config)
            generation_config.return_dict_in_generate = True
            gen_kwargs['generation_config'] = generation_config
        return gen_kwargs

    def _to_model_inputs(self, prompts: Sequence[List[int]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Left-pad the prompts into one batch on the model's device."""
        device = next(self.model.parameters()).device
        width = max(len(p) for p in prompts)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            input_ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            attention_mask[row, width - len(prompt):] = 1
        inputs = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)}
        for key, value in (extra or {}).items():
            inputs[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return inputs

    def _generate_batch(
        self,
        prompts: List[List[int]],
        params: SamplingParams,
        num_samples: int,
        want_logprobs: bool,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
        extra_model_inputs: Dict[str, Any],
        passthrough: Dict[str, Any],
    ) -> List[SampleResponse]:
        inputs = self._to_model_inputs(prompts, extra_model_inputs)
        # A seeded request seeds torch for the generate call only: to_transformers spells
        # the seed as torch.manual_seed, which is process-global. This engine also runs
        # inside training workers (TransformersModel.generate), where leaving the global
        # RNG reseeded would silently change every later consumer (dropout, data order).
        # Snapshot and restore around the seeded section so the request stays reproducible
        # without clobbering anyone else's randomness.
        seeded = params.seed is not None
        if seeded:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            gen_kwargs = self._build_gen_kwargs(params, num_samples=num_samples, logprobs=want_logprobs)
            gen_kwargs.update(passthrough)
            prompt_width = inputs['input_ids'].shape[-1]

            with torch.inference_mode(), self._amp_context():
                output = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if seeded:
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
        sequences = output.sequences if hasattr(output, 'sequences') else output
        # generate() returns prompt+completion; only the tail is new. With num_return_sequences=k the
        # batch dim is len(prompts)*k, ordered prompt-major.
        completions = sequences[:, prompt_width:]
        step_logprobs = _step_logprobs(output, params) if want_logprobs and hasattr(output, 'scores') else None

        prompt_logprobs, topk_logprobs = None, None
        if include_prompt_logprobs:
            prompt_logprobs, topk_logprobs = self._prompt_logprobs(inputs, topk_prompt_logprobs)

        eos_ids = gen_kwargs.get('eos_token_id')
        eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]) - {None}
        max_new = gen_kwargs.get('max_new_tokens')
        responses: List[SampleResponse] = []
        for index, prompt in enumerate(prompts):
            seqs = []
            for offset in range(num_samples):
                row = index * num_samples + offset
                tokens = _strip_padding(completions[row], eos_ids)
                decoded = self.tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False if params.stop else None)
                decoded, matched = self._trim_stop_text(decoded, params)
                stop_reason = (
                    'length'
                    if max_new and len(tokens) >= max_new and not matched
                    and (not tokens or tokens[-1] not in eos_ids)
                    else 'stop'
                )
                if matched and not params.include_stop_str_in_output:
                    tokens = self._trim_stop_tokens(tokens, decoded)
                seqs.append(
                    SampledSequence(
                        stop_reason=stop_reason,
                        tokens=tokens,
                        logprobs=None if step_logprobs is None else step_logprobs[row][:len(tokens)],
                        decoded=decoded,
                    ))
            responses.append(
                SampleResponse(
                    sequences=seqs,
                    prompt_token_ids=list(prompt),
                    prompt_logprobs=None if prompt_logprobs is None else prompt_logprobs[index],
                    topk_prompt_logprobs=None if topk_logprobs is None else topk_logprobs[index],
                )
            )
        return responses

    @staticmethod
    def _trim_stop_text(text: str, params: SamplingParams) -> Tuple[str, bool]:
        """Trim text, not sampled tokens: a stop can start or end inside a token."""
        stops = [params.stop] if isinstance(params.stop, str) else list(params.stop or ())
        if not stops or isinstance(stops[0], int):
            return text, False
        matches = [(text.find(stop), stop) for stop in stops if stop and text.find(stop) >= 0]
        if not matches:
            return text, False
        start, stop = min(matches, key=lambda match: match[0])
        end = start + len(stop) if params.include_stop_str_in_output else start
        return text[:end], True

    def _trim_stop_tokens(self, tokens: List[int], text: str) -> List[int]:
        """The token prefix that stays inside a trimmed stop ``text`` (the same rule as the vLLM engine:
        the sampled ``tokens`` never carry the stop string when it is excluded from the output).
        ``generate()`` runs on to the EOS after a stop string appears, so the tail can be long; the
        decoded prefix grows monotonically with the token count, hence the binary search."""
        def decoded(count: int) -> str:
            return self.tokenizer.decode(tokens[:count], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        low, high = 0, len(tokens)
        while low < high:
            mid = (low + high + 1) // 2
            if text.startswith(decoded(mid)):
                low = mid
            else:
                high = mid - 1
        return tokens[:low]

    def _prompt_logprobs(self, inputs: Dict[str, Any], topk: int):
        """Score the prompt itself -- ``generate()`` never does, so this is a second forward pass.

        Each row is scored on its OWN real tokens, with no padding and no batch neighbours. A
        batched bf16 forward is not numerically identical to a single-row forward on this stack:
        the attention / chunked-recurrence kernels pick their reduction order from the batch
        shape, and hybrid linear-attention layers amplify the drift (measured up to ~0.6 logits /
        ~0.08 nats and even top-k id flips on Qwen3.5-0.8B, runs/H07 g41). Scoring a padded batch
        would therefore make a prompt's score depend on which OTHER prompts share the batch --
        unacceptable for RL prompt scoring, where the score must be a function of the prompt alone
        (the vLLM backend's per-sequence contract). Per-row forwards restore that independence at
        the cost of one forward per prompt; scoring is opt-in (``prompt_logprobs``), and callers
        that need batched throughput at scale use the vLLM backend.
        """
        input_ids = inputs['input_ids']
        attention = inputs['attention_mask']
        out, alternatives = [], [] if topk else None
        for row in range(input_ids.shape[0]):
            keep = attention[row].bool()
            row_inputs = self._score_row_inputs(inputs, row, keep)
            # Score without building a KV cache this pass never reads (tw 3cb3c4dc contract):
            # the forward defaults to caching, which for a long prompt allocates a full cache.
            row_inputs['use_cache'] = False
            with torch.inference_mode(), self._amp_context():
                logits = self.model(**row_inputs).logits
            logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            targets = row_inputs['input_ids'][0, 1:]
            picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            # The first real token has no predecessor to be predicted from, hence the leading None,
            # matching vLLM's prompt_logprobs convention.
            out.append([None] + picked.tolist())
            if topk:
                values, indices = logprobs.topk(topk, dim=-1)
                alternatives.append(
                    [None] + [list(zip(ids, scores, strict=False))
                              for ids, scores in zip(indices.tolist(), values.tolist(), strict=False)])
        return out, alternatives

    def _score_row_inputs(self, inputs: Dict[str, Any], row: int, keep: torch.Tensor) -> Dict[str, Any]:
        """One row's real (unpadded) model inputs, sliced out of the collated batch.

        Per-token companions (``mm_token_type_ids``, M-RoPE ``position_ids``, ...) keep their row
        and lose the padding. Vision payloads (``pixel_values`` + ``image_grid_thw``, and the video
        pair) are concatenated across rows by the template's collator, so they are split by walking
        the grids in row order and matching each row's placeholder-token count; the merge factor is
        derived from the data (total patches / total placeholders) rather than assumed. Anything
        that cannot be attributed to a single row fails loudly instead of being scored wrongly.
        """
        batch, width = inputs['input_ids'].shape
        row_inputs = {
            'input_ids': inputs['input_ids'][row][keep].unsqueeze(0),
            'attention_mask': inputs['attention_mask'][row][keep].unsqueeze(0),
        }
        media = {
            'pixel_values': ('image_grid_thw', 'image_token_id'),
            'pixel_values_videos': ('video_grid_thw', 'video_token_id'),
        }
        handled = set()
        for payload_key, (grid_key, token_attr) in media.items():
            grids = inputs.get(grid_key)
            if grids is None:
                continue
            handled.update((payload_key, grid_key))
            payload = inputs.get(payload_key)
            token_id = self._model_config_value(token_attr)
            if payload is None or token_id is None:
                raise ValueError(f'{grid_key} present but {payload_key}/{token_attr} missing; '
                                 'cannot score this batch per row')
            patches = [int(g.prod()) for g in grids]
            holders = [int((ids[inputs['attention_mask'][r].bool()] == token_id).sum())
                       for r, ids in enumerate(inputs['input_ids'])]
            total_holders = sum(holders)
            if total_holders == 0:
                raise ValueError(f'{grid_key} present but no {token_attr} placeholders in any prompt')
            merge2 = sum(patches) // total_holders
            if merge2 * total_holders != sum(patches):
                raise ValueError(f'{grid_key} patches do not divide evenly into placeholder counts')
            counts = [p // merge2 for p in patches]
            # Walk images/videos in collation (row) order, assigning each row its contiguous share.
            spans, cursor = [], 0
            for holder in holders:
                start, taken = cursor, 0
                while taken < holder:
                    if cursor >= len(counts):
                        raise ValueError(f'{grid_key} runs out of entries before placeholders are matched')
                    taken += counts[cursor]
                    cursor += 1
                if taken != holder:
                    raise ValueError(f'{grid_key} entries do not align with per-row placeholder counts')
                spans.append((start, cursor))
            start, end = spans[row]
            if payload_key == 'pixel_values_videos':
                # ``second_per_grid_ts`` holds one entry per video (seconds per temporal grid),
                # collated in row order exactly like the grids. The template collator hands it over
                # as a list[float], which the generic tensor loop below would silently drop -- and
                # without it the model's internal M-RoPE falls back to the default 1.0 spacing,
                # scoring fps != 1 videos wrongly whenever position_ids are not supplied. Slice it
                # with the same video spans so the complete model-ready inputs survive.
                spg = inputs.get('second_per_grid_ts')
                if spg is not None:
                    handled.add('second_per_grid_ts')
                    if len(spg) != len(grids):
                        raise ValueError('second_per_grid_ts must hold one entry per video grid')
                    if end > start:
                        row_inputs['second_per_grid_ts'] = spg[start:end]
            if start == end:
                continue
            patch_start = sum(patches[:start])
            patch_end = sum(patches[:end])
            row_inputs[payload_key] = payload[patch_start:patch_end]
            row_inputs[grid_key] = grids[start:end]
        audio = inputs.get('input_features')
        if audio is not None:
            # Audio payloads (input_features + input_features_mask / feature_attention_mask) are
            # concatenated across rows by the template collator with ONE entry per audio, like the
            # vision grids -- but they carry no grid tensor, so attribute them by the audio
            # placeholder tokens: every audio expands to the same number of soft tokens (the
            # processor's fixed audio sequence), so total placeholders / audios gives the factor
            # and each row's share follows from its own placeholder count. Without this the
            # generic tensor loop below either dies on a mixed audio+text batch (leading dim !=
            # batch) or silently misattributes when the audio count happens to equal the batch.
            handled.add('input_features')
            mask_keys = [k for k in ('input_features_mask', 'feature_attention_mask') if inputs.get(k) is not None]
            # Mark every audio companion handled up front: a text-only row in a mixed batch carries
            # no audio slice, but the batched mask tensor must still not fall through to the generic
            # per-row loop (its leading dim is the audio count, not the batch size).
            handled.update(mask_keys)
            token_id = self._model_config_value('audio_token_id')
            if token_id is None:
                raise ValueError('input_features present but audio_token_id missing; '
                                 'cannot score this batch per row')
            holders = [int((ids[inputs['attention_mask'][r].bool()] == token_id).sum())
                       for r, ids in enumerate(inputs['input_ids'])]
            total_holders = sum(holders)
            n_audios = audio.shape[0]
            if total_holders == 0:
                raise ValueError('input_features present but no audio placeholders in any prompt')
            per_audio = total_holders // n_audios
            if per_audio * n_audios != total_holders:
                raise ValueError('audio placeholder counts do not divide evenly across audio features')
            spans, cursor = [], 0
            for holder in holders:
                if holder % per_audio:
                    raise ValueError('audio placeholders do not align with per-row audio counts')
                start = cursor
                cursor += holder // per_audio
                spans.append((start, cursor))
            start, end = spans[row]
            if end > start:
                feats = audio[start:end]
                for mask_key in mask_keys:
                    row_mask = inputs[mask_key][start:end]
                    if not bool(row_mask.all()):
                        # The collator pads unequal-length audios to the batch longest with the
                        # mask marking valid frames; restore this row's own (unpadded-to-batch)
                        # processor layout by dropping the trailing padding common to the row.
                        keep_len = int(row_mask.sum(dim=-1).max())
                        if feats.shape[-2] == row_mask.shape[-1]:
                            feats = feats[..., :keep_len, :]  # (audios, time, mel), e.g. gemma3n
                        else:
                            feats = feats[..., :keep_len]  # (audios, mel, time), e.g. qwen2-audio
                        row_mask = row_mask[..., :keep_len]
                    row_inputs[mask_key] = row_mask
                row_inputs['input_features'] = feats
        for key, value in inputs.items():
            if key in ('input_ids', 'attention_mask') or key in handled or not isinstance(value, torch.Tensor):
                continue
            if (key == 'position_ids' and value.dim() == 3 and value.shape[1] == batch
                    and value.shape[-1] == width):
                # M-RoPE layout [sections, batch, seq] (the template collator stacks on dim 1)
                row_inputs[key] = value[:, row, :][:, keep].unsqueeze(1)
            elif value.dim() >= 2 and value.shape[0] == batch and value.shape[-1] == width:
                row_inputs[key] = value[row, ..., keep].unsqueeze(0)
            elif value.dim() >= 1 and value.shape[0] == batch:
                row_inputs[key] = value[row].unsqueeze(0)
            else:
                raise ValueError(f'cannot attribute batched input {key!r} (shape {tuple(value.shape)}) '
                                 'to a single prompt for per-row scoring')
        return row_inputs

    def _model_config_value(self, attr: str):
        """Read a config field from the underlying HF model, looking through a peft wrapper."""
        model = self.model
        for _ in range(4):
            config = getattr(model, 'config', None)
            if config is not None and hasattr(config, attr):
                return getattr(config, attr)
            inner = getattr(model, 'base_model', None) or getattr(model, 'model', None)
            if inner is None or inner is model:
                break
            model = inner
        return None


def _step_logprobs(output: Any, params: SamplingParams) -> List[List[List[Tuple[int, float]]]]:
    """``generate()`` scores -> per-row, per-step ``[(token_id, logprob), ...]``.

    ``params.logprobs`` follows vLLM's meaning: 0 (or None) asks only for the token actually sampled,
    k > 0 additionally asks for the k most likely alternatives at that step.
    """
    topk = params.logprobs or 0
    rows = output.sequences.shape[0]
    prompt_width = output.sequences.shape[-1] - len(output.scores)
    result: List[List[List[Tuple[int, float]]]] = [[] for _ in range(rows)]
    for step, scores in enumerate(output.scores):
        logprobs = torch.log_softmax(scores.float(), dim=-1)
        chosen = output.sequences[:, prompt_width + step]
        for row in range(rows):
            token = int(chosen[row])
            entries = [(token, float(logprobs[row, token]))]
            if topk > 0:
                values, indices = torch.topk(logprobs[row], topk)
                entries.extend((int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist()) if int(i) != token)
            result[row].append(entries)
    return result


def _strip_padding(tokens: torch.Tensor, eos_ids: set[int]) -> List[int]:
    """Keep the sampled EOS; discard only the padding produced after termination."""
    ids = tokens.tolist()
    for index, token in enumerate(ids):
        if token in eos_ids:
            return ids[: index + 1]
    return ids
