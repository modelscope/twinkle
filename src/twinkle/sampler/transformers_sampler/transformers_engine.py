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
        **kwargs,
    ) -> List[SampleResponse]:
        """Generate for several prompts in one padded forward, synchronously.

        The blocking core behind :meth:`batch_sample`. A caller with no event loop to keep responsive --
        a training model's worker generating inline (see ``TransformersModel.generate``) -- drives this
        directly rather than paying for a loop and a thread hop it has no use for.

        ``extra_model_inputs`` carries the already-processed multimodal tensors (``pixel_values`` and
        friends). Unlike vLLM, this engine cannot accept raw images -- the template's processor has to
        run first -- which is the contract note in ``BaseSamplerEngine.sample``.
        """
        if not prompt_token_ids:
            return []
        params = sampling_params or SamplingParams()
        self._check_lengths(prompt_token_ids, params)
        self._activate_adapter(adapter_uri)
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
                **kwargs,
            )

    async def generate_stream(
        self,
        prompt: List[int],
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[str] = None,
        extra_model_inputs: Optional[Dict[str, Any]] = None,
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
            gen_kwargs['return_dict_in_generate'] = True
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

            with torch.inference_mode():
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
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = inputs['input_ids'][:, 1:]
        picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        out, alternatives = [], [] if topk else None
        for row in range(picked.shape[0]):
            attention = inputs['attention_mask'][row].bool()
            mask = attention[1:] & attention[:-1]
            first = [None] if attention.any() else []
            out.append(first + picked[row][mask].tolist())
            if topk:
                values, indices = logprobs[row][mask].topk(topk, dim=-1)
                alternatives.append(
                    first
                    + [
                        list(zip(ids, scores, strict=False))
                        for ids, scores in zip(indices.tolist(), values.tolist(), strict=False)
                    ]
                )
        return out, alternatives


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
