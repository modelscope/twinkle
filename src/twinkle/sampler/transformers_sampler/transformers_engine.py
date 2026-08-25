# Copyright (c) ModelScope Contributors. All rights reserved.
"""A :class:`~twinkle.sampler.BaseSamplerEngine` backed by plain transformers ``generate()``.

Why this exists alongside vLLM and sglang: those two are the throughput backends, but they cannot
load every architecture (custom / brand-new / ``trust_remote_code`` models often land in transformers
first), they cost a separate weight copy in the KV-cache format, and they are heavyweight optional
dependencies. This engine trades throughput for reach -- it runs anything ``AutoModelForCausalLM``
loads, with no extra install.

Three things differ from the vLLM engine and callers do feel them:

- **no continuous batching.** transformers generates one padded batch at a time, so a batch runs at
  the pace of its longest sequence and concurrent ``sample()`` calls serialize behind
  :attr:`_generate_lock`. Batching is therefore done up front (see :meth:`batch_sample`) rather than
  by the engine scheduler.
- **left padding is mandatory.** Decoder-only generation appends to the end of the sequence, so pads
  must sit on the left or the model continues from padding. The tokenizer's own ``padding_side`` is
  overridden per call rather than mutated globally, because the same tokenizer object may be shared
  with a training path that needs right padding.
- **logprobs come from ``scores``, not from a KV-cache side channel.** ``output_scores=True`` returns
  raw logits per step; they are log-softmaxed here. Prompt logprobs need a separate forward pass
  (:meth:`_prompt_logprobs`) because ``generate()`` never scores the prompt.
"""
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


class TransformersEngine(BaseSamplerEngine):
    """Wrap a HF causal LM so it satisfies the sampler-engine contract."""

    def __init__(
        self,
        model_id: str,
        *,
        dtype: Any = None,
        device_map: Any = None,
        max_model_len: Optional[int] = None,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
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
            model: A pre-built model, for callers that already loaded one. Skips ``from_pretrained``.
            tokenizer: A pre-built tokenizer/processor, likewise.
            **model_kwargs: Forwarded to ``from_pretrained``.
        """
        requires('transformers')
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
            load_kwargs: Dict[str, Any] = {
                'dtype': dtype if dtype is not None else 'auto',
                'trust_remote_code': trust_remote_code,
                **model_kwargs,
            }
            if attn_implementation is not None:
                load_kwargs['attn_implementation'] = attn_implementation
            load_kwargs['device_map'] = device_map if device_map is not None else torch_util.get_device(None)
            logger.info(f'TransformersEngine loading {model_id} with {load_kwargs}')
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
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
        """Generate for several prompts in one padded forward.

        Not part of :class:`BaseSamplerEngine`, but the reason this engine is usable at all: routing
        every prompt through :meth:`sample` would serialize them behind :attr:`_generate_lock` and
        give up the only batching transformers has.

        ``extra_model_inputs`` carries the already-processed multimodal tensors (``pixel_values`` and
        friends). Unlike vLLM, this engine cannot accept raw images -- the template's processor has to
        run first -- which is the contract note in ``BaseSamplerEngine.sample``.
        """
        if not prompt_token_ids:
            return []
        params = sampling_params or SamplingParams()
        self._check_lengths(prompt_token_ids, params)

        async with self._generate_lock:
            self._activate_adapter(adapter_uri)
            # generate() is blocking and GPU-bound; off-loading it keeps the caller's event loop
            # responsive (the deploy recipe streams other requests on it).
            return await asyncio.to_thread(
                self._generate_batch,
                list(prompt_token_ids),
                params,
                num_samples,
                logprobs,
                include_prompt_logprobs,
                topk_prompt_logprobs,
                extra_model_inputs or {},
                kwargs,
            )

    async def generate_stream(
        self,
        prompt: List[int],
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[str] = None,
        **kwargs,
    ):
        """Yield ``(delta_text, finish_reason)`` as tokens are produced.

        ``TextIteratorStreamer`` only speaks to a blocking ``generate()`` running in another thread,
        so the bridge here is a thread + a thread-safe queue drained from the event loop. finish_reason
        is only known once generation ends, hence the final ``('', reason)`` tuple.
        """
        from transformers import TextIteratorStreamer

        params = sampling_params or SamplingParams()
        self._check_lengths([prompt], params)

        async with self._generate_lock:
            self._activate_adapter(lora_request)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = self._build_gen_kwargs(params, num_samples=1, logprobs=False)
            gen_kwargs['streamer'] = streamer
            inputs = self._to_model_inputs([prompt])
            produced: List[int] = []

            def _run():
                try:
                    out = self.model.generate(**inputs, **gen_kwargs)
                    produced.append(out.shape[-1] - inputs['input_ids'].shape[-1])
                except Exception as exc:  # surfaced to the consumer via the queue below
                    streamer.on_finalized_text('', stream_end=True)
                    raise exc

            thread = threading.Thread(target=_run, daemon=True, name='TransformersEngine-stream')
            thread.start()

            loop = asyncio.get_running_loop()
            sentinel = object()
            while True:
                delta = await loop.run_in_executor(None, lambda: next(streamer, sentinel))
                if delta is sentinel:
                    break
                if delta:
                    yield delta, None
            thread.join()
            new_tokens = produced[0] if produced else 0
            yield '', ('length' if params.max_tokens and new_tokens >= params.max_tokens else 'stop')

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
        gen_kwargs = self._build_gen_kwargs(params, num_samples=num_samples, logprobs=want_logprobs)
        gen_kwargs.update(passthrough)
        prompt_width = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        sequences = output.sequences if hasattr(output, 'sequences') else output
        # generate() returns prompt+completion; only the tail is new. With num_return_sequences=k the
        # batch dim is len(prompts)*k, ordered prompt-major.
        completions = sequences[:, prompt_width:]
        step_logprobs = _step_logprobs(output, params) if want_logprobs and hasattr(output, 'scores') else None

        prompt_logprobs = None
        if include_prompt_logprobs:
            prompt_logprobs = self._prompt_logprobs(inputs, topk_prompt_logprobs)

        max_new = gen_kwargs.get('max_new_tokens')
        responses: List[SampleResponse] = []
        for index, prompt in enumerate(prompts):
            seqs = []
            for offset in range(num_samples):
                row = index * num_samples + offset
                tokens = _strip_padding(completions[row], self.tokenizer.pad_token_id)
                seqs.append(
                    SampledSequence(
                        stop_reason='length' if max_new and len(tokens) >= max_new else 'stop',
                        tokens=tokens,
                        logprobs=None if step_logprobs is None else step_logprobs[row][:len(tokens)],
                        decoded=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    ))
            responses.append(
                SampleResponse(
                    sequences=seqs,
                    prompt_token_ids=list(prompt),
                    prompt_logprobs=None if prompt_logprobs is None else prompt_logprobs[index],
                ))
        return responses

    def _prompt_logprobs(self, inputs: Dict[str, Any], topk: int) -> List[List[Optional[float]]]:
        """Score the prompt itself -- ``generate()`` never does, so this is a second forward pass."""
        with torch.inference_mode():
            logits = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = inputs['input_ids'][:, 1:]
        picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        out: List[List[Optional[float]]] = []
        for row in range(picked.shape[0]):
            mask = inputs['attention_mask'][row, 1:].bool()
            # The first real token has no predecessor to be predicted from, hence the leading None,
            # matching vLLM's prompt_logprobs convention.
            out.append([None] + picked[row][mask].tolist())
        return out


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


def _strip_padding(tokens: torch.Tensor, pad_id: Optional[int]) -> List[int]:
    """Drop the trailing pads a short sequence in a batch accumulates once it hits eos."""
    ids = tokens.tolist()
    if pad_id is None:
        return ids
    while ids and ids[-1] == pad_id:
        ids.pop()
    return ids
