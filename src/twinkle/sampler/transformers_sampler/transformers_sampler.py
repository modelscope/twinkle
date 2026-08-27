# Copyright (c) ModelScope Contributors. All rights reserved.
"""A :class:`~twinkle.sampler.Sampler` on plain transformers ``generate()``.

Drop-in for :class:`~twinkle.sampler.vLLMSampler`: same constructor shape
(``model_id, engine_args, device_mesh``), same ``sample`` / ``sample_stream`` / ``shutdown`` surface,
same synchronous-blocking convention. Swapping backends should not require touching call sites.

Two capabilities exist here that the vLLM path does not have:

- ``strict=False`` degrades **per input** instead of failing the batch. Offline sampling over a large
  corpus routinely hits a handful of rows that cannot be encoded (over-length, broken media refs); the
  legacy sampler set ``engine.strict = False`` for exactly this, and losing it would mean one bad row
  discards a whole run's worth of work.
- batching is explicit. transformers has no continuous batching, so inputs are grouped and padded up
  front by the engine's ``batch_sample``.

Not provided, deliberately: colocated weight sync. ``vLLMSampler`` mixes in ``CheckpointEngineMixin``
to receive NCCL-broadcast weights mid-training; this backend is for inference, evaluation and offline
sampling, and a half-working ``receive_weights`` would be worse than an absent one. ``update_weights``
on the engine covers the simple in-process refresh.

And not needed, in the one case where it would otherwise be: passing ``model=<TransformersModel>``
instead of ``model_id`` makes this a facade over a model that already exists, generating inside its
workers. Nothing is loaded and nothing is synced, because there is only ever one set of weights.
"""
from __future__ import annotations

import asyncio
import atexit
import threading
from copy import copy
from typing import Any, Dict, List, Optional, Type, Union

from twinkle import DeviceMesh, get_logger, remote_class, remote_function, requires
from twinkle.data_format import InputFeature, SampledSequence, SampleResponse, SamplingParams, Trajectory
from twinkle.hub import HubOperation
from twinkle.patch import Patch, apply_patch
from twinkle.sampler.base import Sampler

logger = get_logger()


@remote_class()
class TransformersSampler(Sampler):
    """Sample with ``AutoModelForCausalLM.generate()``."""

    def __init__(self,
                 model_id: str = None,
                 engine_args: Dict[str, Any] = None,
                 device_mesh: DeviceMesh = None,
                 *,
                 model: Any = None,
                 **kwargs):
        """
        Args:
            model_id: HuggingFace model id or local path. Mutually exclusive with ``model``.
            engine_args: Forwarded to :class:`~twinkle.sampler.TransformersEngine` -- ``dtype``,
                ``device_map``, ``max_model_len``, ``attn_implementation``, ``trust_remote_code``, plus
                anything ``from_pretrained`` accepts. Also accepts ``max_batch_size`` (see below).
                Note that its ``model`` entry is a different thing from the ``model`` argument here: it
                is a bare ``nn.Module`` for the engine to generate with in this process, whereas ``model``
                below is a ``TransformersModel`` whose workers do the generating. Both avoid a reload;
                the first is what ``TransformersModel.generate`` uses internally.
            device_mesh: Data-parallel layout. Slicing is done by ``dispatch='slice_dp'`` on
                :meth:`sample`, so each rank only ever sees its own shard.
            model: A ``TransformersModel`` to generate from instead of loading one. Sampling is then
                done by its workers, on the weights they already hold -- no second copy of the model,
                and no weight sync, which is what makes this worth having during training. Construct
                this facade on the driver without a ``remote_group``: it holds no weights itself, so it
                needs no workers of its own, and the model's ``generate`` does the data-rank slicing.
                ``engine_args`` and ``device_mesh`` belong to the model in this mode and are ignored.
            **kwargs: Accepted and ignored, for signature parity with the other samplers.
        """
        super().__init__()
        requires('transformers')

        if (model_id is None) == (model is None):
            raise ValueError('Pass either model_id, to load a model to sample from, or model, to sample '
                             'from one that already exists -- not both and not neither.')
        self.model = model
        self.model_id = model_id
        self.device_mesh = device_mesh
        engine_kwargs = dict(engine_args or {})
        # How many prompts go into one padded generate(). Larger is faster until the padding waste
        # from mixed prompt lengths outweighs it, and until activations stop fitting.
        self.max_batch_size = engine_kwargs.pop('max_batch_size', 8)

        # Mirror the other samplers: a private loop on a background thread, so the public methods stay
        # synchronous and blocking even when the caller already runs inside an event loop (Ray workers
        # run uvloop, which forbids asyncio.run).
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name='TransformersSampler-EventLoop')
        self._async_thread.start()

        if model is not None:
            # Nothing to build: generation happens in the model's workers.
            self.engine = None
        else:
            from .transformers_engine import TransformersEngine
            self.engine: TransformersEngine = TransformersEngine(model_id, **engine_kwargs)

        self._shutdown_called = False
        atexit.register(self.shutdown)

    def _run_event_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _run_in_loop(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._async_loop).result()

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

    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        apply_patch(self, patch_cls, **kwargs)

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def set_template(self, template_cls, **kwargs):
        """Set the template that turns messages into token ids.

        Handed to the model in facade mode, because that is where encoding happens. It would not work
        here in any case: the base implementation defaults ``model_id`` from this object, and a facade
        has none.
        """
        if self.model is not None:
            return self.model.set_template(template_cls, **kwargs)
        return super().set_template(template_cls, **kwargs)

    # ------------------------------------------------------------------ sampling

    @remote_function(dispatch='slice_dp', collect='flatten', lazy_collect=False)
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        return_encoded: bool = False,
        use_base_model: bool = False,
        adapter_paths: Optional[List[Optional[str]]] = None,
        strict: bool = True,
    ) -> List[SampleResponse]:
        """Sample completions for ``inputs``.

        Args:
            inputs: ``InputFeature``(s) carrying ``input_ids``, or ``Trajectory``(s) carrying
                ``messages`` (which then requires ``set_template`` to have been called).
            sampling_params: ``max_tokens == 0`` means "score, do not generate", matching the vLLM
                backend: one token is generated and discarded so the prompt gets its logprobs.
            adapter_name: Name for the adapter loaded from ``adapter_path``.
            adapter_path: peft adapter to apply to every input in this call.
            adapter_paths: Per-input adapters, one entry per input. Accepted for parity with
                ``vLLMSampler``, but served differently: peft activates one adapter at a time, so
                inputs are grouped by adapter and each group gets its own batch. Correct, and slower
                than vLLM in proportion to the number of distinct adapters -- there is no way to mix
                adapters inside a single transformers forward. Mutually exclusive with ``adapter_path``.
            use_base_model: Ignore any loaded adapter for this call.
            strict: When False, an input that fails to encode or generate yields an empty
                ``SampleResponse`` with ``stop_reason='error'`` instead of aborting the batch, so one
                bad row cannot discard a long offline run. Positions are preserved: the result list
                always lines up with ``inputs``.

        Returns:
            One ``SampleResponse`` per input, each holding ``num_samples`` sequences.
        """
        if self.model is not None:
            # Everything below -- encoding, adapter grouping, chunking -- is what the model runs on its
            # own side, so it is forwarded whole rather than half-done here. This does not double-slice:
            # a facade owns no actors, so the decorator above runs it in place and only the model's
            # ``generate`` dispatches.
            return self.model.generate(
                inputs,
                sampling_params=sampling_params,
                adapter_name=adapter_name,
                adapter_path=adapter_path,
                return_encoded=return_encoded,
                use_base_model=use_base_model,
                adapter_paths=adapter_paths,
                strict=strict)
        params = self._coerce_params(sampling_params)
        inputs_list = self._normalize_inputs(inputs)
        if not inputs_list:
            return []
        if adapter_paths is not None:
            if adapter_path is not None:
                raise ValueError('Pass either adapter_path (one adapter for the whole call) or adapter_paths '
                                 '(one per input), not both.')
            if len(adapter_paths) != len(inputs_list):
                raise ValueError(f'adapter_paths has {len(adapter_paths)} entries but there are '
                                 f'{len(inputs_list)} inputs; they must correspond one-to-one so that DP '
                                 'slicing keeps them aligned.')

        logprobs_only = params.max_tokens == 0
        if logprobs_only:
            params = copy(params)
            params.max_tokens = 1

        encoded, failures = self._encode_all(inputs_list, adapter_name, logprobs_only, strict)

        results: List[Optional[SampleResponse]] = [None] * len(inputs_list)
        for index, response in failures.items():
            results[index] = response

        pending = [index for index in range(len(inputs_list)) if results[index] is None]
        # One bucket per distinct adapter: a batch can only run under a single active adapter.
        for path, indices in _group_by_adapter(pending, adapter_paths, adapter_path).items():
            adapter_uri = self._resolve_adapter(path, adapter_name, use_base_model)
            for chunk in _chunks(indices, self.max_batch_size):
                for index, response in zip(chunk, self._run_chunk(chunk, encoded, params, adapter_uri, strict)):
                    results[index] = response

        if not logprobs_only:
            self._attach_features(results, encoded)
        return [r if r is not None else _error_response(encoded.get(i, {})) for i, r in enumerate(results)]

    def sample_stream(
        self,
        inputs: Union[InputFeature, Trajectory, Dict[str, Any]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
    ):
        """Yield ``(delta_text, finish_reason)`` tuples for a single input."""
        if self.model is not None:
            yield from self.model.generate_stream(
                inputs, sampling_params=sampling_params, adapter_name=adapter_name, adapter_path=adapter_path)
            return
        params = self._coerce_params(sampling_params)
        feat = inputs if not self._not_encoded(inputs) else self.encode_trajectory(inputs, adapter_name)
        adapter_uri = self._resolve_adapter(adapter_path, adapter_name, use_base_model=False)
        prompt = self.template.get_vllm_input_ids(feat['input_ids']) if self.template else feat['input_ids']
        yield from self._iter_in_loop(
            self.engine.generate_stream(prompt=prompt, sampling_params=params, lora_request=adapter_uri))

    # ------------------------------------------------------------------ engine passthrough

    def _own_engine(self):
        """The engine this sampler loaded, or a clear failure when it is a facade over a model.

        A facade has no memory of its own to control, and quietly doing nothing would be worse: a caller
        putting a sampler to sleep to free the device would believe it had.
        """
        if self.engine is None:
            raise RuntimeError('This sampler generates through the model it was given, which owns those '
                               'weights and that memory; there is no engine here to control. Call the '
                               'corresponding method on the model instead.')
        return self.engine

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def get_state_keys(self):
        return self._run_in_loop(self._own_engine().get_state_keys())

    @remote_function(dispatch='all', collect='first')
    def sleep(self, level: int = 1) -> None:
        self._run_in_loop(self._own_engine().sleep(level=level))

    @remote_function(dispatch='all', collect='first')
    def wake_up(self, tags: List[str] = None) -> None:
        self._run_in_loop(self._own_engine().wake_up(tags=tags))

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def add_adapter_to_sampler(self, adapter_name: str, config: Any) -> None:
        raise NotImplementedError('TransformersSampler loads adapters from disk; pass adapter_path to sample().')

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def shutdown(self):
        """Release the model and stop the background loop. Idempotent; also runs via ``atexit``."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        try:
            if getattr(self, 'engine', None) is not None:
                self._run_in_loop(self.engine.shutdown())
        except Exception as e:
            logger.warning(f'TransformersSampler engine shutdown error: {e}')
        try:
            if self._async_loop.is_running():
                self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            if self._async_thread.is_alive():
                self._async_thread.join(timeout=5)
        except Exception as e:
            logger.warning(f'TransformersSampler event loop shutdown error: {e}')

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _coerce_params(sampling_params) -> SamplingParams:
        if sampling_params is None:
            return SamplingParams()
        if isinstance(sampling_params, dict):
            return SamplingParams.from_dict(sampling_params)
        return sampling_params

    def _encode_all(self, inputs_list: List[Dict[str, Any]], adapter_name: str, logprobs_only: bool,
                    strict: bool) -> tuple:
        """Encode trajectories to features, isolating per-input encode failures when not strict."""
        encoded: Dict[int, Dict[str, Any]] = {}
        failures: Dict[int, SampleResponse] = {}
        for index, item in enumerate(inputs_list):
            if not self._not_encoded(item):
                encoded[index] = item
                continue
            try:
                encoded[index] = self.encode_trajectory(item, adapter_name, add_generation_prompt=not logprobs_only)
            except Exception as exc:
                if strict:
                    raise
                logger.warning(f'TransformersSampler: dropping input {index}, encode failed: {exc}')
                encoded[index] = {}
                failures[index] = _error_response({})
        return encoded, failures

    def _resolve_adapter(self, adapter_path: Optional[str], adapter_name: str, use_base_model: bool) -> Optional[str]:
        if use_base_model or adapter_path is None:
            return None
        local_path = HubOperation.download_model(model_id_or_path=adapter_path)
        return self.engine.load_lora(local_path, adapter_name or None)

    def _run_chunk(self, chunk: List[int], encoded: Dict[int, Dict[str, Any]], params: SamplingParams,
                   adapter_uri: Optional[str], strict: bool) -> List[SampleResponse]:
        """Generate for one padded batch, falling back to one-at-a-time to isolate a bad input."""
        prompts = [self._prompt_ids(encoded[i]) for i in chunk]
        try:
            return self._run_in_loop(
                self.engine.batch_sample(
                    prompts,
                    params,
                    num_samples=params.num_samples,
                    logprobs=params.logprobs is not None,
                    adapter_uri=adapter_uri,
                ))
        except Exception as exc:
            if strict or len(chunk) == 1:
                raise
            logger.warning(f'TransformersSampler: batch of {len(chunk)} failed ({exc}); retrying individually '
                           'to isolate the offending input')
            out = []
            for prompt in prompts:
                try:
                    out.append(
                        self._run_in_loop(
                            self.engine.batch_sample(
                                [prompt],
                                params,
                                num_samples=params.num_samples,
                                logprobs=params.logprobs is not None,
                                adapter_uri=adapter_uri,
                            ))[0])
                except Exception as inner:
                    logger.warning(f'TransformersSampler: input dropped, generate failed: {inner}')
                    out.append(_error_response({}))
            return out

    def _prompt_ids(self, feat: Dict[str, Any]) -> List[int]:
        input_ids = feat['input_ids']
        if self.template is not None:
            input_ids = self.template.get_vllm_input_ids(input_ids)
        return input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)

    def _attach_features(self, results: List[Optional[SampleResponse]], encoded: Dict[int, Dict[str, Any]]) -> None:
        """Fill in ``new_input_feature`` so downstream training code can consume the samples directly.

        Matches ``vLLMSampler._sample_single``: the prompt feature plus the generated tokens, run
        through the template so labels and any post-pipeline stay consistent.
        """
        if self.template is None:
            return
        for index, response in enumerate(results):
            feat = encoded.get(index)
            if response is None or not feat:
                continue
            for seq in response.sequences:
                if seq.tokens:
                    seq.new_input_feature = self.template.concat_input_feature(feat, seq.tokens)


def _group_by_adapter(pending: List[int], adapter_paths: Optional[List[Optional[str]]],
                      adapter_path: Optional[str]) -> Dict[Optional[str], List[int]]:
    """Bucket the pending input indices by which adapter they need.

    Insertion order is preserved per bucket, so results still land at their original positions. When
    every input shares one adapter (the common case) this is a single bucket and costs nothing.
    """
    if adapter_paths is None:
        return {adapter_path: pending}
    groups: Dict[Optional[str], List[int]] = {}
    for index in pending:
        groups.setdefault(adapter_paths[index], []).append(index)
    return groups


def _chunks(items: List[int], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _error_response(feat: Dict[str, Any]) -> SampleResponse:
    """The placeholder a dropped input gets under ``strict=False``.

    Empty tokens and ``stop_reason='error'`` rather than an exception, so the caller's zip against the
    input list stays aligned and the failure is visible in the data instead of ending the run.
    """
    return SampleResponse(
        sequences=[SampledSequence(stop_reason='error', tokens=[], decoded='')],
        prompt_token_ids=list(feat.get('input_ids') or []),
    )
