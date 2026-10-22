# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import uuid
import zlib
from abc import ABC, abstractmethod
from concurrent.futures import Future
from peft import PeftConfig
from typing import Any, List, Optional, Type, Union

import twinkle
from twinkle import remote_function
from twinkle.data_format import (InputFeature, PoolingParams, PoolingResponse, SampleResponse, SamplingParams,
                                 Trajectory, is_encoded)
from twinkle.patch import Patch
from twinkle.sampler import generation
from twinkle.template import Template
from twinkle.template.utils import construct_template


def _dispatch_request(worker_count: int, worker_index: int, args: tuple, kwargs: dict, **_):
    """submit's routing policy: one request hashed by request_id onto exactly one worker, the rest
    receive an empty batch.

    Deterministic, stateless, evenly spread: the same request_id always lands on the same worker
    (when combined with sticky ownership the wrapper already pins the call to a single worker, so
    target is always 0 there -- the two stay consistent).
    """
    trajectory = args[0] if args else kwargs.get('trajectory')
    request_id = trajectory.get('request_id') if isinstance(trajectory, dict) else None
    target = zlib.crc32(str(request_id).encode('utf-8')) % worker_count
    shard = trajectory if worker_index == target else None
    if args:
        return (shard, *args[1:]), kwargs
    return args, dict(kwargs, trajectory=shard)


class Sampler(ABC):

    @staticmethod
    def _normalize_sampling_args(args, kwargs):
        args, kwargs = list(args), dict(kwargs)
        inputs = args[0] if args else kwargs['inputs']
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        elif not isinstance(inputs, list) and not getattr(inputs, '_is_lazy_collect', False):
            inputs = [inputs]
        if args:
            args[0] = inputs
        else:
            kwargs['inputs'] = inputs
        return tuple(args), kwargs

    @staticmethod
    def _dispatch_sampling_inputs(worker_count, worker_index, args, kwargs, **_):
        """Slice only the top-level inputs, keeping each input's tokens, tensors and multimodal fields intact."""
        args, kwargs = list(args), dict(kwargs)
        inputs = args[0] if args else kwargs['inputs']
        batch = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        size, remainder = divmod(len(batch), worker_count)
        start = worker_index * size + min(worker_index, remainder)
        stop = start + size + (worker_index < remainder)
        shard = batch[start:stop]
        for index, value in enumerate(args):
            if index > 0 and isinstance(value, list) and len(value) == len(batch):
                args[index] = value[start:stop]
        for key, value in kwargs.items():
            if key != 'inputs' and isinstance(value, list) and len(value) == len(batch):
                kwargs[key] = value[start:stop]
        if args:
            args[0] = shard
        else:
            kwargs['inputs'] = shard
        return tuple(args), kwargs

    def __init__(self):
        self.engine = None
        self.template = None

    @abstractmethod
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[SamplingParams] = None,
        adapter_name: str = '',
        *,
        num_samples: int = 1,
    ) -> List[SampleResponse]:
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
        """
        pass

    def encode(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        pooling_params: Optional[PoolingParams] = None,
        adapter_name: str = '',
    ) -> List[PoolingResponse]:
        """Run a pooling (non-generative) forward and return the pooled output per input.

        The pooling counterpart of :meth:`sample`: one forward per input, no decoding, producing an
        embedding vector / class scores / a relevance score. Only samplers whose backend can serve a
        pooling head (vLLM, SGLang) override this; the default raises so a backend without pooling
        support fails loudly rather than silently returning hidden states that look like embeddings.

        Args:
            inputs: Either InputFeature(s) or Trajectory(s), as for :meth:`sample`.
            pooling_params: Which head to run and how to post-process it. Defaults to a plain embedding.
            adapter_name: Optional LoRA adapter name.

        Returns:
            One PoolingResponse per input, in input order.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support pooling/encode; use a vLLM or '
                                  'SGLang sampler, or a HF forward via build_model(task=...).')

    @abstractmethod
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        ...

    @staticmethod
    def _not_encoded(inputs: Any) -> bool:
        """Check if inputs are not yet encoded (i.e., is Trajectory, not InputFeature).

        Delegates to the single shared predicate so the three backends and the wire
        schema cannot drift apart.
        """
        assert isinstance(inputs, dict), f'Expected dict, got {type(inputs)}'
        return not is_encoded(inputs)

    def _is_trajectory(self, inputs: Any) -> bool:
        """Check if inputs are Trajectory type (not encoded)."""
        if isinstance(inputs, list):
            if not inputs:
                return False
            inputs = inputs[0]
        if isinstance(inputs, dict):
            return self._not_encoded(inputs)
        return False

    def _normalize_inputs(self, inputs) -> List:
        if isinstance(inputs, dict):
            return [inputs]
        return list(inputs)

    def encode_trajectory(self,
                          trajectory: Trajectory,
                          adapter_name: str = '',
                          add_generation_prompt: bool = True) -> InputFeature:
        if self.template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        return generation.encode_trajectory(self.template, trajectory, add_generation_prompt)

    def decode_response(self, token_ids: List[int], adapter_name: str = '') -> str:
        """Decode token ids to text."""
        template = self.template
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        return template.decode(token_ids)

    @remote_function(dispatch='all', collect='first', lazy_collect=False, exclusive=True, replay=True)
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        if 'template_spec' not in kwargs and 'model_id' not in kwargs and hasattr(self, 'model_id'):
            kwargs['model_id'] = self.model_id
        self.template = construct_template(template_cls, **kwargs)

    @remote_function(dispatch='all', collect='first', lazy_collect=False, exclusive=True, replay=True)
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Async trio: submit / pause / stop
    # ------------------------------------------------------------------
    def _submit_in_loop(self, coro) -> Future:
        """Schedule the coroutine on the engine's background event loop, returning a Future waitable
        from this thread.
        
        Engine-side async bookkeeping (loop scheduling and Future hand-off), not cross-call state:
        the Future completes as soon as the request finishes.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._async_loop)

    @staticmethod
    async def _gather_generation_tasks(coroutines):
        """After one input fails or is cancelled, wait for the whole batch's generation coroutines to end before releasing the submission."""
        tasks = []
        try:
            for coroutine in coroutines:
                tasks.append(asyncio.create_task(coroutine))
            # A cancelled parent task does not re-cancel child tasks that are already running cleanup.
            return await asyncio.shield(asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            cleanup = asyncio.gather(*tasks, return_exceptions=True)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            raise

    def submit(
        self,
        trajectory: Trajectory,
        sampling_params: Optional[Union[SamplingParams, dict]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        use_base_model: bool = False,
        token_only: bool = False,
        allow_partial_rollout: bool = False,
    ) -> Future:
        """Submit one trajectory's generation and return a Future immediately (done/result/exception/
        ready share one contract).
        
        Locally this is a same-thread Future (generation runs on the engine's background loop);
        remotely it is a lazy reference that becomes ready only when the generation truly finishes and
        can be passed whole to downstream actors. request_id is established before submission: minted
        in place and written back onto the trajectory when absent. Every result (natural completion or
        the partial after a pause/stop termination) is read from that Future: a single-element list
        (same shape as sample's batch return). ``allow_partial_rollout`` matches sample's semantics:
        an in-flight generation aborted by a weight publish resumes from its own tokens (deep-buffer
        mode); that mode does not pair with pause/stop (the latter are terminal semantics -- the two
        are mutually exclusive deployment choices).
        """
        if not hasattr(self, '_generate_inputs'):
            raise NotImplementedError(f'{type(self).__name__} does not support async submission: it has no '
                                      '_generate_inputs engine coroutine; async RL uses the vLLM/SGLang samplers.')
        if 'request_id' not in trajectory:
            trajectory['request_id'] = uuid.uuid4().hex
        return self._submit(trajectory, sampling_params, adapter_name, adapter_path,
                            use_base_model=use_base_model, token_only=token_only,
                            allow_partial_rollout=allow_partial_rollout)

    @remote_function(dispatch=_dispatch_request, collect='flatten', lazy_collect=True,
                     enable_continous_work=True, sticky=True, sticky_key='trajectory.request_id')
    def _submit(
        self,
        trajectory: Optional[Trajectory],
        sampling_params: Optional[Union[SamplingParams, dict]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        use_base_model: bool = False,
        token_only: bool = False,
        allow_partial_rollout: bool = False,
    ):
        """Worker half: schedule the generation on the engine's background loop and return the Future.

        infra unwraps the wait on the worker side (the caller's reference comes ready when the
        generation finishes); locally the Future is returned as is -- the component body never senses
        the deployment form. Non-target workers of the hash routing receive None and return an empty
        batch immediately.
        """
        if trajectory is None:
            return []
        return self._submit_in_loop(
            self._generate_inputs([trajectory], sampling_params, adapter_name, adapter_path,
                                  use_base_model=use_base_model, token_only=token_only,
                                  allow_partial_rollout=allow_partial_rollout))

    def pause(self, trajectory: Trajectory) -> bool:
        """Terminate the in-flight request and wait for the settle; once True returns, submit's Future
        is necessarily done.
        
        The partial (stop_reason='abort' + generated tokens) lives in submit's Future; assembling the
        resumed trajectory belongs to the caller (multiturnrollout/driver, the multi-turn state layer).
        """
        return self._pause(trajectory['request_id'])

    @remote_function(dispatch='all', collect='first', lazy_collect=False, control=True,
                     sticky_key='request_id', enable_continous_work=True)
    def _pause(self, request_id: str) -> bool:
        """Worker half: abort the engine request and poll the in-flight registry until it is unregistered (settle confirmation)."""
        return self._terminate_request(request_id).result()

    def stop(self, trajectory: Trajectory) -> bool:
        """Terminate the request; the engine operation equals pause's -- the semantic difference is the caller's disposal of the partial (dropped after stop)."""
        return self._pause(trajectory['request_id'])

    def _terminate_request(self, request_id: str) -> Future:
        """Engine-side termination: run abort + settle polling on the background loop, return a Future."""

        async def _terminate():
            await self.engine.abort_request(request_id)
            while request_id in getattr(self, '_inflight_request_ids', ()):
                await asyncio.sleep(0.001)
            return True

        return asyncio.run_coroutine_threadsafe(_terminate(), self._async_loop)
