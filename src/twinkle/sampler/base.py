# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from contextlib import contextmanager
from peft import PeftConfig
from typing import Any, List, Optional, Type, Union

import twinkle
from twinkle import get_logger, remote_function
from twinkle.data_format import (InputFeature, PoolingParams, PoolingResponse, SampleResponse, SamplingParams,
                                 Trajectory, is_encoded)
from twinkle.patch import Patch
from twinkle.sampler import generation
from twinkle.template import Template
from twinkle.template.utils import construct_template

logger = get_logger()


class _GenerationGate:
    """The fallback pause for an engine with no native one: ``sample`` calls read, a pause writes.

    ``pause`` stops new generations and waits for the running ones to finish; the gate stays closed
    until ``resume``. Only used when the engine can neither keep nor wait on its own.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._paused = False

    @contextmanager
    def read(self):
        with self._cond:
            while self._paused:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    def pause(self):
        with self._cond:
            self._paused = True
            while self._readers:
                self._cond.wait()

    def resume(self):
        with self._cond:
            self._paused = False
            self._cond.notify_all()


class Sampler(ABC):

    def __init__(self):
        self.engine = None
        self.template = None

    def _init_template(self, template: Optional[Union[Template, Type[Template], str]],
                       template_kwargs: Optional[dict] = None) -> None:
        """Build the constructor's ``template`` (a template instance, class or registered name)."""
        if template is not None:
            self.template = self._build_template(template, **(template_kwargs or {}))

    def _build_template(self, template_cls, **kwargs) -> Template:
        if 'template_spec' not in kwargs and 'model_id' not in kwargs and getattr(self, 'model_id', None):
            kwargs['model_id'] = self.model_id
        return construct_template(template_cls, **kwargs)

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
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Pass template= to the sampler.")
        return generation.encode_trajectory(self.template, trajectory, add_generation_prompt)

    def decode_response(self, token_ids: List[int], adapter_name: str = '') -> str:
        """Decode token ids to text."""
        template = self.template
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Pass template= to the sampler.")
        return template.decode(token_ids)

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """Replace the template on every replica (server deployments pin it per request stream)."""
        self.template = self._build_template(template_cls, **kwargs)

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Async submission (server)
    # ------------------------------------------------------------------
    def _submit_in_loop(self, coro) -> Future:
        """Schedule the coroutine on the engine's background event loop, returning a Future waitable
        from this thread."""
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

    @remote_function(enable_continous_work=True, lazy_collect=True)
    def submit(
        self,
        trajectory: Trajectory,
        sampling_params: Optional[Union[SamplingParams, dict]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        use_base_model: bool = False,
        token_only: bool = False,
    ) -> List[SampleResponse]:
        """Generate one trajectory; the caller gets a Future of the single-element response list.

        One request routed to the least loaded replica; the replica's thread waits on the engine loop
        while the actor keeps taking other requests.
        """
        if not hasattr(self, '_generate_inputs'):
            raise NotImplementedError(f'{type(self).__name__} does not support async submission: it has no '
                                      '_generate_inputs engine coroutine; use the vLLM/SGLang samplers.')
        with self._gate().read():
            return self._submit_in_loop(
                self._generate_inputs([trajectory], sampling_params, adapter_name, adapter_path,
                                      use_base_model=use_base_model, token_only=token_only)).result()

    # ------------------------------------------------------------------
    # Pause / resume around a weight sync
    # ------------------------------------------------------------------
    def _gate(self) -> _GenerationGate:
        gate = self.__dict__.get('_generation_gate')
        if gate is None:
            gate = self.__dict__.setdefault('_generation_gate', _GenerationGate())
        return gate

    def _pause_engine(self) -> bool:
        """Engine-native pause, keeping in-flight requests (or waiting for them). False if unsupported."""
        return False

    def _resume_engine(self) -> None:
        pass

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def pause_generation(self) -> None:
        """Hold generation so weights can be swapped under it; requests continue after ``resume_generation``.

        Prefers the engine's own pause that keeps running requests frozen in place; failing that the
        engine's wait mode; failing both, a gate that blocks new ``sample`` calls and waits for the
        running ones.
        """
        if self._pause_engine():
            return
        self._gate().pause()
        self._gate_holds_pause = True

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def resume_generation(self) -> None:
        """Release :meth:`pause_generation`; idempotent."""
        if self.__dict__.pop('_gate_holds_pause', False):
            self._gate().resume()
        else:
            self._resume_engine()

    # ------------------------------------------------------------------
    # In-flight requests (abort on shutdown / replica removal)
    # ------------------------------------------------------------------
    async def _sample_engine_request(self, request_id: str, coroutine):
        """Run one engine request registered as in-flight; on caller cancellation abort it explicitly."""
        task = asyncio.create_task(coroutine)
        ids = self.__dict__.setdefault('_inflight_request_ids', set())
        ids.add(request_id)
        try:
            return await asyncio.shield(task)
        except BaseException:

            async def cleanup():
                try:
                    await self.engine.abort_request(request_id)
                except BaseException as error:
                    logger.warning('Failed to abort engine request %s; waiting for it to end naturally: %s',
                                   request_id, error)
                await asyncio.gather(task, return_exceptions=True)

            cleanup_task = asyncio.create_task(cleanup())
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            cleanup_task.result()
            raise
        finally:
            ids.discard(request_id)

    async def _abort_all_inflight(self) -> int:
        aborted = 0
        for request_id in list(self.__dict__.get('_inflight_request_ids', ())):
            try:
                await self.engine.abort_request(request_id)
                aborted += 1
            except Exception as error:
                logger.warning('Failed to abort in-flight request %s: %s', request_id, error)
        return aborted

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def abort_all_inflight(self) -> dict:
        """Abort every in-flight generation on this replica (server shutdown)."""
        if not hasattr(self, '_async_loop'):
            return {'aborted': 0}
        return {'aborted': asyncio.run_coroutine_threadsafe(self._abort_all_inflight(), self._async_loop).result()}

    def _drain(self) -> None:
        """Replica removal: abort what is running so those requests are resubmitted elsewhere."""
        if hasattr(self, '_async_loop'):
            asyncio.run_coroutine_threadsafe(self._abort_all_inflight(), self._async_loop).result()

    def _pressure(self) -> float:
        """Engine load in [0, 1+]: (waiting + running) / max_num_seqs; 0 when the engine cannot tell."""
        return 0.0
