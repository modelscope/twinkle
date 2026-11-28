# Copyright (c) ModelScope Contributors. All rights reserved.
"""Adapt the pinned Tinker SDK to Twinkle authentication and training."""

from __future__ import annotations
import os
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Mapping

from twinkle.protocol.headers import build_request_headers
from twinkle_client.http.context import capture_transport

_patched = False


@dataclass
class _SampleCall:
    sequence: int | None = None


_sample_call: ContextVar[_SampleCall | None] = ContextVar('tinker_sample_call', default=None)


def _sampling_retry_handler(base):
    class SamplingRetryHandler(base):
        async def execute(self, func, *args, **kwargs):
            binding = _sample_call.set(_SampleCall())
            try:
                return await super().execute(func, *args, **kwargs)
            finally:
                _sample_call.reset(binding)

    return SamplingRetryHandler


def _stable_sampling_request(original):
    @wraps(original)
    async def asample(self, *, request, **kwargs):
        # Tinker 0.16.1 allocates seq_id inside the retry loop. Reuse the first
        # one for this logical call, including a retry after result polling.
        if (call := _sample_call.get()) is not None:
            if call.sequence is None:
                call.sequence = request.seq_id
            elif request.seq_id != call.sequence:
                request = request.model_copy(update={'seq_id': call.sequence})
        return await original(self, request=request, **kwargs)

    return asample


def _restore_sampling_client(state):
    from tinker.lib.public_interfaces.sampling_client import _unpickle_sampling_client

    patch_tinker()
    return _unpickle_sampling_client(state)


def _sampling_client_reduce(original):
    @wraps(original)
    def reduce(self):
        # A spawned SDK sidecar also needs the sampling compatibility patch.
        _, args = original(self)
        return _restore_sampling_client, args

    return reduce


async def _create_sampling_session(self, model_path: str | None = None, base_model: str | None = None) -> str:
    """Patched version that skips the tinker:// prefix validation."""
    from tinker import types
    from tinker.lib.internal_client_holder import ClientConnectionPoolType

    sampling_session_seq_id = self._sampling_client_counter
    self._sampling_client_counter += 1
    with self.aclient(ClientConnectionPoolType.SESSION) as client:
        request = types.CreateSamplingSessionRequest(
            session_id=self._session_id,
            sampling_session_seq_id=sampling_session_seq_id,
            model_path=model_path,
            base_model=base_model,
        )
        result = await client.service.create_sampling_session(request=request)
        return result.sampling_session_id


def _patched_async_tinker_init(
    self,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | Any | None | Any = None,
    max_retries: int = 2,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: Any | None = None,
    _strict_response_validation: bool = False,
) -> None:
    """Patched version of AsyncTinker.__init__ that skips 'tml-' prefix validation."""
    from tinker._exceptions import TinkerError
    from tinker._types import NOT_GIVEN

    # Get api_key from environment if not provided
    if api_key is None:
        api_key = os.environ.get('TWINKLE_SERVER_TOKEN')
    if api_key is None:
        raise TinkerError('The api_key client option must be set either by passing api_key to the client '
                          'or by setting the TWINKLE_SERVER_TOKEN environment variable')
    # REMOVED: api_key 'tml-' prefix validation
    # Original code:
    # if not api_key.startswith("tml-"):
    #     raise TinkerError("The api_key must start with the 'tml-' prefix")

    self.api_key = api_key

    if base_url is None:
        base_url = os.environ.get('TINKER_BASE_URL')
    if base_url is None:
        base_url = 'https://tinker.thinkingmachines.dev/services/tinker-prod'

    # Import the parent class and call its __init__
    from tinker._base_client import AsyncAPIClient
    from tinker._version import __version__

    if timeout is None:
        timeout = NOT_GIVEN

    AsyncAPIClient.__init__(
        self,
        version=__version__,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
        http_client=http_client,
        custom_headers=default_headers,
        custom_query=default_query,
        _strict_response_validation=_strict_response_validation,
    )

    self._idempotency_header = 'X-Idempotency-Key'


def _patched_from_tinker_path(cls, tinker_path: str) -> Any:
    """Patched version that supports both 'tinker://' and 'twinkle://' prefixes."""
    prefix = None
    if tinker_path.startswith('tinker://'):
        prefix = 'tinker://'
    elif tinker_path.startswith('twinkle://'):
        prefix = 'twinkle://'

    if prefix is None:
        raise ValueError(f'Invalid tinker path: {tinker_path}')

    parts = tinker_path[len(prefix):].split('/')
    if len(parts) != 3:
        raise ValueError(f'Invalid tinker path: {tinker_path}')
    if parts[1] not in ['weights', 'sampler_weights']:
        raise ValueError(f'Invalid tinker path: {tinker_path}')
    checkpoint_type = 'training' if parts[1] == 'weights' else 'sampler'
    return cls(
        tinker_path=tinker_path,
        training_run_id=parts[0],
        checkpoint_type=checkpoint_type,
        checkpoint_id='/'.join(parts[1:]),
    )


def _make_patched_service_client_init(original):

    def _patched_service_client_init(self, user_metadata=None, **kwargs):
        """Patched version of ServiceClient.__init__ that injects Twinkle-specific headers."""
        context = capture_transport().context
        api_key = kwargs.get('api_key')
        if api_key is None:
            api_key = context.api_key

        twinkle_headers = build_request_headers(context.client_id, 'Bearer ' + api_key, client_id=context.client_id)
        # Merge: caller-supplied default_headers take precedence over twinkle_headers
        user_default_headers = kwargs.pop('default_headers', {})
        kwargs['default_headers'] = twinkle_headers | user_default_headers

        original(self, user_metadata=user_metadata, **kwargs)

    return _patched_service_client_init


def _create_full_training_client_submit(self, base_model, seed=None, user_metadata=None):
    """Submit a full-parameter (non-LoRA) training-client creation request.

    Mirrors the SDK's ``_create_lora_training_client_submit`` but sends
    ``lora_config=None`` so the Twinkle server routes to a full-parameter
    (``train_mode: full``) deployment. Returns the same ``TrainingClient`` so
    the training loop (forward_backward / optim_step / save_weights / ...) is
    identical to the LoRA path.
    """
    from tinker.lib.internal_client_holder import ClientConnectionPoolType
    from tinker.lib.public_interfaces import service_client as _sc

    session_id = self.holder.get_session_id()
    model_seq_id = self.holder.get_training_client_id()

    async def _create_full_training_client_async():
        start_time = _sc.time.time()
        with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            request = _sc.types.CreateModelRequest(
                session_id=session_id,
                model_seq_id=model_seq_id,
                base_model=base_model,
                lora_config=None,
                user_metadata=user_metadata,
            )
            future = await client.models.create(request=request)
        create_model_response = await _sc._APIFuture(
            _sc.types.CreateModelResponse,
            self.holder,
            future,
            request_start_time=start_time,
            request_type='CreateModel',
            queue_state_observer=_sc.QueueStateLogger(base_model, 'Model creation'),
        ).result_async()
        model_id = create_model_response.model_id
        from tinker.lib.public_interfaces.training_client import TrainingClient

        training_client = TrainingClient(self.holder, model_seq_id=model_seq_id, model_id=model_id)
        _sc.logger.info(f'Full-parameter TrainingClient initialized for model {model_id}')
        return training_client

    return self.holder.run_coroutine_threadsafe(_create_full_training_client_async())


def _create_full_training_client(self, base_model, seed=None, user_metadata=None):
    """Create a full-parameter (non-LoRA) training client (blocking)."""
    return _create_full_training_client_submit(self, base_model, seed=seed, user_metadata=user_metadata).result()


async def _create_full_training_client_async(self, base_model, seed=None, user_metadata=None):
    """Async variant of ``create_full_training_client``."""
    return await _create_full_training_client_submit(
        self, base_model, seed=seed, user_metadata=user_metadata).result_async()


def patch_tinker():
    """
    Apply patches to tinker library.

    This function patches:
    1. InternalClientHolder._create_sampling_session to bypass 'tinker://' prefix validation
    2. AsyncTinker.__init__ to bypass 'tml-' prefix validation for api_key
    3. ParsedCheckpointTinkerPath.from_tinker_path to support both 'tinker://' and 'twinkle://' prefixes
    4. ServiceClient.__init__ to inject Twinkle-specific headers
    5. ServiceClient.create_full_training_client to enable full-parameter (non-LoRA) training
    6. SamplingClient to reuse one request sequence across SDK retries

    This patch is idempotent - calling it multiple times has no additional effect.
    """
    global _patched
    if _patched:
        return

    try:
        # Patch 1: bypass tinker:// prefix validation for model_path
        from tinker.lib.internal_client_holder import InternalClientHolder
        InternalClientHolder._create_sampling_session = _create_sampling_session

        # Patch 2: bypass tml- prefix validation for api_key
        from tinker._client import AsyncTinker
        AsyncTinker.__init__ = _patched_async_tinker_init

        # Patch 3: support both tinker:// and twinkle:// prefixes for checkpoint paths
        from tinker.types.checkpoint import ParsedCheckpointTinkerPath
        ParsedCheckpointTinkerPath.from_tinker_path = classmethod(_patched_from_tinker_path)

        # Patch 4: inject Twinkle-specific headers by patching ServiceClient.__init__.
        from tinker.lib.public_interfaces.service_client import ServiceClient
        ServiceClient.__init__ = _make_patched_service_client_init(ServiceClient.__init__)

        # Patch 5: add a full-parameter (non-LoRA) training-client entrypoint.
        ServiceClient.create_full_training_client = _create_full_training_client
        ServiceClient.create_full_training_client_async = _create_full_training_client_async

        from tinker.lib.public_interfaces import sampling_client
        from tinker.resources.sampling import AsyncSamplingResource

        sampling_client.RetryHandler = _sampling_retry_handler(sampling_client.RetryHandler)
        sampling_client.SamplingClient.__reduce__ = _sampling_client_reduce(sampling_client.SamplingClient.__reduce__)
        AsyncSamplingResource.asample = _stable_sampling_request(AsyncSamplingResource.asample)

        _patched = True
    except ImportError:
        # tinker not installed, skip patching
        pass
