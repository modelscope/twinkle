# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Patch tinker's internal_client_holder to bypass model_path prefix validation.

This module patches the _create_sampling_session method to allow model_path
without the 'tinker://' prefix requirement, and patches AsyncTinker.__init__
to bypass the 'tml-' prefix validation for api_key.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, Mapping, Union

_patched = False


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
    timeout: Union[float, Any, None, Any] = None,
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
        api_key = os.environ.get("TINKER_API_KEY")
    if api_key is None:
        raise TinkerError(
            "The api_key client option must be set either by passing api_key to the client or by setting the TINKER_API_KEY environment variable"
        )
    # REMOVED: api_key 'tml-' prefix validation
    # Original code:
    # if not api_key.startswith("tml-"):
    #     raise TinkerError("The api_key must start with the 'tml-' prefix")
    
    self.api_key = api_key

    if base_url is None:
        base_url = os.environ.get("TINKER_BASE_URL")
    if base_url is None:
        base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod"

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

    self._idempotency_header = "X-Idempotency-Key"


def _patched_from_tinker_path(cls, tinker_path: str) -> Any:
    """Patched version that supports both 'tinker://' and 'twinkle://' prefixes."""
    prefix = None
    if tinker_path.startswith("tinker://"):
        prefix = "tinker://"
    elif tinker_path.startswith("twinkle://"):
        prefix = "twinkle://"
    
    if prefix is None:
        raise ValueError(f"Invalid tinker path: {tinker_path}")
    
    parts = tinker_path[len(prefix):].split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid tinker path: {tinker_path}")
    if parts[1] not in ["weights", "sampler_weights"]:
        raise ValueError(f"Invalid tinker path: {tinker_path}")
    checkpoint_type = "training" if parts[1] == "weights" else "sampler"
    return cls(
        tinker_path=tinker_path,
        training_run_id=parts[0],
        checkpoint_type=checkpoint_type,
        checkpoint_id="/".join(parts[1:]),
    )


def patch_tinker():
    """
    Apply patches to tinker library.
    
    This function patches:
    1. InternalClientHolder._create_sampling_session to bypass 'tinker://' prefix validation
    2. AsyncTinker.__init__ to bypass 'tml-' prefix validation for api_key
    3. ParsedCheckpointTinkerPath.from_tinker_path to support both 'tinker://' and 'twinkle://' prefixes
    
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
        
        _patched = True
    except ImportError:
        # tinker not installed, skip patching
        pass
