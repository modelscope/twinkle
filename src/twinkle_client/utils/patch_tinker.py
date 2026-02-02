# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Patch tinker's internal_client_holder to bypass model_path prefix validation.

This module patches the _create_sampling_session method to allow model_path
without the 'tinker://' prefix requirement.
"""

from __future__ import annotations

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


def patch_tinker():
    """
    Apply patches to tinker library.
    
    This function patches the InternalClientHolder._create_sampling_session
    method to bypass the 'tinker://' prefix validation for model_path.
    
    This patch is idempotent - calling it multiple times has no additional effect.
    """
    global _patched
    if _patched:
        return
    
    try:
        from tinker.lib.internal_client_holder import InternalClientHolder
        InternalClientHolder._create_sampling_session = _create_sampling_session
        _patched = True
    except ImportError:
        # tinker not installed, skip patching
        pass
