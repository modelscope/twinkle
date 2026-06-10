# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Patches for Ray Serve to handle HTTP header normalization.

This module patches Ray Serve's HTTPProxy.setup_request_context_and_handle
to handle HTTP header normalization by proxies (AWS/nginx).

The problem:
  curl sends:           serve_multiplexed_model_id: 123
  proxy converts:       Serve-Multiplexed-Model-Id: 123  (_ → -, title-cased)
  uvicorn lowercases:   serve-multiplexed-model-id: 123
  Ray Serve compares:   "serve_multiplexed_model_id" (with underscores)
  RESULT: NO MATCH → multiplexed_model_id is never set

The fix normalizes header names by converting hyphens to underscores for comparison.

IMPORTANT: Ray Serve's ProxyActor runs in a separate worker process.
Use get_runtime_env_for_patches() with ray.init() to ensure the patch
is applied in all worker processes.
"""
from __future__ import annotations

from typing import Tuple

from twinkle.utils.logger import get_logger

logger = get_logger()

# Track if patch has been applied
_patch_applied = False


def _patched_setup_request_context_and_handle(
    self,
    app_name: str,
    handle,
    route: str,
    proxy_request,
    internal_request_id: str,
) -> tuple:
    """Patched version of HTTPProxy.setup_request_context_and_handle.

    This version handles HTTP header normalization by proxies:
    - Converts hyphens to underscores for SERVE_MULTIPLEXED_MODEL_ID comparison
    """
    from ray.serve._private.constants import SERVE_MULTIPLEXED_MODEL_ID

    request_context_info = {
        'route': route,
        'app_name': app_name,
        '_internal_request_id': internal_request_id,
        'is_http_request': True,
    }

    for key, value in proxy_request.headers:
        decoded_key = key.decode()

        # Check for multiplexed model ID header
        # Normalize: convert hyphens to underscores for comparison
        # HTTP proxies convert underscores to hyphens: serve_multiplexed_model_id → serve-multiplexed-model-id
        normalized_key = decoded_key.replace('-', '_')
        if normalized_key == SERVE_MULTIPLEXED_MODEL_ID:
            multiplexed_model_id = value.decode()
            handle = handle.options(multiplexed_model_id=multiplexed_model_id)
            request_context_info['multiplexed_model_id'] = multiplexed_model_id
            logger.debug(f'[Ray Serve Patch] Matched multiplexed_model_id: {multiplexed_model_id}')

        # Original logic for other headers (unchanged)
        if decoded_key == 'x-request-id':
            request_context_info['request_id'] = value.decode()

    import ray.serve.context as serve_context
    serve_context._serve_request_context.set(serve_context._RequestContext(**request_context_info))

    return handle, request_context_info.get('request_id')


def _apply_patch_in_worker_process():
    """Apply patch in Ray worker process.

    This function is called by Ray's worker_process_setup_hook in each worker process.
    """
    global _patch_applied

    if _patch_applied:
        return

    try:
        from ray.serve._private.proxy import HTTPProxy

        HTTPProxy.setup_request_context_and_handle = _patched_setup_request_context_and_handle
        _patch_applied = True

        logger.debug('[Ray Serve Patch] Applied in worker process: '
                     'HTTPProxy.setup_request_context_and_handle patched')
    except ImportError:
        # Ray Serve not available in this worker
        pass
    except Exception as e:
        logger.warning(f'[Ray Serve Patch] Failed to apply in worker process: {e}')


def apply_ray_serve_patches():
    """Apply patches to Ray Serve in the main process.

    Note: This only patches the main process. For Ray Serve's ProxyActor,
    use get_runtime_env_for_patches() with ray.init() to ensure the patch
    is applied in worker processes.
    """
    global _patch_applied

    if _patch_applied:
        return

    try:
        from ray.serve._private.proxy import HTTPProxy

        HTTPProxy.setup_request_context_and_handle = _patched_setup_request_context_and_handle
        _patch_applied = True

        logger.info('Applied Ray Serve patch: HTTPProxy.setup_request_context_and_handle '
                    'now handles header normalization (hyphens → underscores)')
    except ImportError:
        logger.warning('Ray Serve not available, skipping patch')
    except Exception as e:
        logger.warning(f'Failed to apply Ray Serve patch: {e}')


def get_runtime_env_for_patches() -> dict:
    """Get Ray runtime_env to apply patches in worker processes.

    Ray actors run in separate processes. This returns a runtime_env dict
    that configures Ray to run the patch function in each worker process.

    Usage:
        ray.init(runtime_env=get_runtime_env_for_patches())

    Returns:
        dict: Ray runtime_env configuration
    """
    return {'worker_process_setup_hook': ('twinkle.server.utils.ray_serve_patch._apply_patch_in_worker_process')}
