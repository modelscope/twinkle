# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from twinkle.utils import requires
from .http.utils import get_base_url, set_base_url, get_api_key, set_api_key
from .manager import TwinkleClient, TwinkleClientError


if TYPE_CHECKING:
    from tinker import ServiceClient


def init_tinker_compat_client(base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> ServiceClient:
    requires('tinker')
    from tinker import ServiceClient
    from twinkle_client.http.utils import get_request_id, get_api_key
    from twinkle_client.utils.patch_tinker import patch_tinker
    
    # Apply patch to bypass tinker:// prefix validation
    patch_tinker()

    if api_key is None:
        api_key = get_api_key()

    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    
    default_headers = {
        "X-Ray-Serve-Request-Id": get_request_id(),
        "Twinkle-Authorization": 'Bearer ' + api_key,
    } | kwargs.pop("default_headers", {})

    service_client = ServiceClient(base_url=base_url, api_key=api_key, default_headers=default_headers, **kwargs)
    
    return service_client

def init_twinkle_client(base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> TwinkleClient:
    """
    Initialize a Twinkle client and setup context variables.
    """
    if base_url is not None:
        set_base_url(base_url)
    else:
        base_url = get_base_url()
        
    if api_key is not None:
        set_api_key(api_key)
    else:
        api_key = get_api_key()
        
    return TwinkleClient(base_url=base_url, api_key=api_key, **kwargs)

__all__ = [
    'TwinkleClient',
    'TwinkleClientError',
    'init_tinker_compat_client',
    'init_twinkle_client'
]
