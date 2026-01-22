# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from twinkle.utils import requires

if TYPE_CHECKING:
    from tinker import ServiceClient

def init_tinker_compat_client(base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> ServiceClient:
    requires('tinker')
    from tinker import ServiceClient
    from twinkle_client.http.utils import TWINKLE_REQUEST_ID, TWINKLE_SERVER_TOKEN
    
    default_headers = {
        "X-Ray-Serve-Request-Id": TWINKLE_REQUEST_ID,
        "Authorization": 'Bearer ' + (api_key or TWINKLE_SERVER_TOKEN),
    } | kwargs.pop("default_headers", {})
    
    return ServiceClient(base_url=base_url, api_key=api_key, default_headers=default_headers, **kwargs)
