# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

def init_tinker_client(**kwargs) -> None:
    """Initialize Tinker client with Twinkle-specific headers.

    After calling this function, users can directly use:
        from tinker import ServiceClient
        client = ServiceClient(base_url='...', api_key='...')

    The ServiceClient will automatically include Twinkle-specific headers.

    Args:
        **kwargs: Additional keyword arguments (currently unused, reserved for future)

    Example:
        >>> from twinkle import init_tinker_client
        >>> init_tinker_client()
        >>> from tinker import ServiceClient
        >>> client = ServiceClient(base_url='http://localhost:8000', api_key='your_token')
    """
    from twinkle.utils import requires
    
    requires('tinker')
    from twinkle_client.utils.patch_tinker import patch_tinker

    # Apply patches to tinker library (includes header injection)
    patch_tinker()


def init_twinkle_client(base_url: str | None = None, api_key: str | None = None, **kwargs) -> TwinkleClient:
    """
    Initialize a Twinkle client and setup context variables.
    """
    from .http.utils import get_api_key, get_base_url, set_api_key, set_base_url
    from .manager import TwinkleClient, TwinkleClientError
    
    if base_url is not None:
        set_base_url(base_url)
    else:
        base_url = get_base_url()

    if api_key is not None:
        set_api_key(api_key)
    else:
        api_key = get_api_key()

    return TwinkleClient(base_url=base_url, api_key=api_key, **kwargs)


__all__ = ['init_tinker_client', 'init_twinkle_client']
