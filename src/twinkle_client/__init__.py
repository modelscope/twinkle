# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import TwinkleClient


def init_tinker_client(**kwargs) -> None:
    """Initialize Tinker client with Twinkle-specific headers.

    After calling this function, users can directly use::

        from tinker import ServiceClient
        client = ServiceClient(base_url='...', api_key='...')

    The ServiceClient will automatically include Twinkle-specific headers.

    Args:
        **kwargs: Additional keyword arguments (currently unused, reserved for future)

    Example::

        >>> from twinkle_client import init_tinker_client
        >>> init_tinker_client()
        >>> from tinker import ServiceClient
        >>> client = ServiceClient(base_url='http://localhost:8000', api_key='your_token')
    """
    from twinkle.utils import requires

    requires('tinker')
    from twinkle_client.utils.patch_tinker import patch_tinker

    patch_tinker()


def init_twinkle_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    session_heartbeat_interval: int = 10,
    **kwargs,
) -> 'TwinkleClient':
    """
    Initialize a Twinkle client.

    This function:

    * Resolves ``base_url`` and ``api_key`` (env-vars as fallbacks).
    * Sets both values into the shared context so that all other client objects
      (``MultiLoraTransformersModel``, ``vLLMSampler``, processor clients) created
      afterwards automatically inherit the same server configuration.
    * Creates a server-side session and stores the ``session_id`` in context so
      every subsequent HTTP request carries it in ``X-Twinkle-Session-Id``.
    * Starts a background thread that touches the session every
      ``session_heartbeat_interval`` seconds.

    Args:
        base_url: Twinkle server base URL.  Falls back to ``TWINKLE_SERVER_URL``.
        api_key: Authentication token.  Falls back to ``TWINKLE_SERVER_TOKEN``.
        session_heartbeat_interval: Seconds between session touch calls (default: 10).
        **kwargs: Additional keyword arguments forwarded to :class:`TwinkleClient`.

    Returns:
        An initialised :class:`~twinkle_client.manager.TwinkleClient` instance.
    """
    from .manager import TwinkleClient
    return TwinkleClient(
        base_url=base_url,
        api_key=api_key,
        session_heartbeat_interval=session_heartbeat_interval,
        **kwargs,
    )


__all__ = ['init_tinker_client', 'init_twinkle_client']
