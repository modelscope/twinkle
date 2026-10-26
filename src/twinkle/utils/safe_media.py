# Copyright (c) ModelScope Contributors. All rights reserved.
"""Guards for multimodal media loading: SSRF-safe URL fetching and a local-path allowlist.

Two defenses shared by every media entry point (template encoding and the vision
tools -- which together cover training-dataset rows and serving requests):

* :func:`safe_http_get` resolves the host and rejects non-public targets
  (private / loopback / link-local / reserved / multicast ranges -- the cloud
  metadata endpoint 169.254.169.254 is link-local) BEFORE the request, then
  follows redirects manually and re-validates every hop: a public URL that
  redirects to an internal address is an exfiltration attempt, not a media file.
* :func:`checked_media_path` gates local reads on the ``SWIFT_MEDIA_ALLOWED_DIRS``
  allowlist (colon-separated, the same variable legacy swift reads, so one ops
  contract covers both). Unset means unrestricted: the single-node training
  case, where dataset rows legitimately reference files anywhere on disk.

Both raise ``ValueError`` naming the refused target, so a refused fetch is loud
rather than a silently missing image.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests

#: Allowlist variable, shared with legacy swift's ``SafeMediaPath`` for one ops contract.
MEDIA_ALLOWED_DIRS_ENV = 'SWIFT_MEDIA_ALLOWED_DIRS'

#: Manual redirect cap: each hop is re-validated, so this bounds the loop, not the risk.
_MAX_REDIRECTS = 5


def _assert_public_host(host: str) -> None:
    """Refuse hosts whose ANY resolved address is non-public.

    All addresses are checked (not just the first) so a DNS name that answers
    with both a public and a private address cannot slip through; resolving
    once and connecting later is the DNS-rebinding window this closes on the
    resolve side.
    """
    try:
        addrinfos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f'refusing to fetch media from {host!r}: cannot resolve host ({exc}).') from exc
    if not addrinfos:
        raise ValueError(f'refusing to fetch media from {host!r}: host resolves to no address.')
    for addrinfo in addrinfos:
        address = ipaddress.ip_address(addrinfo[4][0])
        if (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved
                or address.is_multicast or address.is_unspecified):
            raise ValueError(f'refusing to fetch media from {host!r}: resolves to non-public address {address}.')


def safe_http_get(url: str, *, timeout: float = 30.0, session: Optional[requests.Session] = None) -> requests.Response:
    """GET a public http(s) URL, validating the target address at every redirect hop.

    A caller-supplied ``session`` keeps its own retry policy and connection pool;
    redirects are still followed here (``allow_redirects=False`` per hop) because
    the validation must see -- and refuse -- each intermediate location. Raises
    ``ValueError`` for a refused target or a redirect loop past the cap; HTTP
    error statuses raise through ``raise_for_status`` as a plain ``requests`` call
    would.
    """
    own_session = session is None
    if own_session:
        session = requests.Session()
    try:
        current = url
        for _hop in range(_MAX_REDIRECTS + 1):
            parsed = urlparse(current)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname:
                raise ValueError(f'unsupported media URL: {current!r}.')
            _assert_public_host(parsed.hostname)
            response = session.get(current, timeout=timeout, allow_redirects=False)
            if response.is_redirect:
                location = response.headers.get('Location')
                if not location:
                    raise ValueError(f'media URL redirected without a Location header: {current!r}.')
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            return response
        raise ValueError(f'media URL exceeded {_MAX_REDIRECTS} redirects: {url!r}.')
    finally:
        if own_session:
            session.close()


def checked_media_path(path: str) -> str:
    """Return ``path`` unchanged when it lies inside the media allowlist.

    The allowlist (``SWIFT_MEDIA_ALLOWED_DIRS``, colon-separated) is unset by
    default, which leaves local reads unrestricted -- the single-node training
    case. When set, every path is resolved through ``realpath`` first, so
    symlinks and ``..`` segments cannot traverse out of an allowed directory.
    """
    allowed = os.environ.get(MEDIA_ALLOWED_DIRS_ENV, '').strip()
    if not allowed:
        return path
    real_path = os.path.realpath(path)
    for root in allowed.split(os.pathsep):
        root = root.strip()
        if not root:
            continue
        real_root = os.path.realpath(root)
        if os.path.commonpath([real_path, real_root]) == real_root:
            return path
    raise ValueError(f'refusing to read media path {path!r}: outside {MEDIA_ALLOWED_DIRS_ENV} allowlist.')
