# Copyright (c) ModelScope Contributors. All rights reserved.
"""Twinkle_Client exceptions for the request lifecycle.

Two axes, kept deliberately distinct:

- Transport / HTTP failures (:class:`TwinkleHTTPError`) inherit ``requests.HTTPError``
  so existing ``except requests.HTTPError`` clauses keep working. They carry the
  server's ``error_code`` / ``category`` when the response body had them.
- Task-outcome and polling failures (:class:`TaskFailedError`,
  :class:`TaskWaitTimeoutError`, :class:`TaskRecordLostError`) do NOT inherit
  ``requests.HTTPError``: a task that reaches a ``failed`` terminal state is
  delivered over HTTP 200, so it is not an HTTP-level error.
"""
from __future__ import annotations

from typing import Any, Optional

import requests


class TwinkleHTTPError(requests.HTTPError):
    """An HTTP 4xx/5xx (other than 410) from a twinkle endpoint.

    Inherits ``requests.HTTPError`` so callers already catching that keep working.
    ``status_code`` is the HTTP status; ``error_code`` / ``category`` come from the
    server's structured error body when present (else ``None`` / ``'Unknown'``).
    """

    def __init__(
        self,
        *args: Any,
        status_code: Optional[int] = None,
        error_code: Optional[int] = None,
        category: str = 'Unknown',
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.status_code = status_code
        self.error_code = error_code
        self.category = category
        self.request_id = request_id


class TaskFailedError(Exception):
    """A task reached the ``failed`` terminal state (delivered over HTTP 200).

    The twinkle counterpart of tinker's ``RequestFailedError``
    (``tinker/_exceptions.py``; carries ``message`` / ``request_id`` / ``category``).
    Deliberately NOT a ``requests.HTTPError`` subclass: the HTTP call succeeded, it
    is the *task* that failed, so this is not an HTTP-level error.
    """

    def __init__(
        self,
        error: str,
        *,
        category: str,
        request_id: str,
        error_code: Optional[int] = None,
        details: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        super().__init__(error)
        self.error = error
        self.category = category
        self.request_id = request_id
        self.error_code = error_code
        self.details = details


class TaskCancelledError(Exception):
    """A task reached the ``cancelled`` terminal state (delivered over HTTP 200).

    Distinct from :class:`TaskFailedError`: the task did not fail, it was cancelled
    before it started running (client cancel). Not a ``requests.HTTPError`` -- the
    HTTP call succeeded; the task was simply dropped.
    """

    def __init__(
        self,
        error: str,
        *,
        request_id: str,
        error_code: Optional[int] = None,
    ) -> None:
        super().__init__(error)
        self.error = error
        self.request_id = request_id
        self.error_code = error_code


class TaskWaitTimeoutError(Exception):
    """The Client_Future_Layer stopped polling after ``total_timeout`` seconds.

    "I am not waiting any longer" -- distinct from :class:`TaskRecordLostError`,
    which indicates a state-layer problem. The task itself is guaranteed to reach a
    terminal state by the server; this only means the client gave up.
    """

    def __init__(self, *, request_id: str, waited: float) -> None:
        super().__init__(f'Timed out after {waited:.1f}s waiting for task {request_id}')
        self.request_id = request_id
        self.waited = waited


class TaskRecordLostError(Exception):
    """Retrieve_Endpoint returned 404 for a whole run of consecutive attempts.

    Distinct from :class:`TaskWaitTimeoutError`: a 404 run points at the state layer
    (Redis blip, actor restart) rather than a slow task, so the operator response
    differs.
    """

    def __init__(self, *, request_id: str) -> None:
        super().__init__(f'Task record for {request_id} was not found after repeated retries')
        self.request_id = request_id
