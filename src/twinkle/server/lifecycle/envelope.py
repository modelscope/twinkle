# Copyright (c) ModelScope Contributors. All rights reserved.
"""The one place a FutureRecord becomes a TaskEnvelope.

Both the Submit_Endpoint and the Retrieve_Endpoint go through this function so a
``failed`` status always lands in ``error`` and never in ``result``. Duplicating
this mapping per endpoint is how a task that failed inside the Inline_Fast_Path
window loses its payload.
"""
from __future__ import annotations

from typing import Any

from twinkle.server.utils.task_errors import error_payload_from_stored
from twinkle_client.types.lifecycle import TaskEnvelope


def envelope_from_record(
    request_id: str,
    record: dict[str, Any] | None,
    *,
    fallback_status: str = 'pending',
) -> TaskEnvelope:
    """Map a stored ``FutureRecord`` dict to the wire ``TaskEnvelope``.

    Two behaviours are load-bearing:

    - The stored ``FutureRecord`` keeps a failure payload in its ``result`` field
      (changing that would break state backward-compatibility). The wire split of
      ``result`` / ``error`` is done here, which is why there must be exactly one
      mapping point.
    - A failure payload is reconstructed through Part 1's
      ``error_payload_from_stored`` rather than a strict ``ErrorPayload.model_validate``.
      Pre-spec records carry only ``{error, category}``; strict validation would
      make retrieve return 500 for a record that should be a 200 + payload during
      any rolling upgrade.

    ``completed`` with ``result is None`` is a valid success (``step`` /
    ``zero_grad`` / ``lr_step`` all return ``None``); it is NOT treated as a
    failure. The tinker endpoint's ``HTTPException(500, 'Task completed but no
    result found')`` is a bug that this function deliberately does not copy.

    ``failed`` and ``cancelled`` both carry an ``ErrorPayload`` in ``error`` (the
    cancel payload is stored the same way a failure payload is), so the client can
    distinguish them by ``status`` while reading one field.
    """
    record = record or {}
    status = record.get('status', fallback_status)
    common = dict(
        queue_state=record.get('queue_state'),
        queue_state_reason=record.get('queue_state_reason'),
    )
    if status in ('failed', 'cancelled'):
        return TaskEnvelope(
            request_id=request_id,
            status=status,
            error=error_payload_from_stored(record.get('result'), request_id=request_id),
            **common,
        )
    return TaskEnvelope(
        request_id=request_id,
        status=status,
        result=record.get('result') if status == 'completed' else None,
        **common,
    )
