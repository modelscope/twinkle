# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stable receipts for retries of non-streaming sample submissions."""

import hashlib
import hmac
from typing import NamedTuple

import json

from twinkle.protocol.headers import H_SAMPLE_IDEMPOTENCY_KEY
from twinkle_server.exceptions import RequestRejectedError
from twinkle_server.middleware.auth import get_session_id_from_request, get_token_from_request


class SampleSubmission(NamedTuple):
    request_id: str
    fingerprint: str


def sample_submission(request, endpoint, base_model, body) -> SampleSubmission | None:
    key = request.headers.get(H_SAMPLE_IDEMPOTENCY_KEY)
    if key is None and endpoint == 'tinker/asample':
        session_id, sequence_id = body.get('sampling_session_id'), body.get('seq_id')
        if session_id and sequence_id is not None:
            key = json.dumps([session_id, sequence_id])
    if key is None:
        return None
    if not key or len(key) > 256:
        raise RequestRejectedError('Invalid sample idempotency key')
    token = get_token_from_request(request)
    identity = json.dumps([base_model, endpoint, key]).encode()
    owner = f'token:{token}' if token else f'session:{get_session_id_from_request(request)}'
    request_id = 'req_' + hmac.digest(owner.encode(), identity, 'sha256').hex()
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    return SampleSubmission(request_id, hashlib.sha256(canonical).hexdigest())


async def existing_sample(state, submission, token):
    record = await state.futures.get_submission(submission.request_id, submission.fingerprint, token)
    return record.model_dump() if record is not None else None
