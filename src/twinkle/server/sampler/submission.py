# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stable receipts for retries of non-streaming sample submissions."""

import hashlib
import hmac
import json
from typing import NamedTuple

from twinkle.protocol.headers import H_SAMPLE_IDEMPOTENCY_KEY
from twinkle.server.exceptions import RequestRejectedError
from twinkle.server.middleware.auth import get_session_id_from_request, get_token_from_request


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
    scope = ['token', token] if token else ['session', get_session_id_from_request(request)]
    identity = json.dumps([scope, base_model, endpoint, key]).encode()
    request_id = 'req_' + hashlib.sha256(identity).hexdigest()
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    return SampleSubmission(request_id, hashlib.sha256(canonical).hexdigest())


async def existing_sample(state, submission, token):
    request_id, fingerprint = submission
    record = await state.get_future(request_id)
    if record is not None and (
            (record.get('owner_token') or '') != token
            or not hmac.compare_digest(record.get('submission_fingerprint') or '', fingerprint)):
        raise RequestRejectedError('Sample idempotency key was already used for a different submission')
    return record
