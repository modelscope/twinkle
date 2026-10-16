# Copyright (c) ModelScope Contributors. All rights reserved.
"""Bounded training retry receipts with a persistent expired-sequence fence."""


def claim_sequence(current, *, sequence, request_id, now, retention):
    floor = current.get('floor') if current else None
    receipts = {}
    for key, receipt in (current or {}).get('receipts', {}).items():
        if receipt['expires_at'] <= now:
            floor = int(key) if floor is None else max(floor, int(key))
        else:
            receipts[key] = receipt
    key = str(sequence)
    # Unexpired receipts remain retryable even if another sequence expired first.
    if key not in receipts and (floor is None or sequence > floor):
        receipts[key] = {'request_id': request_id, 'expires_at': now + retention}
    return {'floor': floor, 'receipts': receipts}


def release_sequence(current, *, sequence, request_id):
    if current is None:
        return None
    receipts = dict(current['receipts'])
    key = str(sequence)
    if receipts.get(key, {}).get('request_id') == request_id:
        receipts.pop(key)
    return {'floor': current['floor'], 'receipts': receipts}
