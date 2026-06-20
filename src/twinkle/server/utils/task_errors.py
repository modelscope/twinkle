# Copyright (c) ModelScope Contributors. All rights reserved.


def task_error_payload(error: str) -> dict[str, str]:
    return {'error': error, 'category': 'Server'}
