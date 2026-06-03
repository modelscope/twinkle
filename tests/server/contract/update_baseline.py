# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regenerate the client-API contract baseline.

Run with::

    python -m tests.server.contract.update_baseline

Only invoke after confirming that the current client-facing surface has been
intentionally changed and approved as part of this refactor.
"""
from __future__ import annotations

from tests.server.contract.client_api_harness import write_baseline


def main() -> None:
    path = write_baseline()
    print(f'Wrote baseline: {path}')


if __name__ == '__main__':
    main()
