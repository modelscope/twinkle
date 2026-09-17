# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regenerate the client-API contract snapshots.

Run with::

    python -m tests.server.contract.update_baseline

Two artifacts, with deliberately different git treatment:

- ``client_api_baseline.json`` -- the full field-level surface. NOT committed
  (gitignored): an 8k-line diff on every intentional wire change is noise nobody
  reads, and being regenerated from the code under test it proves nothing on its own.
- ``client_api_routes.json`` -- the compact route inventory. COMMITTED, and it is the
  actual regression guard. Review its diff: every added/removed route and every
  changed response/body model appears there as a readable line.

Only regenerate after confirming that the current client-facing surface changed
intentionally.
"""
from __future__ import annotations

from tests.server.contract.client_api_harness import write_baseline, write_route_inventory


def main() -> None:
    print(f'Wrote generated baseline (not committed): {write_baseline()}')
    print(f'Wrote route inventory  (COMMIT THIS):     {write_route_inventory()}')


if __name__ == '__main__':
    main()
