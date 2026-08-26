# Copyright (c) ModelScope Contributors. All rights reserved.
"""Standalone launcher for the Twinkle DashServing adapter."""
from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    from .app import create_dashserving_app

    upstream_url = os.getenv('TWINKLE_INTERNAL_URL', 'http://127.0.0.1:8000')
    port = int(os.getenv('PORT', '9000'))
    timeout_seconds = float(os.getenv('TWINKLE_DS_TIMEOUT_SECONDS', '600'))
    app = create_dashserving_app(
        upstream_url=upstream_url,
        timeout_seconds=timeout_seconds,
    )
    uvicorn.run(app, host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
