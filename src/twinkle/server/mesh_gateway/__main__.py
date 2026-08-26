# Copyright (c) ModelScope Contributors. All rights reserved.
"""Standalone launcher for the CPU-side Service Mesh gateway."""
from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    from .app import create_mesh_gateway_app

    gpu_service_id = os.environ['TWINKLE_GPU_SERVICE_ID']
    app = create_mesh_gateway_app(
        gpu_service_id=gpu_service_id,
        mesh_url=os.getenv('TWINKLE_SERVICE_MESH_URL', 'http://127.0.0.1:8880/api/v2/inference'),
        timeout_seconds=float(os.getenv('TWINKLE_SERVICE_MESH_TIMEOUT_SECONDS', '620')),
    )
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '9000')))


if __name__ == '__main__':
    main()
