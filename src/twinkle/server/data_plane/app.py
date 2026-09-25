# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from fastapi import FastAPI
from typing import Any

from twinkle.server.deployment import bind_deployment, build_deployment_app
from .store import TQDataRefStore


class DataPlaneManagement:

    def __init__(self, config: dict[str, Any] | None = None):
        self.store = TQDataRefStore(config)


def build_data_plane_app(
    deploy_options: dict[str, Any],
    config: dict[str, Any] | None = None,
):
    from .handlers import register_data_plane_routes

    deploy_options = dict(deploy_options)
    autoscaling = deploy_options.get('autoscaling_config')
    if autoscaling:
        values = autoscaling.model_dump() if hasattr(autoscaling, 'model_dump') else autoscaling
        if int(values.get('min_replicas', 1)) != 1 or int(values.get('max_replicas', 1)) != 1:
            raise ValueError('data_plane must use exactly one replica')
    else:
        if int(deploy_options.get('num_replicas', 1)) != 1:
            raise ValueError('data_plane must use exactly one replica')
        deploy_options.setdefault('num_replicas', 1)

    def register(app: FastAPI, get_self: Any) -> None:
        register_data_plane_routes(app, get_self)

    app = build_deployment_app('DataPlane', register)
    return bind_deployment(
        app,
        DataPlaneManagement,
        deploy_options,
        deployment_name='DataPlaneManagement',
        bind_kwargs={'config': config},
    )
