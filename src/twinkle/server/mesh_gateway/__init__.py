# Copyright (c) ModelScope Contributors. All rights reserved.
"""Public Twinkle gateway that reaches the training service through Service Mesh."""

from .app import create_mesh_gateway_app

__all__ = ['create_mesh_gateway_app']
