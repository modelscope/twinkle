# Copyright (c) ModelScope Contributors. All rights reserved.
from .twinkle.sampler import build_sampler_app as _build_sampler_app
from .twinkle.model import build_model_app as _build_model_app
from .twinkle.processor import build_processor_app as _build_processor_app
from .twinkle.server import build_server_app
from .utils import wrap_builder_with_device_group_env
from .launcher import ServerLauncher, launch_server


build_model_app = wrap_builder_with_device_group_env(_build_model_app)
build_processor_app = wrap_builder_with_device_group_env(_build_processor_app)
build_sampler_app = wrap_builder_with_device_group_env(_build_sampler_app)

__all__ = [
    'build_model_app',
    'build_processor_app',
    'build_sampler_app',
    'build_server_app',
    'ServerLauncher',
    'launch_server',
]
