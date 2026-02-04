# Copyright (c) ModelScope Contributors. All rights reserved.

from .model import build_model_app as _build_model_app
from .sampler import build_sampler_app as _build_sampler_app
from .server import build_server_app
from ..utils import wrap_builder_with_device_group_env


build_model_app = wrap_builder_with_device_group_env(_build_model_app)
build_sampler_app = wrap_builder_with_device_group_env(_build_sampler_app)