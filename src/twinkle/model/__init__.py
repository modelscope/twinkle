# Copyright (c) ModelScope Contributors. All rights reserved.
from .transformers import TransformersModel
from .base import TwinkleModel
from .transformers import MultiLoraTransformersModel
try:
    from .megatron import MegatronModel
except Exception:
    # Optional dependency: allow transformers-only usage.
    MegatronModel = None
try:
    from .megatron import MultiLoraMegatronModel
except Exception:
    # Optional dependency: allow transformers-only usage.
    MultiLoraMegatronModel = None
