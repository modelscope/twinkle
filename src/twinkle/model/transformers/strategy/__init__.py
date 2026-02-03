# Copyright (c) ModelScope Contributors. All rights reserved.
from .accelerate import AccelerateStrategy
from .native_fsdp import NativeFSDPStrategy

__all__ = ["AccelerateStrategy", "NativeFSDPStrategy"]
