# Copyright (c) ModelScope Contributors. All rights reserved.
"""Mapping-driven kernel replacement.

Three public symbols:

- :func:`kernelize`      apply ``mapping`` to a model
- :func:`hub`            build a Hub kernel reference
- :func:`npu_builtin`    the Ascend NPU built-in bundle
"""
from .builtin import npu_builtin
from .core import hub, kernelize

__all__ = ['kernelize', 'hub', 'npu_builtin']
