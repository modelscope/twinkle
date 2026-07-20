# Copyright (c) ModelScope Contributors. All rights reserved.
"""Mapping-driven kernel replacement.

Public symbols:

- :func:`kernelize`      apply ``mapping`` to a model
- :func:`hub`            build a Hub kernel reference
- :func:`npu_builtin`    the Ascend NPU built-in bundle
- :func:`liger_builtin`  the Liger Kernel built-in bundle (cross-device)
"""
from .builtin import npu_builtin
from .core import hub, kernelize
from .liger import liger_builtin

__all__ = ['kernelize', 'hub', 'npu_builtin', 'liger_builtin']
