# Copyright (c) ModelScope Contributors. All rights reserved.
"""Test tiering. Unmarked tests must run on CPU; ``accel`` needs accelerators; ``slow`` is daily-only.

CI runs one command everywhere -- ``pytest -m "not slow"``. What actually executes is decided here,
from the box's own capability, so a CPU runner, a CUDA box and an NPU box all stay green without CI
carrying three test lists.
"""
import pytest


class Accelerators:
    """The box's accelerators, whichever backend it has."""

    @staticmethod
    def count() -> int:
        try:
            import torch
        except ImportError:
            return 0
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        npu = getattr(torch, 'npu', None)
        return npu.device_count() if npu is not None and npu.is_available() else 0


def pytest_collection_modifyitems(config, items):
    """Skip -- never fail -- tests asking for more accelerators than this box has."""
    available = Accelerators.count()
    for item in items:
        marker = item.get_closest_marker('accel')
        if marker is None:
            continue
        needed = marker.args[0] if marker.args else 1
        if available < needed:
            item.add_marker(pytest.mark.skip(reason=f'needs {needed} accelerator(s), found {available}'))
