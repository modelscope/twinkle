# Copyright (c) ModelScope Contributors. All rights reserved.
"""Test tiering. Unmarked tests must run on CPU; ``accel`` needs accelerators; ``slow`` is daily-only.

CI runs one command everywhere -- ``pytest -m "not slow"``. What actually executes is decided here,
from the box's own capability, so a CPU runner, a CUDA box and an NPU box all stay green without CI
carrying three test lists.
"""
import ctypes
import importlib.util
import os

import pytest


def _preload_pip_cudnn():
    """Pin cuDNN symbol resolution to the pip wheel before Transformer Engine loads.

    On boxes carrying BOTH a system cuDNN (/usr/lib, e.g. 9.14) and torch's pip nvidia-cudnn
    (e.g. 9.19), TE's DT_NEEDED libcudnn.so.9 maps the system main library alongside the pip
    one (same SONAME, two objects). cuDNN then lazily dlopens its sublibraries across the two
    installations and a 9.14 main + 9.19 sublibrary mix dies with
    CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED inside fused attention -- order-dependent suite
    pollution. Loading the pip main + every sublibrary RTLD_GLOBAL first makes all later
    resolutions reuse one consistent family. No-op where the pip wheel is absent.
    """
    spec = importlib.util.find_spec('nvidia.cudnn')
    if spec is None or not spec.submodule_search_locations:
        return
    libdir = os.path.join(spec.submodule_search_locations[0], 'lib')
    main = os.path.join(libdir, 'libcudnn.so.9')
    if not os.path.exists(main):
        return
    try:
        ctypes.CDLL(main, mode=ctypes.RTLD_GLOBAL)
        for name in sorted(os.listdir(libdir)):
            if name.startswith('libcudnn_') and name.endswith('.so.9'):
                ctypes.CDLL(os.path.join(libdir, name), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        # Preloading is an optimization, not a contract: an unresolvable wheel must not
        # break collection for the whole suite. Fall back to the default resolution order.
        import warnings
        warnings.warn(f'pip cuDNN preload skipped ({e}); default symbol resolution applies.', stacklevel=2)


_preload_pip_cudnn()


@pytest.fixture(autouse=True)
def _reset_accelerate_singletons():
    """Clear accelerate's process-global state between tests.

    AcceleratorState/PartialState are singletons keyed on the FIRST setting that built them:
    one test constructing a model with mixed_precision='no' poisons every later test asking
    for a different setting ("AcceleratorState has already been initialized and cannot be
    changed"). Production is one-shot per process so this never bites there; a pytest process
    drives many models through one interpreter. reset_partial_state=True also clears
    PartialState (device/distributed), which AcceleratorState._reset_state alone leaves behind.
    Same pattern as ms-swift tests/v5/_twinkle_runtime.py.
    """
    yield
    try:
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state(reset_partial_state=True)
    except Exception:  # noqa: BLE001 -- teardown is best-effort
        pass

#: Tests of the archived pipelineTQ surface: they import modules that moved into
#: async_rl/legacy/ and would break collection for the whole suite. Skip them at collection.
collect_ignore_glob = ['twinkle_agentic/legacy/*']


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
