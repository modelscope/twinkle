import sys
import types
from unittest.mock import patch

import pytest

from twinkle.kernel.core import HubRef, _load_hub_ref


def _install_fake_kernels(layer_obj=None, no_layers=False):
    """Install a fake `kernels` module with a controllable `get_kernel`."""
    fake = types.ModuleType('kernels')

    def fake_get_kernel(repo_id, **kwargs):
        m = types.ModuleType('fake_kernel')
        if not no_layers:
            layers_ns = types.SimpleNamespace()
            if layer_obj is not None:
                layers_ns.MyLayer = layer_obj
            m.layers = layers_ns
        return m

    fake.get_kernel = fake_get_kernel
    sys.modules['kernels'] = fake


def _uninstall_fake_kernels():
    sys.modules.pop('kernels', None)


def test_load_hub_ref_returns_layer():
    sentinel = object()
    _install_fake_kernels(layer_obj=sentinel)
    try:
        ref = HubRef('org/repo', 'MyLayer', revision='main')
        assert _load_hub_ref(ref) is sentinel
    finally:
        _uninstall_fake_kernels()


def test_load_hub_ref_raises_if_layers_missing():
    _install_fake_kernels(no_layers=True)
    try:
        ref = HubRef('org/repo', 'MyLayer', revision='main')
        with pytest.raises(ValueError, match='does not define any layers'):
            _load_hub_ref(ref)
    finally:
        _uninstall_fake_kernels()


def test_load_hub_ref_raises_if_layer_name_missing():
    _install_fake_kernels(layer_obj=None)  # MyLayer not present
    try:
        ref = HubRef('org/repo', 'Missing', revision='main')
        with pytest.raises(ValueError, match='not found'):
            _load_hub_ref(ref)
    finally:
        _uninstall_fake_kernels()


def test_load_hub_ref_install_hint_when_kernels_missing():
    # Force `import kernels` to fail
    sys.modules['kernels'] = None  # short-circuits import to ImportError
    try:
        ref = HubRef('org/repo', 'MyLayer', revision='main')
        with pytest.raises(ImportError, match='pip install kernels'):
            _load_hub_ref(ref)
    finally:
        sys.modules.pop('kernels', None)