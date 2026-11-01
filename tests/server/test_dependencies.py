# Copyright (c) ModelScope Contributors. All rights reserved.
"""Executable dependency architecture for every ``twinkle.server`` component."""
import grimp
import subprocess
import sys

import pytest

_ROOT = 'twinkle.server'
_ROOT_COMPONENT = '__root__'

# Dependencies must point from a later layer to an earlier layer. Components in
# the same layer are independent and therefore may not import one another.
_LAYERS = (
    frozenset({'exceptions'}),
    frozenset({'config', 'middleware', 'model.tinker_datum', 'runtime', 'task_errors', 'utils', 'validation'}),
    frozenset({'telemetry'}),
    frozenset({'state'}),
    frozenset({'resources'}),
    frozenset({'deployment', 'lifecycle', 'session_resource'}),
    frozenset({'task_queue'}),
    frozenset({'checkpoint', 'data_plane'}),
    frozenset({'gateway', 'model', 'sampler', 'processor'}),
    frozenset({'model.pool', 'sampler.pool'}),
    frozenset({'launcher'}),
    frozenset({'cli'}),
    frozenset({'__main__', _ROOT_COMPONENT}),
)


def _component(module: str) -> str:
    if module == _ROOT:
        return _ROOT_COMPONENT
    relative = module.removeprefix(f'{_ROOT}.')
    for component in ('model.pool', 'sampler.pool', 'model.tinker_datum'):
        if relative == component or relative.startswith(component + '.'):
            return component
    return relative.split('.', 1)[0]


@pytest.mark.parametrize('statement', [
    'import twinkle.server.exceptions',
    'from twinkle.server.config import ServerConfig',
    'from twinkle.server.state.models import FutureRecord',
    'from twinkle.server.state import SessionRecord',
])
def test_lightweight_imports_do_not_load_runtime(statement):
    code = statement + '''
import sys
assert not any(name in sys.modules for name in (
    'torch', 'ray', 'twinkle.server.launcher', 'twinkle.server.model.pool.actor', 'twinkle.hub.hub'
)), sorted(sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_hub_first_import_does_not_reenter_plugin_loader():
    code = '''
from twinkle.hub.hub import HFHub
from twinkle.utils.loader import Plugin
assert HFHub.__module__ == 'twinkle.hub.hub'
assert Plugin.__module__ == 'twinkle.utils.loader'
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_public_launcher_is_the_actual_implementation():
    import twinkle.server
    from twinkle.server.launcher.server_launcher import ServerLauncher, launch_server
    assert twinkle.server.ServerLauncher is ServerLauncher
    assert twinkle.server.launch_server is launch_server
    assert twinkle.server.__all__ == ['ServerLauncher', 'launch_server']


def test_server_components_form_declared_dag():
    """Reject cycles, upward imports, same-layer coupling, and unclassified components."""
    graph = grimp.build_graph('twinkle', include_external_packages=False)
    server_modules = {
        module for module in graph.modules if module == _ROOT or module.startswith(f'{_ROOT}.')
    }
    observed_components = {_component(module) for module in server_modules}
    declared_components = set().union(*_LAYERS)
    assert observed_components == declared_components, (
        f'Update the server DAG when adding or removing components: '
        f'undeclared={sorted(observed_components - declared_components)}, '
        f'missing={sorted(declared_components - observed_components)}')

    layer_by_component = {
        component: layer_index for layer_index, layer in enumerate(_LAYERS) for component in layer
    }
    component_edges = {
        (_component(importer), _component(imported))
        for importer in server_modules
        for imported in graph.find_modules_directly_imported_by(importer)
        if imported in server_modules and _component(importer) != _component(imported)
    }
    violations = sorted(
        (importer, imported) for importer, imported in component_edges
        if layer_by_component[importer] <= layer_by_component[imported])
    assert not violations, (
        'Server dependencies must point strictly down the declared DAG; '
        f'invalid edges={violations}')


@pytest.mark.parametrize('component', ['model.pool', 'sampler.pool'])
def test_controller_import_does_not_load_ray_composition(component):
    code = f'''import twinkle.server.{component}.controller
import sys
assert 'ray' not in sys.modules or not sys.modules['ray'].is_initialized()
assert not any('twinkle.server.{component}.' + name in sys.modules for name in ('actor', 'driver', 'app'))
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
