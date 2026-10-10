import importlib.util
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from ray.runtime_env import RuntimeEnv

from twinkle.utils.network import find_node_ip
from twinkle.utils import parallel
from twinkle.infra._ray.ray_helper import RayHelper


def test_loopback_only_ip_is_valid_runtime_env(monkeypatch):
    monkeypatch.setattr('psutil.net_if_addrs', lambda: {
        'lo': [SimpleNamespace(family=socket.AF_INET, address='127.0.0.1')]})
    ip = find_node_ip()
    assert ip == '127.0.0.1'
    RuntimeEnv(env_vars={'MASTER_ADDR': ip})


def test_real_interface_precedes_virtual_and_loopback(monkeypatch):
    monkeypatch.setattr('psutil.net_if_addrs', lambda: {
        'docker0': [SimpleNamespace(family=socket.AF_INET, address='172.17.0.1')],
        'eth0': [SimpleNamespace(family=socket.AF_INET, address='10.0.0.1')]})
    assert find_node_ip() == '10.0.0.1'


def test_import_does_not_write_to_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location('isolated_parallel', parallel.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not (tmp_path / '.locks').exists()
    with module.processing_lock('model'):
        pass
    assert not (tmp_path / '.locks').exists()


def test_explicit_lock_dir_and_claim_sessions(monkeypatch, tmp_path):
    monkeypatch.setenv('TWINKLE_LOCK_DIR', str(tmp_path / 'locks'))
    monkeypatch.setenv('TWINKLE_SESSION_ID', 'first')
    with parallel.processing_lock('model'):
        assert list((tmp_path / 'locks').glob('*.lock'))
    assert parallel.try_claim_once('same')
    assert not parallel.try_claim_once('same')
    monkeypatch.setenv('TWINKLE_SESSION_ID', 'second')
    assert parallel.try_claim_once('same')


def test_explicit_unwritable_lock_directory(monkeypatch, tmp_path):
    not_a_directory = tmp_path / 'file'
    not_a_directory.write_text('x')
    monkeypatch.setenv('TWINKLE_LOCK_DIR', str(not_a_directory))
    with pytest.raises(OSError, match=str(not_a_directory)):
        with parallel.processing_lock('model'):
            pass


@pytest.mark.parametrize('address', [None, '', 'invalid'])
def test_invalid_master_address_is_rejected(address):
    with pytest.raises(ValueError, match='Invalid Ray master address'):
        RayHelper._validate_master_address(address)


def test_loopback_master_only_allowed_for_single_node(monkeypatch):
    monkeypatch.setattr('ray.nodes', lambda: [{'Alive': True}, {'Alive': False}])
    RayHelper._validate_master_address('127.0.0.1')
    monkeypatch.setattr('ray.nodes', lambda: [{'Alive': True}, {'Alive': True}])
    with pytest.raises(ValueError, match='multi-node'):
        RayHelper._validate_master_address('127.0.0.1')
    RayHelper._validate_master_address('10.0.0.1')


def test_read_only_working_directory_can_import_and_lock(tmp_path):
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        pytest.skip('Unix permission test requires a non-root process')
    source = parallel.__file__
    script = f'''import importlib.util
spec = importlib.util.spec_from_file_location('parallel', {source!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module.processing_lock('readonly-test'):
    pass
'''
    tmp_path.chmod(0o555)
    try:
        result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / '.locks').exists()
    finally:
        tmp_path.chmod(0o755)


def test_claim_has_one_winner_across_processes(tmp_path):
    script = f'''import importlib.util
spec = importlib.util.spec_from_file_location('parallel', {parallel.__file__!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(int(module.try_claim_once('shared-process-key')))
'''
    env = dict(os.environ, TWINKLE_LOCK_DIR=str(tmp_path / 'locks'), TWINKLE_SESSION_ID='shared-session')
    def claim(_):
        result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return int(result.stdout.strip())
    with ThreadPoolExecutor(max_workers=3) as executor:
        assert sum(executor.map(claim, range(3))) == 1
