# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for ``twinkle.utils.network``."""
import socket
import sys
import types

import pytest

from twinkle.utils.network import find_free_port, find_node_ip


def _fake_psutil(interfaces):
    mod = types.ModuleType('psutil')
    mod.net_if_addrs = lambda: interfaces
    return mod


def _addr(family, address):
    return types.SimpleNamespace(
        family=types.SimpleNamespace(name=family), address=address)


def test_find_node_ip_prefers_non_virtual(monkeypatch):
    monkeypatch.setitem(sys.modules, 'psutil', _fake_psutil({
        'lo': [_addr('AF_INET', '127.0.0.1')],
        'eth0': [_addr('AF_INET', '10.0.0.5')],
        'docker0': [_addr('AF_INET', '172.17.0.1')],
    }))
    assert find_node_ip() == '10.0.0.5'


def test_find_node_ip_falls_back_to_virtual(monkeypatch):
    monkeypatch.setitem(sys.modules, 'psutil', _fake_psutil({
        'lo': [_addr('AF_INET', '127.0.0.1')],
        'docker0': [_addr('AF_INET', '172.17.0.1')],
    }))
    assert find_node_ip() == '172.17.0.1'


def test_find_node_ip_loopback_only(monkeypatch):
    """A host with only loopback addresses (``docker run --network none``).

    ``find_node_ip`` must still return a usable address instead of ``None``:
    callers assign the result to ``MASTER_ADDR``, and ``None`` fails later
    inside Ray's ``env_vars`` validation with an unrelated error.
    """
    monkeypatch.setitem(sys.modules, 'psutil', _fake_psutil({
        'lo': [_addr('AF_INET', '127.0.0.1')],
    }))
    assert find_node_ip() == '127.0.0.1'


def test_find_node_ip_ignores_non_ipv4(monkeypatch):
    monkeypatch.setitem(sys.modules, 'psutil', _fake_psutil({
        'eth0': [_addr('AF_PACKET', 'aa:bb:cc:dd:ee:ff'),
                 _addr('AF_INET6', 'fe80::1')],
    }))
    assert find_node_ip() == '127.0.0.1'


@pytest.mark.parametrize('address', ['', '::1'])
def test_find_free_port_can_be_bound(address):
    port = find_free_port(address=address)
    family = socket.AF_INET6 if address else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        listener.bind(('', port))
        listener.listen(1)


def test_find_free_port_without_reuse_options(monkeypatch):
    monkeypatch.delattr(socket, 'SO_REUSEPORT', raising=False)
    port = find_free_port()
    assert 0 < port <= 65535


@pytest.mark.parametrize('reuse_port', [False, True])
def test_find_free_port_rejects_live_listener(reuse_port):
    if reuse_port and not hasattr(socket, 'SO_REUSEPORT'):
        pytest.skip('SO_REUSEPORT listener is not supported on this platform')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if reuse_port:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        listener.bind(('', 0))
        listener.listen(1)
        busy = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match='No free port found'):
            find_free_port(start_port=busy, retry=1)


def test_find_free_port_empty_range():
    with pytest.raises(RuntimeError, match='No free port found'):
        find_free_port(start_port=20000, retry=0)


def test_find_free_port_retries_after_busy_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('', 0))
        listener.listen(1)
        busy = listener.getsockname()[1]
        port = find_free_port(start_port=busy, retry=65536 - busy)
        assert busy < port <= 65535
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receiver:
            receiver.bind(('', port))
            receiver.listen(1)
