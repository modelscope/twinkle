# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for ``twinkle.utils.network.find_node_ip``."""
import sys
import types

from twinkle.utils.network import find_node_ip


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
