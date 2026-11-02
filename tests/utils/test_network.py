# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for ``twinkle.utils.network``."""
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


# ── find_free_port ────────────────────────────────────────────────────────────


def _listen_on_any_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


def test_find_free_port_works_without_so_reuseport(monkeypatch):
    """``SO_REUSEPORT`` is absent on Windows; setting it unconditionally raised.

    Callers only need *a* free port, so the option has to be set when the
    platform has it and skipped when it does not.
    """
    import socket
    monkeypatch.delattr(socket, 'SO_REUSEPORT', raising=False)
    port = find_free_port()
    assert 0 < port <= 65535


def test_find_free_port_skips_a_port_that_is_already_listening():
    """A port held by a live listener is not free, whatever the platform says."""
    sock, busy = _listen_on_any_port()
    try:
        try:
            returned = find_free_port(start_port=busy, retry=1)
        except RuntimeError:
            return  # refused, which is the correct answer
        assert returned != busy, 'a port another process is listening on was reported as free'
    finally:
        sock.close()


def test_find_free_port_raises_when_the_range_is_exhausted():
    """Returning a port that was never bound turns a startup error into a bind
    failure inside the caller, far from the probe that lied."""
    sock, busy = _listen_on_any_port()
    try:
        with pytest.raises(RuntimeError):
            find_free_port(start_port=busy, retry=1)
    finally:
        sock.close()
