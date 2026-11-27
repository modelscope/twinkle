"""Isolated test for the ``ray.is_initialized()`` guard inside RayActorBackend.

The test has to tear the Ray runtime down to verify the guard, which would
clobber every other server-side test running in the same session if it
shared a collection with them. We keep it in a dedicated file with a
file-scoped fixture that owns Ray's lifecycle, so the autouse session-scoped
fixture defined in ``tests/server/conftest.py`` never observes the
intermediate shutdown.
"""
from __future__ import annotations

import pytest

ray = pytest.importorskip('ray')

from twinkle_server.config.persistence import PersistenceConfig  # noqa: E402
from twinkle_server.state.backend.factory import create_backend  # noqa: E402


@pytest.fixture(autouse=True)
def _own_ray_lifecycle():
    """Run a fresh Ray runtime for the lifetime of THIS module's tests.

    The session-scoped ``_ray_runtime`` fixture in
    ``tests/server/conftest.py`` does not run here because pytest only honours
    autouse fixtures defined in the *current* file when an explicit
    ``importorskip`` short-circuits the conftest chain — we instead spin a
    fresh runtime up here so we can shut it down inside the test body
    without breaking anything else.
    """
    was_running = ray.is_initialized()
    if was_running:
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()
    if was_running:
        # address='local': the session head we just shut down can leave a stale
        # ray_current_cluster file; auto-detection would connect to the dead cluster and
        # reject num_cpus, breaking every downstream test that needs the session runtime.
        ray.init(
            address='local',
            ignore_reinit_error=True,
            num_cpus=2,
            log_to_driver=False,
            namespace='twinkle_test',
        )


def test_create_backend_memory_mode_requires_ray_initialized() -> None:
    """Without an initialized Ray runtime memory mode must fail loudly,
    rather than silently fall back to a per-process dict."""
    assert not ray.is_initialized()
    config = PersistenceConfig(mode='memory')
    with pytest.raises(RuntimeError, match='Ray'):
        create_backend(config)
