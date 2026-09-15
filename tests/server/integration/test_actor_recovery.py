# Copyright (c) ModelScope Contributors. All rights reserved.
"""Post-timeout liveness probe and health status bit (T4.2 / R3#2-3).

Binds the real ``ModelManagement`` health methods onto a minimal harness with a
toggleable mock ``ping`` and a direct ``call_backend``. No GPU/Ray/full server.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI

from twinkle.server.model.app import ModelManagement
from twinkle.server.model.twinkle_handlers import _register_twinkle_routes


class _MockModel:

    def __init__(self) -> None:
        self.alive = True

    def ping(self) -> bool:
        if not self.alive:
            raise RuntimeError('actor unreachable (simulated)')
        return True


class _HealthHarness:
    # Reuse the real implementations under test.
    _run_model_health_probe = ModelManagement._run_model_health_probe
    check_model_health = ModelManagement.check_model_health
    mark_unhealthy = ModelManagement.mark_unhealthy
    _probe_after_timeout = ModelManagement._probe_after_timeout

    def __init__(self, model: _MockModel) -> None:
        self.model = model
        self._model_unhealthy = False

    async def call_backend(self, fn, /, *args, admit: bool = True, **kwargs):
        return fn(*args, **kwargs)


@pytest.mark.asyncio
async def test_timeout_probe_marks_unhealthy_then_recovers():
    model = _MockModel()
    h = _HealthHarness(model)

    # Healthy at first.
    result = await h.check_model_health()
    assert result['healthy'] is True
    assert h._model_unhealthy is False

    # A backend timeout fires the probe while the actor is unreachable.
    model.alive = False
    await h._probe_after_timeout()
    assert h._model_unhealthy is True  # /healthz would return 503

    # Actor recovers; one successful probe clears the bit (no restart needed).
    model.alive = True
    result = await h.check_model_health()
    assert result['healthy'] is True
    assert h._model_unhealthy is False


@pytest.mark.asyncio
async def test_health_route_returns_503_when_probe_fails():
    model = _MockModel()
    model.alive = False
    harness = _HealthHarness(model)
    app = FastAPI()
    _register_twinkle_routes(app, lambda: harness)
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/healthz')

    response = await route.endpoint(object(), harness)

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_mark_unhealthy_is_cleared_by_successful_probe():
    h = _HealthHarness(_MockModel())
    h.mark_unhealthy()
    assert h._model_unhealthy is True

    result = await h.check_model_health()
    assert result['healthy'] is True
    assert h._model_unhealthy is False
