# Copyright (c) ModelScope Contributors. All rights reserved.
"""Phase 3 config-drift integration tests against Docker Redis (R15).

Verifies the launch-time signature gate end-to-end:
1. First launch with persistence: redis stores the signature on a fresh DB.
2. Relaunch with the **same** persistence-relevant config returns clean.
3. Relaunch with a **changed** persistence-relevant config raises
   ``ConfigMismatchError`` from ``validate_against_backend`` and the
   ``launch`` CLI exits non-zero with the diff + remediation hint.
"""
from __future__ import annotations

import asyncio
import os
import pytest
import re
import uuid
import yaml
from pathlib import Path
from typer.testing import CliRunner
from unittest import mock

from twinkle.server.cli.app import app
from twinkle.server.config import ServerConfig
from twinkle.server.exceptions import ConfigMismatchError
from twinkle.server.state.backend.factory import PersistenceConfig, create_backend
from twinkle.server.state.backend.redis_backend import RedisBackend
from twinkle.server.state.config_signature import _SIGNATURE_KEY, compute_signature, validate_against_backend

REDIS_URL = os.environ.get('TWINKLE_TEST_REDIS_URL', 'redis://localhost:6379/0')


def _can_reach_redis() -> bool:

    async def _check() -> bool:
        backend = RedisBackend(REDIS_URL)
        try:
            return await backend.health_check()
        except Exception:
            return False
        finally:
            try:
                await backend.close()
            except Exception:
                pass

    try:
        return asyncio.run(_check())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _can_reach_redis(),
    reason=f'Redis at {REDIS_URL} unreachable',
)


@pytest.fixture
def fresh_prefix() -> str:
    return f'twinkle-drift-{uuid.uuid4().hex[:8]}::'


@pytest.fixture
def write_config(tmp_path: Path):
    """Build a YAML config file with a parametric persistence section."""

    def _write(persistence: dict) -> Path:
        payload = {
            'http_options': {
                'host': 'localhost',
                'port': 8000
            },
            'telemetry': {
                'enabled': False
            },
            'persistence':
            persistence,
            'applications': [{
                'name': 'server',
                'route_prefix': '/api/v1',
                'import_path': 'server',
                'args': {
                    'supported_models': ['mock']
                },
            }],
        }
        path = tmp_path / f'config-{uuid.uuid4().hex[:6]}.yaml'
        path.write_text(yaml.safe_dump(payload))
        return path

    return _write


# ---------- direct validate_against_backend behaviour --------------------- #


@pytest.mark.asyncio
async def test_first_run_stores_signature_then_match_passes(fresh_prefix: str) -> None:
    pcfg = PersistenceConfig(mode='redis', redis_url=REDIS_URL, key_prefix=fresh_prefix)
    payload = {'persistence': pcfg.model_dump(mode='json')}

    # Cleanly fresh — first run stores.
    await validate_against_backend(pcfg, payload)
    backend = create_backend(pcfg)
    try:
        assert await backend.get(_SIGNATURE_KEY) == compute_signature(payload)
        # Same payload — second run should pass without error.
        await validate_against_backend(pcfg, payload)
    finally:
        for k in await backend.keys('*'):
            await backend.delete(k)
        await backend.close()


@pytest.mark.asyncio
async def test_drift_raises_with_diff_and_remediation(fresh_prefix: str) -> None:
    pcfg = PersistenceConfig(mode='redis', redis_url=REDIS_URL, key_prefix=fresh_prefix)
    initial = {'persistence': pcfg.model_dump(mode='json')}
    drifted = {'persistence': {**pcfg.model_dump(mode='json'), 'redis_url': REDIS_URL + '?changed=1'}}

    backend = create_backend(pcfg)
    try:
        await validate_against_backend(pcfg, initial)
        with pytest.raises(ConfigMismatchError) as exc:
            await validate_against_backend(pcfg, drifted)
        msg = str(exc.value)
        assert 'drifted' in msg.lower() or 'mismatch' in msg.lower()
        assert 'Remediation' in msg
        assert 'redis_url' in msg
    finally:
        for k in await backend.keys('*'):
            await backend.delete(k)
        await backend.close()


# ---------- CLI launch path: drift exit code = 3 -------------------------- #


def test_cli_launch_drift_exit_nonzero_with_diff(fresh_prefix: str, write_config) -> None:
    """``launch`` calls ``validate_against_backend`` BEFORE the heavy
    ServerLauncher import, so we can stub ServerLauncher to a sentinel and
    still observe the drift error."""
    runner = CliRunner()

    cfg_a = write_config({'mode': 'redis', 'redis_url': REDIS_URL, 'key_prefix': fresh_prefix})
    cfg_b = write_config({
        'mode': 'redis',
        'redis_url': REDIS_URL,
        'key_prefix': fresh_prefix,
        # ``file_path`` doesn't apply to redis mode but its serialized
        # presence (or any non-default field) flips the signature.
        'file_path': '/tmp/intentional-drift.json',
    })

    # Make the first launch a no-op after drift validation by stubbing the
    # launcher; we only care about the signature side effects on Redis.
    with mock.patch('twinkle.server.launcher.ServerLauncher') as launcher_spy:
        launcher_spy.return_value.launch = mock.MagicMock(return_value=None)

        first = runner.invoke(app, ['launch', '--config', str(cfg_a)])
        assert first.exit_code == 0, first.output
        assert launcher_spy.call_count == 1

        # Second launch with drifted persistence config — never reaches the launcher.
        launcher_spy.reset_mock()
        second = runner.invoke(app, ['launch', '--config', str(cfg_b)])

    assert second.exit_code == 3, second.output
    assert launcher_spy.call_count == 0
    assert re.search(r'drifted|mismatch', second.output, re.IGNORECASE)
    assert 'Remediation' in second.output

    # `clear persistence` clears the namespace so a follow-up launch with the
    # drifted config can succeed (this is the documented remediation).
    cleared = runner.invoke(app, ['clear', 'persistence', '--config', str(cfg_b)])
    assert cleared.exit_code == 0, cleared.output

    with mock.patch('twinkle.server.launcher.ServerLauncher') as launcher_spy_2:
        launcher_spy_2.return_value.launch = mock.MagicMock(return_value=None)
        post_clear = runner.invoke(app, ['launch', '--config', str(cfg_b)])
    assert post_clear.exit_code == 0, post_clear.output


def test_cli_launch_first_run_succeeds_then_match(fresh_prefix: str, write_config) -> None:
    runner = CliRunner()
    cfg = write_config({'mode': 'redis', 'redis_url': REDIS_URL, 'key_prefix': fresh_prefix})

    with mock.patch('twinkle.server.launcher.ServerLauncher') as launcher_spy:
        launcher_spy.return_value.launch = mock.MagicMock(return_value=None)

        first = runner.invoke(app, ['launch', '--config', str(cfg)])
        second = runner.invoke(app, ['launch', '--config', str(cfg)])

    assert first.exit_code == 0
    assert second.exit_code == 0


# ---------- check-config doesn't touch the backend ------------------------ #


def test_check_config_does_not_touch_redis(fresh_prefix: str, write_config) -> None:
    """check-config only validates the YAML; no signature is stored."""
    cfg = write_config({'mode': 'redis', 'redis_url': REDIS_URL, 'key_prefix': fresh_prefix})
    runner = CliRunner()
    res = runner.invoke(app, ['check-config', '--config', str(cfg)])
    assert res.exit_code == 0

    async def _read_signature() -> object:
        backend = create_backend(PersistenceConfig(mode='redis', redis_url=REDIS_URL, key_prefix=fresh_prefix))
        try:
            return await backend.get(_SIGNATURE_KEY)
        finally:
            await backend.close()

    assert asyncio.run(_read_signature()) is None
