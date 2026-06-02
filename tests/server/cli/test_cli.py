# Copyright (c) ModelScope Contributors. All rights reserved.
"""Phase 3 — typer CLI + config-drift validation tests (R14, R15, R16).

Properties covered:
- # Feature: server-config-observability-refactor, Property 29: Config-drift detection and first-run storage
"""
from __future__ import annotations

import json
import pytest
import yaml
from pathlib import Path
from typer.testing import CliRunner
from unittest import mock

from twinkle.server.cli.app import app, main
from twinkle.server.config import ServerConfig
from twinkle.server.exceptions import ConfigMismatchError
from twinkle.server.state.backend.factory import PersistenceConfig
from twinkle.server.state.backend.memory_backend import MemoryBackend
from twinkle.server.state.config_signature import _SIGNATURE_KEY, compute_signature, validate_against_backend

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = REPO_ROOT / 'cookbook' / 'client' / 'server' / 'server_config.example.yaml'
MOCK_CFG = REPO_ROOT / 'cookbook' / 'client' / 'server' / 'mock' / 'server_config.yaml'

# ---------- 9.5 CLI subcommand existence + exit codes (R14.3, R14.4) ------ #


def test_subcommands_present() -> None:
    runner = CliRunner()
    out = runner.invoke(app, ['--help'])
    assert out.exit_code == 0, out.output
    for cmd in ('launch', 'check-config', 'print-config', 'clear'):
        assert cmd in out.output


def test_check_config_exit_zero_on_valid() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ['check-config', '--config', str(EXAMPLE)])
    assert res.exit_code == 0
    assert 'ok' in res.output


def test_check_config_nonzero_on_missing(tmp_path: Path) -> None:
    runner = CliRunner()
    p = tmp_path / 'nope.yaml'
    res = runner.invoke(app, ['check-config', '--config', str(p)])
    assert res.exit_code != 0
    assert 'not found' in res.output.lower()


def test_check_config_nonzero_on_invalid(tmp_path: Path) -> None:
    runner = CliRunner()
    p = tmp_path / 'bad.yaml'
    p.write_text('persistence: {mode: redis}\napplications: []\n')  # missing redis_url
    res = runner.invoke(app, ['check-config', '--config', str(p)])
    assert res.exit_code != 0
    assert 'invalid configuration' in res.output.lower() or 'redis_url' in res.output


# ---------- 9.5 print-config round-trip (R14.5) --------------------------- #


def test_print_config_round_trip(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ['print-config', '--config', str(EXAMPLE), '--format', 'json'])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    rebuilt = ServerConfig.model_validate(payload)
    original = ServerConfig.from_yaml(EXAMPLE)
    assert rebuilt == original


# ---------- 9.5 env-var override (R14.6) ---------------------------------- #


def test_env_var_overrides_when_flag_omitted(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv('TWINKLE_SERVER_CONFIG', str(EXAMPLE))
    res = runner.invoke(app, ['check-config'])
    assert res.exit_code == 0


# ---------- 9.5 launch validates drift BEFORE ray.init (R15.1) ------------ #


def test_launch_validates_drift_before_ray_init() -> None:
    """Order check: ``validate_against_backend`` is called before
    ``ServerLauncher`` is even imported (and thus before ray.init)."""
    runner = CliRunner()

    def _abort_drift(*args, **kwargs):
        raise ConfigMismatchError('drift sentinel')

    with mock.patch(
            'twinkle.server.state.config_signature.validate_against_backend',
            side_effect=_abort_drift,
    ):
        # Should never reach the launcher import — patch it to a sentinel that
        # would make the test fail loudly if reached.
        with mock.patch('twinkle.server.launcher.ServerLauncher') as launcher_spy:
            res = runner.invoke(app, ['launch', '--config', str(MOCK_CFG)])
            assert res.exit_code == 3, res.output
            assert 'drift sentinel' in res.output
            assert launcher_spy.call_count == 0


# ---------- Property 29: drift detection + first-run storage (R15.2/4) ---- #


@pytest.mark.asyncio
async def test_property_29_first_run_stores_signature() -> None:
    """First run with no stored signature stores it and returns silently (R15.4)."""
    backend = MemoryBackend()
    cfg_payload = {'persistence': {'mode': 'memory'}}
    pcfg = PersistenceConfig(mode='memory')
    # Patch create_backend to return our shared in-process backend so we can
    # inspect the stored signature afterwards.
    with mock.patch('twinkle.server.state.backend.factory.create_backend', return_value=backend):
        await validate_against_backend(pcfg, cfg_payload)
        assert await backend.get(_SIGNATURE_KEY) == compute_signature(cfg_payload)
        # Second run with same payload is a no-op.
        await validate_against_backend(pcfg, cfg_payload)


@pytest.mark.asyncio
async def test_property_29_drift_raises_with_diff_and_remediation() -> None:
    backend = MemoryBackend()
    pcfg = PersistenceConfig(mode='memory')
    initial = {'persistence': {'mode': 'memory'}}
    later = {'persistence': {'mode': 'file', 'file_path': '/tmp/x.json'}}

    with mock.patch('twinkle.server.state.backend.factory.create_backend', return_value=backend):
        await validate_against_backend(pcfg, initial)
        with pytest.raises(ConfigMismatchError) as exc:
            await validate_against_backend(pcfg, later)

    msg = str(exc.value)
    assert 'drifted' in msg.lower() or 'mismatch' in msg.lower()
    assert 'Remediation' in msg


# ---------- 9.8 example config loads (R16.3) ------------------------------ #


def test_example_config_loads_via_server_config() -> None:
    cfg = ServerConfig.from_yaml(EXAMPLE)
    assert isinstance(cfg, ServerConfig)
    assert any(a.import_path == 'model' for a in cfg.applications)
    assert any(a.import_path == 'sampler' for a in cfg.applications)
