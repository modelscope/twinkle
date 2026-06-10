# Copyright (c) ModelScope Contributors. All rights reserved.
"""Typer CLI tests.

Pins the behaviour of the launcher CLI (``twinkle.server.cli.app``): config
loading and env-var overrides.
"""
from __future__ import annotations

import json
from pathlib import Path
from typer.testing import CliRunner

from tests.server.fixtures import MOCK_SERVER_CONFIG
from twinkle.server.cli.app import app
from twinkle.server.config import ServerConfig

EXAMPLE = MOCK_SERVER_CONFIG
MOCK_CFG = MOCK_SERVER_CONFIG

# ---------- CLI subcommand existence + exit codes ----------------------- #


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


# ---------- print-config round-trip -------------------------------------- #


def test_print_config_round_trip(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ['print-config', '--config', str(EXAMPLE), '--format', 'json'])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    rebuilt = ServerConfig.model_validate(payload)
    original = ServerConfig.from_yaml(EXAMPLE)
    assert rebuilt == original


def test_print_config_rejects_unknown_format() -> None:
    """``--format=xml`` is rejected instead of silently emitting YAML."""
    runner = CliRunner()
    res = runner.invoke(app, ['print-config', '--config', str(EXAMPLE), '--format', 'xml'])
    assert res.exit_code != 0
    assert 'format' in res.output.lower()


# ---------- env-var override --------------------------------------------- #


def test_env_var_overrides_when_flag_omitted(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv('TWINKLE_SERVER_CONFIG', str(EXAMPLE))
    res = runner.invoke(app, ['check-config'])
    assert res.exit_code == 0


# ---------- example config loads ----------------------------------------- #


def test_example_config_loads_via_server_config() -> None:
    cfg = ServerConfig.from_yaml(EXAMPLE)
    assert isinstance(cfg, ServerConfig)
    assert any(a.import_path == 'model' for a in cfg.applications)
    assert any(a.import_path == 'sampler' for a in cfg.applications)
