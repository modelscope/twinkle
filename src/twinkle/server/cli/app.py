# Copyright (c) ModelScope Contributors. All rights reserved.
"""Typer-based operations CLI.

Provides four subcommands:

- ``launch``           — start the Twinkle Server from a YAML config.
- ``check-config``     — validate a config file; exit 0 on success, non-zero
                        with the validation error on failure.
- ``print-config``     — emit the fully resolved + normalized ``ServerConfig``
                        as YAML or JSON.
- ``clear persistence``— delete persisted state for the namespace derived
                        from a config file.
"""
from __future__ import annotations

import asyncio
import json
import pydantic
import sys
import typer
import yaml
from pathlib import Path

from twinkle.server.config import ServerConfig
from twinkle.server.exceptions import ConfigParseError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help='Operations CLI for Twinkle Server.',
)

clear_app = typer.Typer(
    no_args_is_help=True,
    help='Clear server-side state.',
)
app.add_typer(clear_app, name='clear')

CONFIG_OPTION = typer.Option(
    ...,
    '--config',
    '-c',
    envvar='TWINKLE_SERVER_CONFIG',
    help='Path to the YAML configuration file.',
    metavar='PATH',
)
NAMESPACE_OPTION = typer.Option(
    None,
    '--namespace',
    envvar='TWINKLE_RAY_NAMESPACE',
    help='Ray namespace (overrides ray_namespace in the config).',
)


def _load_config(path: Path) -> ServerConfig:
    """Load + validate ``path``; print typed errors and exit non-zero on failure."""
    try:
        return ServerConfig.from_yaml(path)
    except FileNotFoundError as e:
        typer.echo(f'error: {e}', err=True)
        raise typer.Exit(code=2)
    except ConfigParseError as e:
        typer.echo(f'error: {e}', err=True)
        raise typer.Exit(code=2)
    except pydantic.ValidationError as e:
        # Narrowed from a bare ``except Exception`` so unrelated errors are not
        # mislabelled as "invalid configuration".
        typer.echo(f'error: invalid configuration\n{e}', err=True)
        raise typer.Exit(code=2)


@app.command('launch')
def launch_cmd(
    config: Path = CONFIG_OPTION,
    namespace: str | None = NAMESPACE_OPTION,
) -> None:
    """Start the Twinkle Server from a YAML config file."""
    cfg = _load_config(config)

    # Defer the heavy launcher import until after config validation so the
    # failure path stays cheap (and a missing Ray install doesn't block
    # `check-config`). ``launch_server`` is the single launch choke point.
    from twinkle.server.launcher import launch_server

    launch_server(config=cfg, ray_namespace=namespace or cfg.ray_namespace)


@app.command('check-config')
def check_config_cmd(config: Path = CONFIG_OPTION) -> None:
    """Validate ``config`` and exit 0 on success, non-zero on failure."""
    _load_config(config)
    typer.echo('ok')


@app.command('print-config')
def print_config_cmd(
        config: Path = CONFIG_OPTION,
        fmt: str = typer.Option('yaml', '--format', envvar='TWINKLE_PRINT_FORMAT', help='yaml|json'),
) -> None:
    """Emit the validated, normalized ``ServerConfig`` as YAML or JSON."""
    if fmt not in ('yaml', 'json'):
        raise typer.BadParameter("must be 'yaml' or 'json'", param_hint='--format')
    cfg = _load_config(config)
    payload = cfg.to_yaml_dict()
    if fmt == 'json':
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(yaml.safe_dump(payload, sort_keys=True).rstrip())


@clear_app.command('persistence')
def clear_persistence_cmd(config: Path = CONFIG_OPTION) -> None:
    """Remove persisted state for the namespace derived from ``config``."""
    cfg = _load_config(config)
    from twinkle.server.state.backend.factory import create_backend

    async def _clear() -> int:
        backend = create_backend(cfg.persistence)
        keys = await backend.keys('*')
        removed = 0
        for k in keys:
            await backend.delete(k)
            removed += 1
        return removed

    n = asyncio.run(_clear())
    typer.echo(f'cleared {n} keys from persistence backend (mode={cfg.persistence.mode})')


def main(argv: list[str] | None = None) -> int:
    """Programmatic entry point used by ``__main__.py`` and tests.

    Runs the typer app in standalone mode and converts its ``SystemExit``
    into a plain return code so callers can react without re-trapping.
    """
    try:
        app(args=argv, standalone_mode=True)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code) if not isinstance(code, int) else code
    return 0


if __name__ == '__main__':
    sys.exit(main())
