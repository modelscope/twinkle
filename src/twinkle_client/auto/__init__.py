# Copyright (c) Twinkle Contributors. All rights reserved.
"""Twinkle Auto - Minimal chat interface for ML training control."""

import logging
from pathlib import Path

from twinkle.utils.logger import get_logger

# ── Log file: ./auto.log (current working directory) ──
_LOG_FILE = Path.cwd() / 'auto.log'


def _configure_logging(verbose: bool = False) -> None:
    """Configure file-only logging for auto.

    All logs go to ./auto.log, not stdout (avoids mixing with chat output).
    """
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8',
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))

    level = logging.DEBUG if verbose else logging.INFO
    twinkle_logger = logging.getLogger('twinkle')
    twinkle_logger.setLevel(level)
    twinkle_logger.propagate = False
    twinkle_logger.handlers.clear()
    twinkle_logger.addHandler(handler)

    from twinkle.utils.logger import init_loggers
    init_loggers[twinkle_logger.name] = True
    logging.captureWarnings(True)


def main(argv: list[str] | None = None) -> int:
    """Programmatic entry point for ``twinkle-auto``.

    Mirrors the pattern of ``twinkle.server.cli:main`` — runs the Typer app
    in standalone mode and converts SystemExit to a plain return code.
    """
    import sys
    import typer
    from twinkle.version import __version__

    app = typer.Typer(
        add_completion=False,
        no_args_is_help=False,
        help='Twinkle Auto — ML Training Control via Natural Language.',
    )

    def _version_callback(value: bool) -> None:
        if value:
            typer.echo(f'twinkle-auto {__version__}')
            raise typer.Exit()

    @app.command()
    def launch(
        run_id: str | None = typer.Option(
            None, '--run-id', '-r',
            envvar='TWINKLE_AUTO_RUN_ID',
            help='Attach to an existing training run by ID.',
        ),
        llm_base_url: str = typer.Option(
            'http://localhost:11434/v1', '--llm-base-url',
            envvar='TWINKLE_LLM_BASE_URL',
            help='LLM API base URL.',
        ),
        llm_model: str = typer.Option(
            'qwen3.5', '--llm-model',
            envvar='TWINKLE_LLM_MODEL',
            help='LLM model name.',
        ),
        llm_api_key: str = typer.Option(
            'not-needed', '--llm-api-key',
            envvar='TWINKLE_LLM_API_KEY',
            help='LLM API key.',
        ),
        verbose: bool = typer.Option(
            False, '--verbose', '-v',
            envvar='TWINKLE_AUTO_VERBOSE',
            help='Enable verbose (DEBUG) logging.',
        ),
        version: bool = typer.Option(
            False, '--version', '-V',
            callback=_version_callback, is_eager=True,
            help='Show version and exit.',
        ),
    ) -> None:
        """Launch Twinkle Auto."""
        _configure_logging(verbose=verbose)
        logger = get_logger()
        logger.info(
            f'Auto starting — model={llm_model}, base_url={llm_base_url}, '
            f'run_id={run_id}, log_file={_LOG_FILE}'
        )

        from twinkle_client.auto.app import TwinkleAuto

        auto = TwinkleAuto(
            run_id=run_id,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )
        auto.run()

    try:
        app(args=argv, standalone_mode=True)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code) if not isinstance(code, int) else code
    return 0
