# Copyright (c) Twinkle Contributors. All rights reserved.
"""Twinkle TUI - Terminal User Interface for training control."""

import logging
from pathlib import Path

from twinkle.utils.logger import get_logger

# ── Log file: ./tui.log (current working directory) ──
_LOG_FILE = Path.cwd() / 'tui.log'


def _configure_logging(verbose: bool = False) -> None:
    """Configure file-only logging for TUI.

    All logs are written to ./tui.log in the current working directory.
    NO console output — avoids corrupting Textual's alt-screen buffer.
    The file is rotated at 5MB with 3 backups.
    """
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8',
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))

    level = logging.DEBUG if verbose else logging.INFO

    # Get the 'twinkle' logger (same one returned by get_logger())
    twinkle_logger = logging.getLogger('twinkle')
    twinkle_logger.setLevel(level)
    twinkle_logger.propagate = False

    # Remove any existing handlers (especially StreamHandlers that print to terminal)
    twinkle_logger.handlers.clear()

    # Only attach the file handler — no terminal output
    twinkle_logger.addHandler(handler)

    # Mark as initialized so get_logger() won't re-add a StreamHandler
    from twinkle.utils.logger import init_loggers
    init_loggers[twinkle_logger.name] = True

    # Also capture warnings from third-party libs
    logging.captureWarnings(True)


def main(argv: list[str] | None = None) -> int:
    """Programmatic entry point for ``twinkle-tui``.

    Mirrors the pattern of ``twinkle.server.cli:main`` — runs the Typer app
    in standalone mode and converts SystemExit to a plain return code.
    """
    import sys
    import typer
    from twinkle.version import __version__

    app = typer.Typer(
        add_completion=False,
        no_args_is_help=False,
        help='Twinkle TUI — ML Training Control via Natural Language.',
    )

    def _version_callback(value: bool) -> None:
        if value:
            typer.echo(f'twinkle-tui {__version__}')
            raise typer.Exit()

    @app.command()
    def launch(
        run_id: str | None = typer.Option(
            None, '--run-id', '-r',
            envvar='TWINKLE_TUI_RUN_ID',
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
            envvar='TWINKLE_TUI_VERBOSE',
            help='Enable verbose (DEBUG) logging.',
        ),
        version: bool = typer.Option(
            False, '--version', '-V',
            callback=_version_callback, is_eager=True,
            help='Show version and exit.',
        ),
    ) -> None:
        """Launch the Twinkle TUI."""
        _configure_logging(verbose=verbose)
        logger = get_logger()
        logger.info(
            f'TUI starting — model={llm_model}, base_url={llm_base_url}, '
            f'run_id={run_id}, log_file={_LOG_FILE}'
        )

        from twinkle_client.tui.app import TwinkleTUI

        tui = TwinkleTUI(
            run_id=run_id,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )
        tui.run()

    try:
        app(args=argv, standalone_mode=True)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code) if not isinstance(code, int) else code
    return 0
