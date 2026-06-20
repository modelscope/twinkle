# Copyright (c) Twinkle Contributors. All rights reserved.
"""Twinkle TUI - Terminal User Interface for training control."""


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
        version: bool = typer.Option(
            False, '--version', '-V',
            callback=_version_callback, is_eager=True,
            help='Show version and exit.',
        ),
    ) -> None:
        """Launch the Twinkle TUI."""
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
