# Copyright (c) ModelScope Contributors. All rights reserved.
"""
CLI entry point for Twinkle Server.

Usage:
    # From config file
    python -m twinkle.server --config server_config.yaml

    # With server type override
    python -m twinkle.server --config server_config.yaml --server-type tinker

    # Quick start with minimal args
    python -m twinkle.server --server-type tinker --port 8000 --model-id "Qwen/Qwen2.5-7B-Instruct"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from twinkle import get_logger

logger = get_logger()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m twinkle.server",
        description="Twinkle Server Launcher - Unified launcher for tinker and twinkle servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server from YAML config file
  python -m twinkle.server --config server_config.yaml

  # Start tinker server with specific config
  python -m twinkle.server -c config.yaml -t tinker

  # Run in background (daemon mode)
  python -m twinkle.server -c config.yaml --no-wait
        """,
    )

    # Config file option
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to YAML configuration file (required)",
    )

    # Server type
    parser.add_argument(
        "-t", "--server-type",
        type=str,
        default="twinkle",
        choices=["tinker", "twinkle"],
        metavar="TYPE",
        help="Server type: 'tinker' or 'twinkle' (default: twinkle)",
    )

    # Ray options
    parser.add_argument(
        "--namespace",
        type=str,
        metavar="NS",
        help="Ray namespace (default: 'twinkle_cluster' for tinker, None for twinkle)",
    )

    # Runtime options
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't block waiting for Enter (daemon mode)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging level (default: INFO)",
    )

    return parser


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    try:
        from twinkle.server.launcher import launch_server

        # Config file mode
        config_path = Path(parsed_args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1

        launch_server(
            config_path=config_path,
            server_type=parsed_args.server_type,
            ray_namespace=parsed_args.namespace,
            wait=not parsed_args.no_wait,
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required dependencies are installed")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
