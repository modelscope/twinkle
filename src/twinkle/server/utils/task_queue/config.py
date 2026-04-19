# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Task queue configuration.

Provides TaskQueueConfig for controlling rate limits, timeouts,
and queue behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TaskQueueConfig:
    """Configuration for task queue and rate limiting.

    Attributes:
        rps_limit: Maximum requests per second per user token.
        tps_limit: Maximum input tokens per second per user token.
        window_seconds: Time window for rate limiting calculations.
        queue_timeout: Maximum time a task can wait in queue (seconds).
        execution_timeout: Maximum time a task can execute (seconds). 0 means no limit.
        enabled: Whether rate limiting is enabled.
        token_cleanup_multiplier: Multiplier for token cleanup threshold.
        token_cleanup_interval: How often to run cleanup task (seconds).
        max_input_tokens: Maximum allowed input tokens per request (default 16000).
    """
    rps_limit: float = 100.0  # 100 requests per second
    tps_limit: float = 16000.0  # 16000 input tokens per second
    window_seconds: float = 1.0  # 1 second sliding window
    queue_timeout: float = 300.0  # 5 minutes queue timeout
    execution_timeout: float = 120.0  # 120 seconds execution timeout (0 to disable)
    enabled: bool = True  # Rate limiting enabled by default
    # Remove tokens after 10x window inactivity
    token_cleanup_multiplier: float = 10.0
    token_cleanup_interval: float = 60.0  # Run cleanup every 60 seconds
    max_input_tokens: int = 16000  # Maximum input tokens per request

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any] | None = None) -> TaskQueueConfig:
        """Create TaskQueueConfig from a dictionary.

        Args:
            config_dict: Dictionary with configuration values. Supports keys:
                - rps_limit: requests per second limit
                - tps_limit: input tokens per second limit
                - window_seconds: sliding window duration
                - queue_timeout: queue timeout in seconds
                - execution_timeout: task execution timeout in seconds (0 to disable)
                - enabled: whether rate limiting is enabled
                - token_cleanup_multiplier: multiplier for token cleanup threshold
                - token_cleanup_interval: cleanup task interval in seconds
                - max_input_tokens: maximum input tokens per request

        Returns:
            TaskQueueConfig instance with values from dict merged with defaults.
        """
        config = cls()
        if config_dict:
            if 'rps_limit' in config_dict:
                config.rps_limit = float(config_dict['rps_limit'])
            if 'tps_limit' in config_dict:
                config.tps_limit = float(config_dict['tps_limit'])
            if 'window_seconds' in config_dict:
                config.window_seconds = float(config_dict['window_seconds'])
            if 'queue_timeout' in config_dict:
                config.queue_timeout = float(config_dict['queue_timeout'])
            if 'execution_timeout' in config_dict:
                config.execution_timeout = float(config_dict['execution_timeout'])
            if 'enabled' in config_dict:
                config.enabled = bool(config_dict['enabled'])
            if 'token_cleanup_multiplier' in config_dict:
                config.token_cleanup_multiplier = float(config_dict['token_cleanup_multiplier'])
            if 'token_cleanup_interval' in config_dict:
                config.token_cleanup_interval = float(config_dict['token_cleanup_interval'])
            if 'max_input_tokens' in config_dict:
                config.max_input_tokens = int(config_dict['max_input_tokens'])
        return config
