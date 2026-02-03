# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Rate Limiter for Tinker Server.

This module provides a sliding window rate limiter that supports both
requests-per-second (rps) and tokens-per-second (tps) limits with automatic
memory cleanup to prevent unbounded memory growth.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from twinkle.utils.logger import get_logger

logger = get_logger()


class RateLimiter:
    """Sliding window rate limiter supporting both rps and tps limits.

    This rate limiter tracks request history per user token and enforces
    both requests-per-second (rps) and tokens-per-second (tps) limits.

    To prevent unbounded memory growth, inactive tokens are automatically
    removed after token_cleanup_multiplier * window_seconds of inactivity.

    Attributes:
        rps_limit: Maximum requests per second.
        tps_limit: Maximum input tokens per second.
        window_seconds: Time window for rate calculations.
        token_cleanup_multiplier: Multiplier for token cleanup threshold.
        token_cleanup_interval: How often to run cleanup task (seconds).
    """

    def __init__(
        self,
        rps_limit: float,
        tps_limit: float,
        window_seconds: float = 1.0,
        token_cleanup_multiplier: float = 10.0,
        token_cleanup_interval: float = 60.0,
    ):
        """Initialize the rate limiter.

        Args:
            rps_limit: Maximum requests per second per user token.
            tps_limit: Maximum input tokens per second per user token.
            window_seconds: Time window for rate limiting (default 1.0s).
            token_cleanup_multiplier: Multiplier for token cleanup threshold.
                Tokens inactive for window_seconds * token_cleanup_multiplier
                will be removed. Default is 10.0 (10x the window).
            token_cleanup_interval: How often to run the cleanup task in seconds.
                Default is 60.0 (every minute).
        """
        self.rps_limit = rps_limit
        self.tps_limit = tps_limit
        self.window_seconds = window_seconds
        self.token_cleanup_multiplier = token_cleanup_multiplier
        self.token_cleanup_interval = token_cleanup_interval

        # Dict mapping user token -> list of (timestamp, token_count) tuples
        self._token_requests: Dict[str, List[Tuple[float, int]]] = {}
        # Track last activity time for each token
        self._last_activity: Dict[str, float] = {}

        # Async lock for rate limiting operations
        self._lock = asyncio.Lock()

        # Cleanup tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_started = False

    def _cleanup_old_requests(self, token: str, current_time: float) -> None:
        """Remove requests outside the sliding window.

        Args:
            token: User token to clean up.
            current_time: Current timestamp.
        """
        if token not in self._token_requests:
            return
        cutoff_time = current_time - self.window_seconds
        self._token_requests[token] = [
            (ts, count) for ts, count in self._token_requests[token]
            if ts > cutoff_time
        ]

        # Remove token completely if it has no requests in the current window
        if not self._token_requests[token]:
            del self._token_requests[token]
            if token in self._last_activity:
                del self._last_activity[token]

    async def _cleanup_inactive_tokens(self) -> None:
        """Background task that periodically removes inactive tokens.

        This prevents unbounded memory growth by removing tokens that haven't
        been active for token_cleanup_multiplier * window_seconds.
        """
        logger.debug(
            f"[RateLimiter] Cleanup task started (interval={self.token_cleanup_interval}s)")
        while True:
            try:
                await asyncio.sleep(self.token_cleanup_interval)

                async with self._lock:
                    current_time = time.time()
                    inactive_threshold = current_time - \
                        (self.window_seconds * self.token_cleanup_multiplier)

                    # Find tokens that haven't been active recently
                    tokens_to_remove = [
                        token for token, last_time in self._last_activity.items()
                        if last_time < inactive_threshold
                    ]

                    # Remove inactive tokens
                    for token in tokens_to_remove:
                        if token in self._token_requests:
                            del self._token_requests[token]
                        if token in self._last_activity:
                            del self._last_activity[token]

                    if tokens_to_remove:
                        logger.debug(
                            f"[RateLimiter] Cleaned up {len(tokens_to_remove)} inactive tokens. "
                            f"Active tokens remaining: {len(self._token_requests)}"
                        )

            except asyncio.CancelledError:
                logger.debug("[RateLimiter] Cleanup task cancelled")
                break
            except Exception as e:
                logger.warning(f"[RateLimiter] Error in cleanup task: {e}")
                continue

    def start_cleanup_task(self) -> None:
        """Start the background cleanup task.

        This should be called once when the rate limiter is initialized.
        It's safe to call multiple times - subsequent calls are ignored.
        """
        if not self._cleanup_started:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_inactive_tokens())
            self._cleanup_started = True
            logger.debug("[RateLimiter] Background cleanup task started")

    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task.

        This should be called when shutting down the server.
        """
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.debug("[RateLimiter] Background cleanup task stopped")

    async def check_and_record(
        self, token: str, input_tokens: int
    ) -> Tuple[bool, Optional[str]]:
        """Check if request is allowed and record it if so.

        Args:
            token: User token for rate limiting.
            input_tokens: Number of input tokens in this request.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str]).
            If allowed is False, reason contains the rate limit explanation.
        """
        async with self._lock:
            current_time = time.time()

            # Clean up old requests
            self._cleanup_old_requests(token, current_time)

            # Initialize if needed
            if token not in self._token_requests:
                self._token_requests[token] = []

            # Update last activity time
            self._last_activity[token] = current_time

            requests = self._token_requests[token]

            # Count current window stats
            request_count = len(requests)
            token_count = sum(count for _, count in requests)

            # Check rps limit
            if request_count >= self.rps_limit:
                return False, f"RPS limit exceeded: {request_count}/{self.rps_limit} requests/s"

            # Check tps limit
            if token_count + input_tokens > self.tps_limit:
                return False, f"TPS limit exceeded: {token_count + input_tokens}/{self.tps_limit} tokens/s"

            # Record this request
            self._token_requests[token].append((current_time, input_tokens))
            return True, None

    def get_stats(self, token: str) -> Dict[str, Any]:
        """Get current rate limiting stats for a token.

        Args:
            token: User token to get stats for.

        Returns:
            Dict with current rps, tps, and limits.
        """
        current_time = time.time()
        self._cleanup_old_requests(token, current_time)

        # Update last activity time even for stats queries
        if token in self._token_requests:
            self._last_activity[token] = current_time

        requests = self._token_requests.get(token, [])
        request_count = len(requests)
        token_count = sum(count for _, count in requests)

        return {
            'current_rps': request_count,
            'current_tps': token_count,
            'rps_limit': self.rps_limit,
            'tps_limit': self.tps_limit,
            'rps_available': self.rps_limit - request_count,
            'tps_available': self.tps_limit - token_count,
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for monitoring.

        Returns:
            Dict with active token count and cleanup configuration.
        """
        return {
            'active_tokens': len(self._token_requests),
            'tracked_tokens': len(self._last_activity),
            'cleanup_threshold_seconds': self.window_seconds * self.token_cleanup_multiplier,
            'cleanup_interval_seconds': self.token_cleanup_interval,
            'cleanup_task_running': self._cleanup_started and self._cleanup_task and not self._cleanup_task.done(),
        }
