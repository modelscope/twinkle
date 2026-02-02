# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Task Queue Management for Tinker Server.

This module provides:
1. TaskStatus - Enum for tracking task lifecycle states
2. RateLimiter - Sliding window rate limiter supporting rps and tps
3. TaskQueueConfig - Configuration for rate limits and queue behavior
4. TaskQueueMixin - Mixin class for serial task execution with rate limiting
"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Optional, Tuple

import yaml

if TYPE_CHECKING:
    from twinkle.server.utils.state import ServerStateProxy


class TaskStatus(Enum):
    """Task lifecycle status."""
    PENDING = "pending"           # Task created, waiting to be processed
    QUEUED = "queued"             # Task in queue waiting for execution
    RUNNING = "running"           # Task currently executing
    COMPLETED = "completed"       # Task completed successfully
    FAILED = "failed"             # Task failed with error
    RATE_LIMITED = "rate_limited" # Task rejected due to rate limiting


@dataclass
class TaskQueueConfig:
    """Configuration for task queue and rate limiting.
    
    Attributes:
        rps_limit: Maximum requests per second per user token.
        tps_limit: Maximum input tokens per second per user token.
        window_seconds: Time window for rate limiting calculations.
        queue_timeout: Maximum time a task can wait in queue (seconds).
        enabled: Whether rate limiting is enabled.
    """
    rps_limit: float = 10.0           # 10 requests per second
    tps_limit: float = 10000.0        # 10000 input tokens per second
    window_seconds: float = 1.0       # 1 second sliding window
    queue_timeout: float = 300.0      # 5 minutes queue timeout
    enabled: bool = True              # Rate limiting enabled by default
    
    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]] = None) -> 'TaskQueueConfig':
        """Create TaskQueueConfig from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values. Supports keys:
                - rps_limit: requests per second limit
                - tps_limit: input tokens per second limit
                - window_seconds: sliding window duration
                - queue_timeout: queue timeout in seconds
                - enabled: whether rate limiting is enabled
                
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
            if 'enabled' in config_dict:
                config.enabled = bool(config_dict['enabled'])
        return config
    
    @classmethod
    def from_env_or_yaml(cls, yaml_path: Optional[str] = None) -> 'TaskQueueConfig':
        """Load configuration from environment variables or YAML file.
        
        Priority: Environment variables > YAML > Default values
        
        Environment variables:
            - TWINKLE_RPS_LIMIT: requests per second limit
            - TWINKLE_TPS_LIMIT: input tokens per second limit
            - TWINKLE_QUEUE_TIMEOUT: queue timeout in seconds
            - TWINKLE_RATE_LIMIT_ENABLED: enable/disable rate limiting (true/false)
            
        Args:
            yaml_path: Optional path to YAML configuration file.
            
        Returns:
            TaskQueueConfig instance with loaded configuration.
        """
        config = cls()
        
        # Load from YAML if provided
        if yaml_path and os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}
                    rate_limit_config = yaml_config.get('rate_limit', {})
                    if 'rps_limit' in rate_limit_config:
                        config.rps_limit = float(rate_limit_config['rps_limit'])
                    if 'tps_limit' in rate_limit_config:
                        config.tps_limit = float(rate_limit_config['tps_limit'])
                    if 'window_seconds' in rate_limit_config:
                        config.window_seconds = float(rate_limit_config['window_seconds'])
                    if 'queue_timeout' in rate_limit_config:
                        config.queue_timeout = float(rate_limit_config['queue_timeout'])
                    if 'enabled' in rate_limit_config:
                        config.enabled = bool(rate_limit_config['enabled'])
            except Exception:
                pass  # Use defaults if YAML parsing fails
        
        # Override with environment variables (highest priority)
        if os.environ.get('TWINKLE_RPS_LIMIT'):
            config.rps_limit = float(os.environ['TWINKLE_RPS_LIMIT'])
        if os.environ.get('TWINKLE_TPS_LIMIT'):
            config.tps_limit = float(os.environ['TWINKLE_TPS_LIMIT'])
        if os.environ.get('TWINKLE_QUEUE_TIMEOUT'):
            config.queue_timeout = float(os.environ['TWINKLE_QUEUE_TIMEOUT'])
        if os.environ.get('TWINKLE_RATE_LIMIT_ENABLED'):
            config.enabled = os.environ['TWINKLE_RATE_LIMIT_ENABLED'].lower() in ('true', '1', 'yes')
            
        return config


class RateLimiter:
    """Sliding window rate limiter supporting both rps and tps limits.
    
    This rate limiter tracks request history per user token and enforces
    both requests-per-second (rps) and tokens-per-second (tps) limits.
    
    Attributes:
        rps_limit: Maximum requests per second.
        tps_limit: Maximum input tokens per second.
        window_seconds: Time window for rate calculations.
    """
    
    def __init__(self, rps_limit: float, tps_limit: float, window_seconds: float = 1.0):
        """Initialize the rate limiter.
        
        Args:
            rps_limit: Maximum requests per second per user token.
            tps_limit: Maximum input tokens per second per user token.
            window_seconds: Time window for rate limiting (default 1.0s).
        """
        self.rps_limit = rps_limit
        self.tps_limit = tps_limit
        self.window_seconds = window_seconds
        # Dict mapping user token -> list of (timestamp, token_count) tuples
        self._token_requests: Dict[str, List[Tuple[float, int]]] = {}
        self._lock = asyncio.Lock()
    
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


class TaskQueueMixin:
    """Mixin providing task queue management, rate limiting, and status tracking.
    
    This mixin should be inherited by classes that need to:
    1. Execute async tasks serially through a queue
    2. Apply per-user rate limiting (rps and tps)
    3. Track task lifecycle status for proper client polling
    
    Requirements:
        - Inheriting class must have `self.state: ServerStateProxy` attribute
        - Call `_init_task_queue()` in `__init__` to initialize the queue
        - Call `await _start_worker()` to start the background worker
        
    Example:
        class MyService(TaskQueueMixin):
            def __init__(self):
                self.state = get_server_state()
                self._init_task_queue(TaskQueueConfig.from_env_or_yaml())
                
            async def my_endpoint(self, request, body):
                async def _do_work():
                    return await some_operation()
                return await self.schedule_task(
                    _do_work(),
                    model_id=body.model_id,
                    token=request.state.token,
                    input_tokens=len(body.tokens)
                )
    """
    
    # Type hint for state attribute that inheriting classes must provide
    state: 'ServerStateProxy'
    
    def _init_task_queue(self, config: Optional[TaskQueueConfig] = None) -> None:
        """Initialize the task queue system.
        
        Args:
            config: Optional TaskQueueConfig. If None, loads from env/yaml.
        """
        self._task_queue_config = config or TaskQueueConfig.from_env_or_yaml()
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._rate_limiter = RateLimiter(
            rps_limit=self._task_queue_config.rps_limit,
            tps_limit=self._task_queue_config.tps_limit,
            window_seconds=self._task_queue_config.window_seconds,
        )
        self._worker_task: Optional[asyncio.Task] = None
        self._worker_started = False
    
    async def _ensure_worker_started(self) -> None:
        """Ensure the background worker is running."""
        if not self._worker_started:
            self._worker_task = asyncio.create_task(self._queue_worker())
            self._worker_started = True
    
    async def _queue_worker(self) -> None:
        """Background worker that processes tasks from the queue serially.
        
        This worker runs indefinitely, pulling tasks from the queue and
        executing them one at a time. This ensures thread-safe execution
        of model operations that cannot be parallelized.
        """
        while True:
            try:
                # Wait for a task from the queue
                request_id, coro, model_id = await self._task_queue.get()
                
                try:
                    # Update status to RUNNING
                    self.state.store_future_status(
                        request_id, TaskStatus.RUNNING.value, model_id
                    )
                    
                    # Execute the task
                    result = await coro
                    
                    # Store completed result
                    self.state.store_future_status(
                        request_id, TaskStatus.COMPLETED.value, model_id, result=result
                    )
                except Exception:
                    # Store error result
                    error_payload = {
                        'error': traceback.format_exc(),
                        'category': 'Server'
                    }
                    self.state.store_future_status(
                        request_id, TaskStatus.FAILED.value, model_id, result=error_payload
                    )
                finally:
                    self._task_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception:
                # Log but don't crash the worker
                import logging
                logging.exception("Error in task queue worker")
                continue
    
    async def schedule_task(
        self,
        coro: Coroutine,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
        input_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Schedule an async task with rate limiting and status tracking.
        
        This method replaces the old `schedule_task` function with proper
        status tracking to fix the race condition where clients would receive
        404 instead of 408 when polling before task execution started.
        
        Key improvements:
        1. Register PENDING status BEFORE creating the task
        2. Apply rate limiting per user token
        3. Execute tasks serially through a queue
        
        Args:
            coro: The coroutine to execute.
            model_id: Optional model_id to associate with the result.
            token: Optional user token for rate limiting.
            input_tokens: Number of input tokens for tps rate limiting.
            
        Returns:
            Dict containing request_id and model_id for future retrieval.
        """
        request_id = f"req_{uuid.uuid4().hex}"
        
        # 1. Register PENDING status FIRST (fixes race condition)
        self.state.store_future_status(
            request_id, TaskStatus.PENDING.value, model_id
        )
        
        # 2. Check rate limiting if enabled and token provided
        if self._task_queue_config.enabled and token:
            allowed, reason = await self._rate_limiter.check_and_record(token, input_tokens)
            if not allowed:
                self.state.store_future_status(
                    request_id, TaskStatus.RATE_LIMITED.value, model_id, reason=reason
                )
                return {'request_id': request_id, 'model_id': model_id}
        
        # 3. Ensure worker is started
        await self._ensure_worker_started()
        
        # 4. Put task in queue and update status
        await self._task_queue.put((request_id, coro, model_id))
        self.state.store_future_status(
            request_id, TaskStatus.QUEUED.value, model_id
        )
        
        return {'request_id': request_id, 'model_id': model_id}
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics.
        
        Returns:
            Dict with queue size and worker status.
        """
        return {
            'queue_size': self._task_queue.qsize(),
            'worker_running': self._worker_started and self._worker_task is not None,
            'rate_limit_config': {
                'rps_limit': self._task_queue_config.rps_limit,
                'tps_limit': self._task_queue_config.tps_limit,
                'enabled': self._task_queue_config.enabled,
            }
        }
    
    def get_rate_limit_stats(self, token: str) -> Dict[str, Any]:
        """Get rate limiting stats for a specific user token.
        
        Args:
            token: User token to get stats for.
            
        Returns:
            Dict with current and available rate limits.
        """
        return self._rate_limiter.get_stats(token)
