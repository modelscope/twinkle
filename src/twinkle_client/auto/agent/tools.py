# Copyright (c) Twinkle Contributors. All rights reserved.
"""Agent tool definitions for training control, search, and metrics."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from twinkle.utils.logger import get_logger
from twinkle_client.auto.connection import LocalConnection
from .search_tools import _SearchTools
from .server_tools import _ServerTools
from .tool_schemas import TOOL_SCHEMAS

logger = get_logger()


class ToolExecutor(_ServerTools, _SearchTools):
    """Executes agent tool calls against the local connection."""

    def __init__(self, connection: LocalConnection):
        self.connection = connection
        self.on_run_selected: Callable[[str], None] | None = None
        self._server_url: str | None = None  # Set after successful start_server

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name and return the result as a JSON string."""
        handler = getattr(self, f'_tool_{name}', None)
        if handler is None:
            logger.warning(f'Unknown tool called: {name}')
            return json.dumps({'error': f'Unknown tool: {name}'})
        try:
            result = await handler(**arguments)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f'Tool {name} raised exception: {type(e).__name__}: {e}', exc_info=True)
            return json.dumps({'error': f'{name} failed: {e}'})

    # ── Training lifecycle ──

    def _resolve_server_url(self) -> str:
        """Resolve server URL: instance state > env var > default."""
        return (self._server_url or os.environ.get('TWINKLE_SERVER_URL') or 'http://localhost:8000')

    async def _tool_list_training_runs(self) -> list[dict]:
        return self.connection.list_training_runs()

    async def _tool_get_training_status(self, run_id: str) -> dict:
        metrics = self.connection.get_metrics(run_id, last_n=10)
        meta = self.connection.get_meta(run_id) or {}
        state = meta.get('status', 'unknown')
        return {'run_id': run_id, 'state': state, 'model_id': meta.get('model_id'), 'recent_metrics': metrics}

    async def _tool_start_training(self, run_id: str, script_content: str, model_id: str = '') -> dict:
        # Pre-check: Twinkle Server must be reachable
        server_url = self._resolve_server_url()
        if not await self._check_server_health(server_url):
            return {
                'status':
                'error',
                'run_id':
                run_id,
                'error': (f'Twinkle Server is not reachable at {server_url}. '
                          'Call start_server first to launch Ray cluster and Twinkle Server.'),
            }
        result = self.connection.start_training(run_id, script_content, model_id)
        actual_run_id = result.get('run_id', run_id)
        if self.on_run_selected:
            self.on_run_selected(actual_run_id)
        return result

    async def _tool_select_run(self, run_id: str) -> dict:
        self.connection.current_run_id = run_id
        if self.on_run_selected:
            self.on_run_selected(run_id)
        return {'run_id': run_id, 'status': 'selected'}

    async def _tool_pause_training(self, run_id: str) -> dict:
        return self.connection.pause_training(run_id)

    async def _tool_resume_training(self, run_id: str) -> dict:
        return self.connection.resume_training(run_id)

    async def _tool_stop_training(self, run_id: str) -> dict:
        # Send SIGTERM for graceful shutdown (checkpoint saving via registered handler).
        # Server retains model state in GPU memory — use resume_training to continue.
        return self.connection.stop_training(run_id)

    async def _tool_update_script(self, run_id: str, script_content: str) -> dict:
        return self.connection.update_script(run_id, script_content)
