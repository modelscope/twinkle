# Copyright (c) Twinkle Contributors. All rights reserved.
"""Agent tool definitions for training control, search, and metrics."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable

from twinkle_client.tui.connection import LocalConnection

# ──────────────────────────────────────────────────────────────────────────────
# Tool schemas (OpenAI function calling format)
# ──────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name': 'list_training_runs',
            'description': 'List all active and historical training runs.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_training_status',
            'description': 'Get detailed status and recent metrics for a training run.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID.'},
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'start_training',
            'description': 'Create a new training run: write the script, launch it, and start monitoring. Call this after generating a complete training script.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Unique run ID (e.g., "grpo-gsm8k").'},
                    'script_content': {'type': 'string', 'description': 'Full Python source code of the training script.'},
                    'model_id': {'type': 'string', 'description': 'Model identifier for metadata (e.g., "Qwen/Qwen3.5-4B").'},
                },
                'required': ['run_id', 'script_content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'select_run',
            'description': 'Switch the TUI to monitor a different training run. Updates metrics panel and status bar.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID to monitor.'},
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'pause_training',
            'description': 'Pause training by killing the client process (SIGKILL). Server retains all state — call resume_training to continue.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID to pause.'},
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'resume_training',
            'description': 'Resume a paused training run by re-launching the client script. Server state is preserved.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID to resume.'},
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'stop_training',
            'description': 'Gracefully stop training via SIGTERM. The script saves a checkpoint (model + dataloader state) before exiting. Can be resumed later from checkpoint.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID to stop.'},
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'update_script',
            'description': 'Update the training script for a run. Archives the current train.py as train_v{N}.py and writes the new version. Use after diagnosing a script error, then call resume_training.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID.'},
                    'script_content': {'type': 'string', 'description': 'Full Python source code of the new training script.'},
                },
                'required': ['run_id', 'script_content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'list_supported_models',
            'description': 'Query the Twinkle server for its list of supported base models. Always call this before writing a training script to verify model availability.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'base_url': {
                        'type': 'string',
                        'description': 'Server base URL. Default: http://localhost:8000. Cloud: http://www.modelscope.cn/twinkle',
                    },
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_datasets',
            'description': 'Search ModelScope Hub for datasets matching a query.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query for datasets.'},
                    'limit': {'type': 'integer', 'description': 'Max results (default 5).'},
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_models',
            'description': 'Search ModelScope Hub for models matching a query.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query for models.'},
                    'limit': {'type': 'integer', 'description': 'Max results (default 5).'},
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'zoom_metrics',
            'description': 'Adjust the metrics chart view. Zoom into specific step ranges or reset to show all.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'action': {
                        'type': 'string',
                        'enum': ['zoom', 'reset'],
                        'description': '"zoom" to set range, "reset" to show all.',
                    },
                    'x_start': {'type': 'integer', 'description': 'Start step for x-axis.'},
                    'x_end': {'type': 'integer', 'description': 'End step for x-axis.'},
                    'y_min': {'type': 'number', 'description': 'Min value for y-axis.'},
                    'y_max': {'type': 'number', 'description': 'Max value for y-axis.'},
                },
                'required': ['action'],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Tool executor
# ──────────────────────────────────────────────────────────────────────────────


class ToolExecutor:
    """Executes agent tool calls against the local connection."""

    def __init__(self, connection: LocalConnection):
        self.connection = connection
        self.metrics_callback: Callable | None = None
        self.on_run_selected: Callable[[str], None] | None = None

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name and return the result as a JSON string."""
        handler = getattr(self, f'_tool_{name}', None)
        if handler is None:
            return json.dumps({'error': f'Unknown tool: {name}'})
        try:
            result = await handler(**arguments)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({'error': f'{name} failed: {e}'})

    # ── Training lifecycle ──

    async def _tool_list_training_runs(self) -> list[dict]:
        return self.connection.list_training_runs()

    async def _tool_get_training_status(self, run_id: str) -> dict:
        metrics = self.connection.get_metrics(run_id, last_n=10)
        meta = self.connection.get_meta(run_id) or {}
        state = meta.get('status', 'unknown')
        return {'run_id': run_id, 'state': state, 'model_id': meta.get('model_id'), 'recent_metrics': metrics}

    async def _tool_start_training(self, run_id: str, script_content: str, model_id: str = '') -> dict:
        result = self.connection.start_training(run_id, script_content, model_id)
        if self.on_run_selected:
            self.on_run_selected(run_id)
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
        return self.connection.stop_training(run_id)

    async def _tool_update_script(self, run_id: str, script_content: str) -> dict:
        return self.connection.update_script(run_id, script_content)

    # ── Server queries ──

    async def _tool_list_supported_models(self, base_url: str | None = None) -> dict:
        """Query the Twinkle server for supported models."""
        url = base_url or os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000')
        key = os.environ.get('MODELSCOPE_TOKEN') or os.environ.get('TWINKLE_SERVER_TOKEN') or 'EMPTY_API_KEY'

        def _query():
            from twinkle_client import init_twinkle_client
            client = init_twinkle_client(base_url=url, api_key=key)
            caps = client.get_server_capabilities()
            return {
                'base_url': url,
                'supported_models': [m.model_name for m in caps.supported_models],
            }

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _query)
        except ImportError:
            return {'error': 'twinkle_client not installed. Run: pip install twinkle-kit'}
        except Exception as e:
            return {'error': f'Failed to query {url}: {e}'}

    async def _tool_search_datasets(self, query: str, limit: int = 5) -> dict:
        """Search ModelScope for datasets."""

        def _search():
            from modelscope.hub.api import HubApi
            api = HubApi()
            results = api.list_datasets(dataset_name=query, limit=limit)
            return [
                {'id': getattr(d, 'dataset_id', getattr(d, 'id', str(d))),
                 'name': getattr(d, 'dataset_name', getattr(d, 'name', str(d)))}
                for d in results
            ]

        try:
            items = await asyncio.get_event_loop().run_in_executor(None, _search)
            return {'query': query, 'results': items}
        except Exception as e:
            return {'error': f'Dataset search failed: {e}'}

    async def _tool_search_models(self, query: str, limit: int = 5) -> dict:
        """Search ModelScope for models."""

        def _search():
            from modelscope.hub.api import HubApi
            api = HubApi()
            results = api.list_models(model_name=query, limit=limit)
            return [
                {'id': getattr(m, 'model_id', getattr(m, 'id', str(m))),
                 'name': getattr(m, 'model_name', getattr(m, 'name', str(m)))}
                for m in results
            ]

        try:
            items = await asyncio.get_event_loop().run_in_executor(None, _search)
            return {'query': query, 'results': items}
        except Exception as e:
            return {'error': f'Model search failed: {e}'}

    # ── Metrics chart ──

    async def _tool_zoom_metrics(
        self,
        action: str,
        x_start: int | None = None,
        x_end: int | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> dict:
        if self.metrics_callback:
            if action == 'reset':
                self.metrics_callback('reset')
            else:
                self.metrics_callback('zoom', x_start=x_start, x_end=x_end, y_min=y_min, y_max=y_max)
        return {'action': action, 'status': 'applied'}
