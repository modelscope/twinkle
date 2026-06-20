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
    {
        'type': 'function',
        'function': {
            'name': 'select_metrics',
            'description': 'Choose which metric keys to display on the chart. The chart shows at most 4 metrics at once. Use this when the user asks to see specific metrics (e.g. "show reward-related metrics"). Pass an empty keys array to list all available metrics without changing the selection.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'keys': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Metric key names to display. Match against available keys (supports partial: pick keys containing the keyword). Pass [] to query available keys only.',
                    },
                },
                'required': ['keys'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_cluster_info',
            'description': 'Get Ray cluster resource info: total GPUs, GPU types, available resources, and number of nodes. Call this before planning training to determine parallelism.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
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
        self.select_metrics_callback: Callable[[list[str]], dict] | None = None
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
        return await self._search_hub('datasets', query, limit)

    async def _tool_search_models(self, query: str, limit: int = 5) -> dict:
        """Search ModelScope for models."""
        return await self._search_hub('models', query, limit)

    async def _search_hub(self, resource_type: str, query: str, limit: int) -> dict:
        """Unified ModelScope Hub search for models or datasets."""

        def _search():
            if resource_type == 'datasets':
                return self._search_datasets_impl(query, limit)
            else:
                return self._search_models_impl(query, limit)

        try:
            items = await asyncio.get_event_loop().run_in_executor(None, _search)
            return {'query': query, 'results': items}
        except Exception as e:
            return {'error': f'{resource_type.title()} search failed: {e}'}

    @staticmethod
    def _search_datasets_impl(query: str, limit: int) -> list[dict]:
        """Search datasets via ModelScope SDK (new API)."""
        from modelscope.hub.api import HubApi
        api = HubApi()
        result = api.list_datasets('', search=query, page_size=limit)
        datasets = result.get('datasets', [])
        return [
            {'id': d.get('id', ''), 'name': d.get('display_name', d.get('id', ''))}
            for d in datasets
        ]

    @staticmethod
    def _search_models_impl(query: str, limit: int) -> list[dict]:
        """Search models via ModelScope HTTP API (SDK doesn't support search)."""
        import requests
        resp = requests.put(
            'https://modelscope.cn/api/v1/models/',
            json={'Name': query, 'PageSize': limit, 'PageNumber': 1},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get('Success'):
            raise RuntimeError(data.get('Message', 'Unknown error'))
        models = data.get('Data', {}).get('Models', [])
        return [
            {
                'id': f"{m.get('Path', '')}/{m.get('Name', '')}",
                'name': m.get('ChineseName') or m.get('Name', ''),
            }
            for m in models
        ]

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

    async def _tool_select_metrics(self, keys: list[str]) -> dict:
        """Select which metrics to display on the chart."""
        if self.select_metrics_callback:
            return self.select_metrics_callback(keys)
        return {'error': 'Metrics panel not available'}

    # ── Cluster info ──

    async def _tool_get_cluster_info(self) -> dict:
        """Query Ray cluster for available resources."""

        def _query():
            import ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            resources = ray.cluster_resources()
            available = ray.available_resources()
            nodes = ray.nodes()
            gpu_total = resources.get('GPU', 0)
            gpu_available = available.get('GPU', 0)
            # Detect GPU types from node resources
            gpu_types = set()
            for node in nodes:
                for key in node.get('Resources', {}):
                    if key.startswith('accelerator_type:'):
                        gpu_types.add(key.split(':', 1)[1])
            return {
                'num_nodes': len([n for n in nodes if n.get('Alive')]),
                'gpu_total': int(gpu_total),
                'gpu_available': int(gpu_available),
                'gpu_types': sorted(gpu_types) if gpu_types else ['unknown'],
                'cpu_total': resources.get('CPU', 0),
                'memory_bytes': resources.get('memory', 0),
            }

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _query)
        except ImportError:
            return {'error': 'ray not installed. Run: pip install ray'}
        except Exception as e:
            return {'error': f'Failed to query Ray cluster: {e}'}
