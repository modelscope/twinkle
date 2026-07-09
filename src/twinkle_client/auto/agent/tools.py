# Copyright (c) Twinkle Contributors. All rights reserved.
"""Agent tool definitions for training control, search, and metrics."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable

from twinkle.utils.logger import get_logger
from twinkle_client.auto.connection import LocalConnection

logger = get_logger()

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
            'name': 'start_server',
            'description': (
                'Start Ray cluster and Twinkle Server. MUST be called before start_training. '
                'Idempotent: skips if server is already reachable. '
                'Supports multi-model deployments: one training model + N sampler/teacher models. '
                'Automatically generates server_config.yaml from parameters.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'model_id': {
                        'type': 'string',
                        'description': 'Student/training model ID (e.g. "Qwen/Qwen3.5-4B").',
                    },
                    'train_gpus': {
                        'type': 'integer',
                        'description': 'GPUs for the training model. Default: auto-detect remaining GPUs.',
                    },
                    'backend': {
                        'type': 'string',
                        'enum': ['transformers', 'megatron'],
                        'description': 'Training model backend. Default: transformers.',
                    },
                    'samplers': {
                        'type': 'array',
                        'description': (
                            'List of sampler/teacher models for RL/OPD. Each entry deploys '
                            'an inference service (vLLM or torch). Omit for simple SFT.'
                        ),
                        'items': {
                            'type': 'object',
                            'properties': {
                                'model_id': {
                                    'type': 'string',
                                    'description': 'Teacher/reference model ID (e.g. "Qwen/Qwen3.5-72B").',
                                },
                                'gpus': {
                                    'type': 'integer',
                                    'description': 'Total number of GPUs for this sampler. Default: 1. Must equal tp * dp.',
                                },
                                'tp': {
                                    'type': 'integer',
                                    'description': (
                                        'Tensor parallelism size (GPUs per vLLM worker process). '
                                        'Use tp>1 for large models that do not fit on a single GPU. Default: 1.'
                                    ),
                                },
                                'dp': {
                                    'type': 'integer',
                                    'description': (
                                        'Data parallelism size (number of independent inference replicas). '
                                        'If not specified, computed as gpus // tp. Default: 1.'
                                    ),
                                },
                                'engine': {
                                    'type': 'string',
                                    'enum': ['vllm', 'torch'],
                                    'description': 'Inference engine. Default: vllm.',
                                },
                                'max_model_len': {
                                    'type': 'integer',
                                    'description': 'Max sequence length for inference. Default: 16000.',
                                },
                            },
                            'required': ['model_id'],
                        },
                    },
                    'port': {
                        'type': 'integer',
                        'description': 'HTTP port for server. Default: 8000.',
                    },
                },
                'required': ['model_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'shutdown_server',
            'description': (
                'Shut down Twinkle Server and Ray cluster. WARNING: This releases all GPU resources '
                'and DESTROYS model state held in server memory. Only call when training is truly '
                'finished and you no longer need the server. Model weights/optimizer state in GPU '
                'will be LOST unless a checkpoint was explicitly saved.'
            ),
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'start_training',
            'description': (
                'Create a new training run: write the client script, launch it, and start monitoring. '
                'REQUIRES: Twinkle Server must be running (call start_server first). '
                'The client script connects to the server — server holds model state in GPU memory. '
                'Kill client = pause (state preserved). Re-launch client = resume.'
            ),
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
            'description': 'Switch to monitor a different training run. Updates connection context.',
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
            'description': (
                'Gracefully stop the training client (SIGTERM). The script saves a checkpoint '
                'before exiting. Server retains model/optimizer state in GPU memory — '
                'use resume_training to continue. Similar to pause_training but with checkpoint save. '
                'To fully release GPU resources, use shutdown_server.'
            ),
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
            'name': 'get_cluster_info',
            'description': (
                'Get cluster GPU resource info for planning training parallelism. '
                'First attempts to query a running Ray cluster; if Ray is not available, '
                'falls back to nvidia-smi for local GPU discovery. '
                'The result indicates whether Ray is active — if not, the training script '
                'should either start a local Ray cluster itself or the user should launch '
                'Ray manually (see server mode run.sh).'
            ),
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
        return (
            self._server_url
            or os.environ.get('TWINKLE_SERVER_URL')
            or 'http://localhost:8000'
        )

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
                'status': 'error',
                'run_id': run_id,
                'error': (
                    f'Twinkle Server is not reachable at {server_url}. '
                    'Call start_server first to launch Ray cluster and Twinkle Server.'
                ),
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

    # ── Server lifecycle ──

    async def _check_server_health(self, url: str) -> bool:
        """Check if Twinkle Server is reachable (non-blocking)."""
        import urllib.request
        import urllib.error

        def _probe():
            try:
                req = urllib.request.Request(f'{url}/api/v1/healthz', method='GET')
                urllib.request.urlopen(req, timeout=3)
                return True
            except (urllib.error.URLError, OSError):
                # Try a simpler connectivity check
                try:
                    urllib.request.urlopen(url, timeout=3)
                    return True
                except (urllib.error.URLError, OSError):
                    return False

        return await asyncio.get_event_loop().run_in_executor(None, _probe)

    # ── Server startup pipeline ──

    async def _tool_start_server(
        self,
        model_id: str,
        train_gpus: int | None = None,
        port: int = 8000,
        backend: str = 'transformers',
        samplers: list[dict] | None = None,
    ) -> dict:
        """Start Ray cluster + Twinkle Server. Idempotent. Supports multi-model."""
        server_url = self._server_url or os.environ.get('TWINKLE_SERVER_URL') or f'http://localhost:{port}'

        # Idempotent: skip if already running
        if await self._check_server_health(server_url):
            self._server_url = server_url
            return {'status': 'already_running', 'server_url': server_url}

        def _start():
            sampler_list = samplers or []

            # Step 1: Detect hardware & compute GPU partition
            total_hw_gpus = self._detect_gpu_count()
            if total_hw_gpus == 0:
                return {'status': 'error', 'error': 'No GPUs detected. Cannot start training server.'}

            alloc = self._compute_gpu_allocation(sampler_list, train_gpus, total_hw_gpus)
            if 'error' in alloc:
                return {'status': 'error', 'error': alloc['error']}
            t_gpus, sampler_gpu_total = alloc['train_gpus'], alloc['sampler_gpus']

            # Step 2: Generate server_config.yaml
            config_path = self._generate_server_config(
                model_id=model_id, train_gpus=t_gpus,
                port=port, backend=backend, samplers=sampler_list,
            )

            # Step 3: Start Ray cluster (multi-node GPU partitioning)
            ray_err = self._start_ray_cluster(t_gpus, sampler_gpu_total)
            if ray_err:
                return {'status': 'error', 'error': ray_err}

            # Step 4: Launch Twinkle Server process
            proc, log_path, err = self._launch_server_process(config_path)
            if err:
                return {'status': 'error', 'error': err}

            # Step 5: Wait for readiness (healthz + sampler engine)
            return self._wait_server_ready(
                server_url=server_url, proc=proc, log_path=log_path,
                sampler_list=sampler_list, model_id=model_id,
                t_gpus=t_gpus, backend=backend, config_path=config_path,
            )

        result = await asyncio.get_event_loop().run_in_executor(None, _start)
        if result.get('status') in ('started', 'already_running'):
            self._server_url = server_url
        return result

    # ── Server startup helpers ──

    @staticmethod
    def _detect_gpu_count() -> int:
        """Detect total hardware GPU count via nvidia-smi."""
        import subprocess as _sp
        try:
            r = _sp.run(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                return len([ln for ln in r.stdout.strip().split('\n') if ln.strip()])
        except (FileNotFoundError, OSError):
            pass
        return 0

    @staticmethod
    def _compute_gpu_allocation(
        sampler_list: list[dict],
        train_gpus: int | None,
        total_hw_gpus: int,
    ) -> dict:
        """Compute GPU partition: {train_gpus, sampler_gpus} or {error}."""
        sampler_gpu_total = 0
        for s in sampler_list:
            s_tp = s.get('tp', 1)
            s_dp, s_gpus = s.get('dp'), s.get('gpus')
            if s_gpus is not None:
                sampler_gpu_total += s_gpus
            elif s_dp is not None:
                sampler_gpu_total += s_tp * s_dp
            else:
                sampler_gpu_total += s_tp  # default dp=1

        t_gpus = train_gpus if train_gpus is not None else max(1, total_hw_gpus - sampler_gpu_total)
        needed = t_gpus + sampler_gpu_total
        if needed > total_hw_gpus:
            return {
                'error': (
                    f'Requested {needed} GPUs (train={t_gpus}, samplers={sampler_gpu_total}) '
                    f'but only {total_hw_gpus} available.'
                ),
            }
        return {'train_gpus': t_gpus, 'sampler_gpus': sampler_gpu_total}

    @staticmethod
    def _start_ray_cluster(train_gpus: int, sampler_gpus: int) -> str | None:
        """Start Ray multi-node cluster with GPU partitioning.

        Each role gets its own Ray node with dedicated CUDA_VISIBLE_DEVICES
        so GPUs are indexed from 0 within each node. This prevents the
        GPU ID mapping issues that occur with a single-node setup.

        On a single machine, multiple raylets need separate --temp-dir to
        avoid being detected as "already running".

        Returns an error message on failure, or None on success.
        """
        import subprocess as _sp
        import tempfile
        from pathlib import Path

        _sp.run(['ray', 'stop', '--force'], capture_output=True, timeout=15)

        # Create unique temp dirs so each `ray start` spawns a separate raylet
        ray_base = Path(tempfile.gettempdir()) / 'twinkle_ray'
        ray_base.mkdir(parents=True, exist_ok=True)

        def _ray_node(
            devices: str, num_gpus: int, *,
            head: bool = False, node_name: str = 'worker',
        ) -> str | None:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = devices
            temp_dir = str(ray_base / node_name)
            cmd = ['ray', 'start', f'--temp-dir={temp_dir}']
            if head:
                cmd += ['--head', '--port=6379', '--disable-usage-stats', '--include-dashboard=false']
            else:
                cmd += ['--address=127.0.0.1:6379']
            cmd.append(f'--num-gpus={num_gpus}')
            r = _sp.run(cmd, capture_output=True, text=True, timeout=30, env=env)
            if r.returncode != 0 and 'already' not in r.stderr.lower():
                return r.stderr.strip()
            return None

        # Head node — training model GPUs
        model_devices = ','.join(str(i) for i in range(train_gpus))
        err = _ray_node(model_devices, train_gpus, head=True, node_name='head')
        if err:
            return f'Ray head start failed: {err}'

        # GPU Worker node — sampler GPUs
        if sampler_gpus > 0:
            sampler_devices = ','.join(str(i) for i in range(train_gpus, train_gpus + sampler_gpus))
            err = _ray_node(sampler_devices, sampler_gpus, node_name='gpu_worker')
            if err:
                return f'Ray GPU worker start failed: {err}'

        # CPU Worker node — processor (no GPU)
        _ray_node('', 0, node_name='cpu_worker')
        return None

    @staticmethod
    def _launch_server_process(config_path: str) -> tuple:
        """Launch Twinkle Server as a detached background process.

        Returns (proc, log_path, error). On success error is None.
        """
        import subprocess as _sp
        from pathlib import Path

        log_dir = Path.home() / '.cache' / 'twinkle'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / 'server.log')
        log_file = open(log_path, 'w')

        cmd = ['python', '-m', 'twinkle.server', 'launch', '--config', config_path]
        try:
            proc = _sp.Popen(
                cmd, stdout=log_file, stderr=_sp.STDOUT,
                start_new_session=True,
            )
        except OSError as e:
            log_file.close()
            return None, log_path, f'Failed to start Twinkle server: {e}'
        return proc, log_path, None

    @staticmethod
    def _wait_server_ready(
        server_url: str,
        proc,
        log_path: str,
        sampler_list: list[dict],
        model_id: str,
        t_gpus: int,
        backend: str,
        config_path: str,
    ) -> dict:
        """Poll server until healthy (healthz + sampler engine ready)."""
        import time
        import urllib.request
        import urllib.error

        timeout_s = 120 if sampler_list else 60
        needed = t_gpus + sum(
            s.get('gpus') or (s.get('tp', 1) * s.get('dp', 1)) for s in sampler_list
        )

        for _ in range(timeout_s):
            time.sleep(1)
            if proc.poll() is not None:
                # Server died — read log tail to diagnose
                log_tail = ToolExecutor._read_log_tail(log_path, max_chars=2000)
                error_msg = (
                    f'Server exited immediately (code={proc.returncode}). '
                    f'Model: {model_id}, GPUs: {t_gpus}, Samplers: {len(sampler_list)}.\n'
                    f'--- server.log tail ---\n{log_tail}'
                )
                return {
                    'status': 'error',
                    'error': error_msg,
                    'log_path': log_path,
                    'hint': 'Check if required packages are installed (pip install -e ".[all]").',
                }
            try:
                urllib.request.urlopen(f'{server_url}/api/v1/healthz', timeout=2)
            except (OSError, Exception):
                continue

            # healthz OK — additionally wait for sampler vLLM engines
            if sampler_list and not ToolExecutor._probe_sampler_ready(server_url, sampler_list, model_id):
                return {
                    'status': 'started',
                    'warning': 'Server is up but sampler may still be loading.',
                    'server_url': server_url, 'server_pid': proc.pid,
                    'model_id': model_id, 'log_path': log_path,
                }

            return {
                'status': 'started',
                'server_url': server_url, 'server_pid': proc.pid,
                'model_id': model_id, 'train_gpus': t_gpus,
                'backend': backend,
                'samplers': [s.get('model_id') for s in sampler_list],
                'total_gpus_used': needed,
                'config_path': config_path, 'log_path': log_path,
            }

        return {
            'status': 'timeout',
            'error': 'Health check did not pass within timeout. Models may still be loading.',
            'server_pid': proc.pid, 'log_path': log_path,
        }

    @staticmethod
    def _read_log_tail(log_path: str, max_chars: int = 2000) -> str:
        """Read the tail of a log file for error diagnosis."""
        try:
            with open(log_path, 'r', errors='replace') as f:
                content = f.read()
            if len(content) <= max_chars:
                return content.strip()
            return content[-max_chars:].strip()
        except OSError:
            return '(could not read log file)'

    @staticmethod
    def _probe_sampler_ready(server_url: str, sampler_list: list[dict], fallback_model_id: str) -> bool:
        """Probe sampler route up to 90s to confirm vLLM engine is loaded."""
        import time
        import urllib.request
        import urllib.error

        s_mid = sampler_list[0].get('model_id', fallback_model_id)
        probe_url = f'{server_url}/api/v1/sampler/{s_mid}/twinkle/create'

        for _ in range(90):
            try:
                req = urllib.request.Request(
                    probe_url, method='POST', data=b'{}',
                    headers={'Content-Type': 'application/json'},
                )
                urllib.request.urlopen(req, timeout=5)
                return True  # non-error response = ready
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    return True  # 4xx = actor alive, just bad request
                time.sleep(1)  # 5xx = still loading
            except (OSError, Exception):
                time.sleep(1)
        return False

    @staticmethod
    def _generate_server_config(
        model_id: str,
        train_gpus: int,
        port: int = 8000,
        backend: str = 'transformers',
        samplers: list[dict] | None = None,
    ) -> str:
        """Generate a server_config.yaml from template and return its path.

        Supports multi-model topology:
          - 1 training model (student)
          - N sampler/teacher models (for RL/OPD)
          - 1 processor service
        """
        from pathlib import Path
        import yaml

        sampler_list = samplers or []

        # Sanitize model name for use in route/names
        def _short(mid: str) -> str:
            return mid.split('/')[-1] if '/' in mid else mid

        model_short = _short(model_id)

        # Collect all model IDs for supported_models
        all_model_ids = [model_id] + [s['model_id'] for s in sampler_list]

        # === Build applications list ===
        applications = []

        # 1. API Gateway
        applications.append({
            'name': 'server',
            'route_prefix': '/api/v1',
            'import_path': 'server',
            'args': {
                'server_config': {'per_token_model_limit': 3},
                'supported_models': all_model_ids,
            },
            'deployments': [{
                'name': 'TinkerCompatServer',
                'max_ongoing_requests': 50,
                'autoscaling_config': {
                    'min_replicas': 1,
                    'max_replicas': 1,
                    'target_ongoing_requests': 128,
                },
                'ray_actor_options': {'num_cpus': 0.1},
            }],
        })

        # 2. Build GPU-requiring applications (model + samplers),
        #    then sort by GPU count DESCENDING before appending.
        #    Largest PG deploys first → it has the fewest node choices →
        #    avoids GPU scheduling deadlock on single-machine multi-node.
        gpu_apps: list[tuple[int, dict]] = []  # (gpu_count, app_config)

        # 2a. Training model worker (student)
        gpu_apps.append((train_gpus, {
            'name': f'models-{model_short}',
            'route_prefix': f'/api/v1/model/{model_id}',
            'import_path': 'model',
            'args': {
                'backend': backend,
                'model_id': f'ms://{model_id}',
                'max_length': 500000,  # total tokens per forward pass (must match max_input_tokens)
                'nproc_per_node': train_gpus,
                'device_group': {
                    'name': 'model',
                    'ranks': train_gpus,
                    'device_type': 'cuda',
                },
                'device_mesh': {
                    'device_type': 'cuda',
                    'dp_size': train_gpus,
                },
                'queue_config': {
                    'rps_limit': 100,
                    'tps_limit': 100000,
                    'max_input_tokens': 500000,
                },
                'adapter_config': {
                    'adapter_timeout': 600,
                },
            },
            'deployments': [{
                'name': 'ModelManagement',
                'autoscaling_config': {
                    'min_replicas': 1,
                    'max_replicas': 1,
                    'target_ongoing_requests': 16,
                },
                'ray_actor_options': {
                    'num_cpus': 0.1,
                    'runtime_env': {
                        'env_vars': {'TWINKLE_TRUST_REMOTE_CODE': '1'},
                    },
                },
            }],
        }))

        # 2b. Sampler/teacher models
        sampler_name_count: dict[str, int] = {}
        for sampler_cfg in sampler_list:
            s_model_id = sampler_cfg['model_id']
            s_short = _short(s_model_id)

            # Deduplicate names when multiple samplers share the same short name
            sampler_name_count[s_short] = sampler_name_count.get(s_short, 0) + 1
            if sampler_name_count[s_short] > 1:
                s_name = f'sampler-{s_short}-{sampler_name_count[s_short]}'
            else:
                s_name = f'sampler-{s_short}'

            s_engine = sampler_cfg.get('engine', 'vllm')
            s_max_len = sampler_cfg.get('max_model_len', 16000)

            # Compute tp / dp / total GPUs:
            #   tp = tensor parallelism (GPUs per vLLM process, for large models)
            #   dp = data parallelism (number of independent inference replicas)
            #   total GPUs = tp * dp
            s_tp = sampler_cfg.get('tp', 1)
            s_dp = sampler_cfg.get('dp', None)
            s_gpus = sampler_cfg.get('gpus', None)

            if s_dp is not None and s_gpus is not None:
                # Both specified: validate consistency
                s_tp = s_gpus // s_dp if s_tp == 1 else s_tp
            elif s_gpus is not None:
                # Only total GPUs specified: derive dp
                s_dp = max(1, s_gpus // s_tp)
            elif s_dp is not None:
                # Only dp specified: derive total
                s_gpus = s_tp * s_dp
            else:
                # Nothing specified: default to 1 GPU (tp=1, dp=1)
                s_dp = 1
                s_gpus = s_tp * s_dp

            s_total_gpus = s_tp * s_dp

            # Build device_mesh: include tp_size when tp>1 so that
            # world_size = tp*dp and slice_dp dispatch computes correct
            # rank_stride for DP data sharding.
            mesh_config: dict = {'device_type': 'cuda', 'dp_size': s_dp}
            if s_tp > 1:
                mesh_config['tp_size'] = s_tp

            sampler_app: dict = {
                'name': s_name,
                'route_prefix': f'/api/v1/sampler/{s_model_id}',
                'import_path': 'sampler',
                'args': {
                    'model_id': f'ms://{s_model_id}',
                    'nproc_per_node': s_total_gpus,
                    'sampler_type': s_engine,
                    'device_group': {
                        'name': s_name,
                        'ranks': s_total_gpus,
                        'device_type': 'cuda',
                        'gpus_per_worker': s_tp,
                    },
                    'device_mesh': mesh_config,
                    'queue_config': {
                        'rps_limit': 100,
                        'tps_limit': 100000,
                    },
                },
                'deployments': [{
                    'name': 'SamplerManagement',
                    'autoscaling_config': {
                        'min_replicas': 1,
                        'max_replicas': 1,
                        'target_ongoing_requests': 16,
                    },
                    'ray_actor_options': {
                        'num_cpus': 0.1,
                        'runtime_env': {
                            'env_vars': {'TWINKLE_TRUST_REMOTE_CODE': '1'},
                        },
                    },
                }],
            }

            # Add engine-specific args
            if s_engine == 'vllm':
                engine_args = {
                    'max_model_len': s_max_len,
                    'gpu_memory_utilization': 0.85,
                    'enable_lora': True,
                    'logprobs_mode': 'processed_logprobs',
                }
                # Set tensor_parallel_size when tp > 1
                if s_tp > 1:
                    engine_args['tensor_parallel_size'] = s_tp
                sampler_app['args']['engine_args'] = engine_args

            gpu_apps.append((s_total_gpus, sampler_app))

        # 3. Sort GPU apps by GPU count DESCENDING, then append in order.
        #    Largest PG deploys first → claims the largest node → avoids deadlock.
        gpu_apps.sort(key=lambda x: x[0], reverse=True)
        for _, app_cfg in gpu_apps:
            applications.append(app_cfg)

        # 4. Processor service
        applications.append({
            'name': 'processor',
            'route_prefix': '/api/v1/processor',
            'import_path': 'processor',
            'args': {
                'ncpu_proc_per_node': 2,
                'device_group': {
                    'name': 'processor',
                    'ranks': 2,
                    'device_type': 'CPU',
                },
                'device_mesh': {
                    'device_type': 'CPU',
                    'dp_size': 2,
                },
            },
            'deployments': [{
                'name': 'ProcessorManagement',
                'autoscaling_config': {
                    'min_replicas': 1,
                    'max_replicas': 1,
                    'target_ongoing_requests': 128,
                },
                'ray_actor_options': {'num_cpus': 0.1},
            }],
        })

        # === Assemble final config ===
        config = {
            'proxy_location': 'EveryNode',
            'http_options': {
                'host': '0.0.0.0',
                'port': port,
            },
            'applications': applications,
        }

        # Write to ~/.cache/twinkle/server_config.yaml
        config_dir = Path.home() / '.cache' / 'twinkle'
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / 'server_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return str(config_path)

    async def _tool_shutdown_server(self) -> dict:
        """Shut down Twinkle Server and Ray cluster. DESTROYS GPU model state."""
        import subprocess as _sp

        def _shutdown():
            results = {}

            # 1. Try `serve shutdown` to cleanly stop Ray Serve deployments
            try:
                r = _sp.run(['serve', 'shutdown', '-y'], capture_output=True, text=True, timeout=30)
                results['serve_shutdown'] = 'ok' if r.returncode == 0 else r.stderr.strip()
            except (FileNotFoundError, OSError) as e:
                results['serve_shutdown'] = f'skipped: {e}'

            # 2. Kill any remaining twinkle.server processes
            try:
                _sp.run(['pkill', '-f', 'twinkle.server'], capture_output=True, timeout=5)
            except (FileNotFoundError, OSError):
                pass

            # 3. Stop Ray cluster
            try:
                r = _sp.run(['ray', 'stop', '--force'], capture_output=True, text=True, timeout=15)
                results['ray_stop'] = 'ok' if r.returncode == 0 else r.stderr.strip()
            except (FileNotFoundError, OSError) as e:
                results['ray_stop'] = f'failed: {e}'

            results['status'] = 'shutdown_complete'
            results['warning'] = 'All GPU model state has been released.'
            return results

        return await asyncio.get_event_loop().run_in_executor(None, _shutdown)

    # ── Server queries ──

    async def _tool_list_supported_models(self, base_url: str | None = None) -> dict:
        """Query the Twinkle server for supported models."""
        url = base_url or self._resolve_server_url()

        def _query():
            # Use a lightweight HTTP GET instead of init_twinkle_client() which
            # creates a session + heartbeat thread that would leak since we never
            # call close().
            import urllib.request
            import urllib.error

            endpoint = f'{url}/api/v1/twinkle/get_server_capabilities'
            req = urllib.request.Request(endpoint, method='GET')
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())
            models = data.get('supported_models', [])
            # Each model entry may be a dict with 'model_name' or a plain string
            model_names = []
            for m in models:
                if isinstance(m, dict):
                    model_names.append(m.get('model_name', ''))
                else:
                    model_names.append(str(m))
            return {
                'base_url': url,
                'supported_models': model_names,
            }

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _query)
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

    # ── Cluster info ──

    async def _tool_get_cluster_info(self) -> dict:
        """Query cluster resources: try Ray first, fall back to nvidia-smi."""

        def _query():
            # 1. Try connecting to an existing Ray cluster
            ray_info = self._try_ray_cluster()
            if ray_info is not None:
                ray_info['ray_active'] = True
                return ray_info

            # 2. Ray not available — fall back to nvidia-smi
            nvidia_info = self._try_nvidia_smi()
            nvidia_info['ray_active'] = False
            nvidia_info['hint'] = (
                'Ray cluster is not running. To use distributed training, '
                'start Ray first: `ray start --head --num-gpus=N` or use '
                'the server mode run.sh script.'
            )
            return nvidia_info

        return await asyncio.get_event_loop().run_in_executor(None, _query)

    @staticmethod
    def _try_ray_cluster() -> dict | None:
        """Attempt to query an existing Ray cluster. Returns None if unavailable."""
        try:
            import ray
        except ImportError:
            return None

        import logging as _logging

        try:
            if not ray.is_initialized():
                ray.init(
                    address='auto',
                    ignore_reinit_error=True,
                    _timeout_s=5,
                    logging_level=_logging.ERROR,
                    configure_logging=False,
                )

            resources = ray.cluster_resources()
            available = ray.available_resources()
            nodes = ray.nodes()
            gpu_total = resources.get('GPU', 0)
            gpu_available = available.get('GPU', 0)
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
        except Exception:
            try:
                import ray as _ray
                if _ray.is_initialized():
                    _ray.shutdown()
            except Exception:
                pass
            return None

    @staticmethod
    def _try_nvidia_smi() -> dict:
        """Parse nvidia-smi output for local GPU info."""
        import subprocess as _sp

        try:
            result = _sp.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return {'error': f'nvidia-smi failed: {result.stderr.strip()}', 'gpu_total': 0}

            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    try:
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_total_mb': int(parts[2]),
                            'memory_free_mb': int(parts[3]),
                            'utilization_pct': int(parts[4]) if parts[4].isdigit() else 0,
                        })
                    except (ValueError, IndexError):
                        # Skip lines with unparseable values (e.g. [N/A])
                        continue

            gpu_types = sorted(set(g['name'] for g in gpus))
            return {
                'gpu_total': len(gpus),
                'gpu_available': len([g for g in gpus if g['utilization_pct'] < 10]),
                'gpu_types': gpu_types if gpu_types else ['none'],
                'gpus': gpus,
                'source': 'nvidia-smi',
            }
        except FileNotFoundError:
            return {'error': 'nvidia-smi not found (no NVIDIA GPU or driver not installed)', 'gpu_total': 0}
        except Exception as e:
            return {'error': f'nvidia-smi query failed: {e}', 'gpu_total': 0}
