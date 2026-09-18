# Copyright (c) Twinkle Contributors. All rights reserved.
"""Private server lifecycle and cluster tools used by ``ToolExecutor``."""
from __future__ import annotations

import asyncio
import json
import os


class _ServerTools:
    """Server-side operations mixed into ``ToolExecutor``.

    ``ToolExecutor`` owns the URL value; declaring it here makes that host
    requirement visible without assigning runtime state in the mixin.
    """

    _server_url: str | None

    async def _check_server_health(self, url: str) -> bool:
        """Check if Twinkle Server is reachable (non-blocking)."""
        import urllib.error
        import urllib.request

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
                model_id=model_id,
                train_gpus=t_gpus,
                port=port,
                backend=backend,
                samplers=sampler_list,
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
                server_url=server_url,
                proc=proc,
                log_path=log_path,
                sampler_list=sampler_list,
                model_id=model_id,
                t_gpus=t_gpus,
                backend=backend,
                config_path=config_path,
            )

        result = await asyncio.get_event_loop().run_in_executor(None, _start)
        if result.get('status') in ('started', 'already_running'):
            self._server_url = server_url
        return result

    @staticmethod
    def _detect_gpu_count() -> int:
        """Detect total hardware GPU count via nvidia-smi."""
        import subprocess as _sp
        try:
            r = _sp.run(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=10,
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
                'error': (f'Requested {needed} GPUs (train={t_gpus}, samplers={sampler_gpu_total}) '
                          f'but only {total_hw_gpus} available.'),
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
            devices: str,
            num_gpus: int,
            *,
            head: bool = False,
            node_name: str = 'worker',
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
                cmd,
                stdout=log_file,
                stderr=_sp.STDOUT,
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
        import urllib.error
        import urllib.request

        timeout_s = 120 if sampler_list else 60
        needed = t_gpus + sum(s.get('gpus') or (s.get('tp', 1) * s.get('dp', 1)) for s in sampler_list)

        for _ in range(timeout_s):
            time.sleep(1)
            if proc.poll() is not None:
                # Server died — read log tail to diagnose
                log_tail = _ServerTools._read_log_tail(log_path, max_chars=2000)
                error_msg = (f'Server exited immediately (code={proc.returncode}). '
                             f'Model: {model_id}, GPUs: {t_gpus}, Samplers: {len(sampler_list)}.\n'
                             f'--- server.log tail ---\n{log_tail}')
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
            if sampler_list and not _ServerTools._probe_sampler_ready(server_url, sampler_list, model_id):
                return {
                    'status': 'started',
                    'warning': 'Server is up but sampler may still be loading.',
                    'server_url': server_url,
                    'server_pid': proc.pid,
                    'model_id': model_id,
                    'log_path': log_path,
                }

            return {
                'status': 'started',
                'server_url': server_url,
                'server_pid': proc.pid,
                'model_id': model_id,
                'train_gpus': t_gpus,
                'backend': backend,
                'samplers': [s.get('model_id') for s in sampler_list],
                'total_gpus_used': needed,
                'config_path': config_path,
                'log_path': log_path,
            }

        return {
            'status': 'timeout',
            'error': 'Health check did not pass within timeout. Models may still be loading.',
            'server_pid': proc.pid,
            'log_path': log_path,
        }

    @staticmethod
    def _read_log_tail(log_path: str, max_chars: int = 2000) -> str:
        """Read the tail of a log file for error diagnosis."""
        try:
            with open(log_path, errors='replace') as f:
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
        import urllib.error
        import urllib.request

        s_mid = sampler_list[0].get('model_id', fallback_model_id)
        probe_url = f'{server_url}/api/v1/sampler/{s_mid}/twinkle/create'

        for _ in range(90):
            try:
                req = urllib.request.Request(
                    probe_url,
                    method='POST',
                    data=b'{}',
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
        import yaml
        from pathlib import Path

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
            'name':
            'server',
            'route_prefix':
            '/api/v1',
            'import_path':
            'server',
            'args': {
                'server_config': {
                    'per_token_model_limit': 3
                },
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
                'ray_actor_options': {
                    'num_cpus': 0.1
                },
            }],
        })

        # 2. Build GPU-requiring applications (model + samplers),
        #    then sort by GPU count DESCENDING before appending.
        #    Largest PG deploys first → it has the fewest node choices →
        #    avoids GPU scheduling deadlock on single-machine multi-node.
        gpu_apps: list[tuple[int, dict]] = []  # (gpu_count, app_config)

        # 2a. Training model worker (student)
        gpu_apps.append((
            train_gpus,
            {
                'name':
                f'models-{model_short}',
                'route_prefix':
                f'/api/v1/model/{model_id}',
                'import_path':
                'model',
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
                            'env_vars': {
                                'TWINKLE_TRUST_REMOTE_CODE': '1'
                            },
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
                'name':
                s_name,
                'route_prefix':
                f'/api/v1/sampler/{s_model_id}',
                'import_path':
                'sampler',
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
                            'env_vars': {
                                'TWINKLE_TRUST_REMOTE_CODE': '1'
                            },
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
            'name':
            'processor',
            'route_prefix':
            '/api/v1/processor',
            'import_path':
            'processor',
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
                'ray_actor_options': {
                    'num_cpus': 0.1
                },
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

    async def _tool_list_supported_models(self, base_url: str | None = None) -> dict:
        """Query the Twinkle server for supported models."""
        url = base_url or self._server_url or os.environ.get('TWINKLE_SERVER_URL') or 'http://localhost:8000'

        def _query():
            # Use a lightweight HTTP GET instead of init_twinkle_client() which
            # creates a session + heartbeat thread that would leak since we never
            # call close().
            import urllib.error
            import urllib.request

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
            nvidia_info['hint'] = ('Ray cluster is not running. To use distributed training, '
                                   'start Ray first: `ray start --head --num-gpus=N` or use '
                                   'the server mode run.sh script.')
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
                [
                    'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=10,
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

            gpu_types = sorted({g['name'] for g in gpus})
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
