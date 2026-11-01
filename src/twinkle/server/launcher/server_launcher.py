# Copyright (c) ModelScope Contributors. All rights reserved.
"""Initialize server configuration, deploy applications, and own their shutdown."""
from __future__ import annotations

import json
import os
import signal
import threading
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from twinkle import get_logger
from twinkle.hub.model_alias import MODEL_ID_ALIASES_ENV, build_model_alias_map, extract_model_alias
from twinkle.patch.ray_serve import apply_ray_serve_patches
from twinkle.server.config import ServerConfig
from twinkle.server.config.application_spec import ApplicationSpec
from twinkle.server.config.resources import build_propagated_env_vars, build_worker_env_vars
from .builder_registry import get_builders, resolve_builder

logger = get_logger()


def _worker_setup_hook() -> None:
    """Ray ``worker_process_setup_hook``: apply the Serve patches, then the env-var hook.

    Composes the two things every twinkle worker may need at process start: the Ray Serve
    monkey-patches (proxy actors) and :func:`twinkle.utils.loader.run_worker_setup_hook`
    (startup plugin registration, driven by ``TWINKLE_WORKER_SETUP_HOOK``). Keeping them in one
    job-level hook is what makes the composition reach every worker process -- Ray only honours
    ``worker_process_setup_hook`` at ``ray.init`` time, not per actor.
    """
    from twinkle.patch.ray_serve import _apply_patch_in_worker_process
    from twinkle.utils.loader import run_worker_setup_hook

    _apply_patch_in_worker_process()
    run_worker_setup_hook()


def build_ray_runtime_env() -> dict[str, Any]:
    """The ``runtime_env`` every Ray worker in a twinkle server needs.

    Bundles the Ray Serve monkey-patches (which must be re-applied inside worker processes such as the
    Serve ProxyActor) and the env-var worker setup hook with the telemetry/persistence env vars
    propagated from the driver. Shared by the launcher's own ``ray.init`` and by any caller that starts
    a local Ray head before handing off to :func:`launch_server` (e.g. swift's deploy recipe), so both
    seed worker processes identically rather than each re-deriving the merge.
    """
    runtime_env = {'worker_process_setup_hook': 'twinkle.server.launcher.server_launcher._worker_setup_hook'}
    propagated_env_vars = build_propagated_env_vars()
    if propagated_env_vars:
        merged_env_vars = dict(runtime_env.get('env_vars') or {})
        merged_env_vars.update(propagated_env_vars)
        runtime_env['env_vars'] = merged_env_vars
    return runtime_env


def ensure_ray_cluster(*, namespace: str, address: str | None = None) -> None:
    """Bring up (or attach to) the Ray runtime a Serve deployment needs.

    The deploy-side entry for Ray bring-up: with no cluster running this starts a local head seeded
    with :func:`build_ray_runtime_env` (the patches Ray Serve's proxy actors need); ``address='auto'``
    attaches to an existing cluster instead. Idempotent -- an already-initialized runtime is left
    untouched. Both the launcher itself and a caller that starts the head before handing off to
    :func:`launch_server` (e.g. swift's deploy recipe) go through here, so worker seeding stays
    single-sourced.
    """
    import ray

    if ray.is_initialized():
        return
    init_kwargs: dict[str, Any] = {
        'namespace': namespace,
        'runtime_env': build_ray_runtime_env(),
        'ignore_reinit_error': True,
    }
    if address:
        init_kwargs['address'] = address
    ray.init(**init_kwargs)
    logger.info(f'ray runtime ready (namespace={namespace}, address={address or "local head"})')


class ServerLauncher:
    """Own startup and cleanup; business scheduling stays in the controllers."""

    def __init__(self, config: ServerConfig, ray_namespace: str | None = None):
        if not isinstance(config, ServerConfig):
            raise TypeError('ServerLauncher requires a typed ServerConfig instance; '
                            f'got {type(config).__name__}. Build one with '
                            'ServerConfig.from_yaml(path) or ServerConfig(...).')
        self.config = config
        self.ray_namespace = ray_namespace or config.ray_namespace or 'twinkle_cluster'
        self._builders: dict[str, Callable] = {}
        self._ray_initialized = False
        self._serve_started = False
        self._training_controller = None
        self._sampler_controllers = []

    def _get_builders(self) -> dict[str, Callable]:
        if not self._builders:
            self._builders = get_builders()
        return self._builders

    def _resolve_builder(self, import_path: str) -> Callable:
        return resolve_builder(import_path, self._get_builders())

    def _configure_environment(self) -> None:
        if any(app.import_path in ('model', 'sampler') for app in self.config.applications):
            os.environ['TWINKLE_STATE_RAY_NAMESPACE'] = self.ray_namespace
        os.environ.update(self.config.persistence.to_env_vars())
        gateways = [app for app in self.config.applications if app.import_path == 'server']
        if gateways:
            os.environ.update(gateways[0].args.server_config.to_env_vars())
        aliases = build_model_alias_map(self.config.applications)
        if aliases:
            os.environ[MODEL_ID_ALIASES_ENV] = json.dumps(aliases, ensure_ascii=False)
        else:
            os.environ.pop(MODEL_ID_ALIASES_ENV, None)
        self._configure_telemetry()

    def _configure_telemetry(self) -> None:
        telemetry = self.config.telemetry
        os.environ['TWINKLE_TELEMETRY_ENABLED'] = str(int(telemetry.enabled))
        if not telemetry.enabled:
            return
        from twinkle.server.telemetry import init_telemetry
        init_telemetry(telemetry)
        os.environ.update({
            'TWINKLE_TELEMETRY_DEBUG': str(int(telemetry.debug)),
            'TWINKLE_TELEMETRY_SERVICE': telemetry.service_name,
            'TWINKLE_TELEMETRY_ENDPOINT': telemetry.otlp_endpoint,
            'TWINKLE_TELEMETRY_INTERVAL': str(telemetry.export_interval_ms),
        })

    def _init_ray(self) -> None:
        """Initialize Ray with the resolved namespace (attach-or-local via :func:`ensure_ray_cluster`)."""
        if self._ray_initialized:
            return
        namespace = self.ray_namespace or self.config.ray_namespace or 'twinkle_cluster'
        ensure_ray_cluster(namespace=namespace, address='auto')
        self._ray_initialized = True

    def _start_serve(self) -> None:
        """Start Ray Serve with http_options from config, or attach to a running instance.

        Serve's http_options are cluster-scoped: when a Serve instance is already running on
        the cluster (another deployment started it), it is REUSED -- shutting it down here
        would kill every live application, which is exactly wrong for coexisting deployments.
        The running instance's listener (host/port/TLS) wins; the request is logged so the
        operator can see which knobs were ignored.
        """
        if self._serve_started:
            return
        from ray import serve

        try:
            from ray.serve.context import _get_global_client
            client = _get_global_client()
        except Exception:
            client = None
        if client is not None:
            requested = self.config.http_options.model_dump()
            logger.info('Ray Serve is already running on this cluster; reusing it. Its listener '
                        '(host/port) was fixed when Serve first started; the requested '
                        f'http_options {requested} take effect only through route prefixes.')
            self._serve_started = True
            return

        http_options = self.config.http_options.model_dump()
        serve_kwargs: dict[str, Any] = {'http_options': http_options}
        # ``proxy_location`` controls where the Ray Serve HTTP proxy runs
        # (``EveryNode`` / ``HeadOnly`` / ``Disabled``). The example configs
        # set this field, so honour it here instead of silently ignoring.
        if self.config.proxy_location:
            serve_kwargs['proxy_location'] = self.config.proxy_location
        serve.start(**serve_kwargs)
        self._serve_started = True

    def _deployment_options(self, options: dict, env: dict | None = None) -> dict:
        options = deepcopy(options)
        options.pop('name', None)
        actor = options.setdefault('ray_actor_options', {})
        runtime_env = actor.setdefault('runtime_env', {})
        runtime_env['env_vars'] = {**build_worker_env_vars(), **runtime_env.get('env_vars', {}), **(env or {})}
        return options

    def _gateway_arguments(self) -> dict[str, Any]:
        args = {'http_options': self.config.http_options.model_dump()}
        models = self.config.training_config.models
        if models:
            args['training_models'] = [model.name for model in models]
            args['supported_models'] = list(
                dict.fromkeys(
                    alias for app in self.config.applications
                    if app.import_path == 'sampler' and (alias := extract_model_alias(app.route_prefix, 'sampler'))))
        return args

    def _deploy_sampler(self, app: ApplicationSpec) -> None:
        import ray
        from ray import serve

        from twinkle.server.sampler.pool.actor import create_sampler_controller
        from twinkle.server.sampler.pool.router import build_sampler_router
        controller, options, pool_id = create_sampler_controller(app, self.ray_namespace,
                                                                 self.config.resources.model_dump(),
                                                                 build_worker_env_vars())
        self._sampler_controllers.append(controller)
        ray.get(controller.initialize.remote())
        serve.run(
            build_sampler_router(self.ray_namespace, pool_id, options), name=app.name, route_prefix=app.route_prefix)

    def _deploy_application(self, app_spec: ApplicationSpec) -> None:
        """Model declarations go to the training controller; other apps deploy here."""
        if app_spec.import_path == 'model':
            return
        if app_spec.import_path == 'sampler':
            self._deploy_sampler(app_spec)
            return
        from ray import serve
        args = {key: value for key, value in dict(app_spec.args).items() if value is not None}
        if app_spec.import_path == 'server':
            defaults = self._gateway_arguments()
            for key, value in defaults.items():
                args.setdefault(key, value)
        deployment = app_spec.deployments[0] if app_spec.deployments else {}
        options = self._deployment_options(deployment)
        builder = self._resolve_builder(app_spec.import_path)
        serve.run(builder(deploy_options=options, **args), name=app_spec.name, route_prefix=app_spec.route_prefix)

    def _training_router_options(self, model) -> dict:
        options = self._deployment_options(model.deploy_options, model.env_vars)
        if 'autoscaling_config' not in options:
            options.setdefault('num_replicas', 1)
        options['ray_actor_options'].setdefault('num_cpus', 0.1)
        return options

    def _deploy_training(self) -> None:
        import ray
        from ray import serve

        from twinkle.server.model.pool.actor import TrainingControllerActor, controller_name
        from twinkle.server.model.pool.router import build_training_router
        config = self.config.training_config
        controller = TrainingControllerActor.options(
            name=controller_name(self.ray_namespace),
            namespace=self.ray_namespace,
            runtime_env={
                'env_vars': build_worker_env_vars()
            }).remote(config.model_dump(), self.ray_namespace)
        self._training_controller = controller
        try:
            ray.get(controller.initialize.remote())
            for model in config.models:
                app = build_training_router(config, model, self.ray_namespace, self.config.http_options.model_dump(),
                                            self._training_router_options(model))
                serve.run(
                    app,
                    name=f'training-router-{self.ray_namespace}-{model.name}',
                    route_prefix=model.route_prefix or f'/api/v1/model/{model.name}')
        except Exception:
            self._run_cleanup([('training controller rollback', self._stop_training_controller)], raise_errors=False)
            raise

    def _stop_controller(self, controller) -> None:
        import ray

        # Keep an unconfirmed owner available for cleanup retry; a timeout alone
        # does not certify its workers or devices were released.
        ray.get(controller.shutdown.remote(), timeout=self.config.resources.shutdown_timeout_s)
        ray.kill(controller, no_restart=True)

    def _stop_training_controller(self) -> None:
        if self._training_controller is not None:
            self._stop_controller(self._training_controller)
            self._training_controller = None

    def _stop_sampler_controller(self, controller) -> None:
        self._stop_controller(controller)
        self._sampler_controllers.remove(controller)

    @staticmethod
    def _run_cleanup(actions: list[tuple[str, Callable]], *, raise_errors: bool) -> None:
        """Attempt every independent cleanup; rollback preserves the startup error."""
        failures = []
        for name, action in actions:
            try:
                action()
            except Exception as exc:
                failures.append((name, exc))
                logger.exception('%s failed', name)
        if failures and raise_errors:
            names = ', '.join(name for name, _ in failures)
            raise RuntimeError(f'Server cleanup is incomplete: {names}') from failures[0][1]

    def _shutdown(self, *, raise_errors: bool = True) -> None:
        from functools import partial
        from ray import serve

        from twinkle.server.telemetry import flush_telemetry_safely
        actions = [('training controller shutdown', self._stop_training_controller)]
        actions.extend(('sampler controller shutdown', partial(self._stop_sampler_controller, controller))
                       for controller in tuple(self._sampler_controllers))
        if self._serve_started:
            actions.append(('Ray Serve shutdown', serve.shutdown))
        actions.append(('telemetry flush', flush_telemetry_safely))
        self._run_cleanup(actions, raise_errors=raise_errors)
        self._serve_started = False

    @staticmethod
    def _wait_for_shutdown() -> None:
        stopped = threading.Event()

        def signal_shutdown(signum, frame):
            logger.info('Received %s', signal.Signals(signum).name)
            stopped.set()

        signal.signal(signal.SIGINT, signal_shutdown)
        signal.signal(signal.SIGTERM, signal_shutdown)
        stopped.wait()

    def launch(self, blocking: bool = True) -> None:
        """Launch applications and their controllers, optionally waiting for shutdown."""
        if not self.config.applications:
            logger.warning('No applications configured')
            return
        apply_ray_serve_patches()
        self._configure_environment()
        self._init_ray()
        try:
            self._start_serve()
            for app in self.config.applications:
                self._deploy_application(app)
            if self.config.training_config.models:
                self._deploy_training()
            logger.info('All applications started')
            if not blocking:
                return
            self._wait_for_shutdown()
        except Exception:
            self._shutdown(raise_errors=False)
            raise
        self._shutdown()


def _warn_public_unauthenticated(config: ServerConfig) -> None:
    """Loudly flag a 0.0.0.0 listener with no gateway ``api_key`` anywhere.

    Every inference route would be reachable without credentials from any host that
    can route to this machine -- the advisory class where an open deployment becomes
    free compute plus an internal-network pivot. Keeping the check here (the single
    launch choke point) covers YAML and programmatic configs alike.
    """
    if config.http_options.host != '0.0.0.0':
        return
    if any(app.import_path == 'server' and getattr(app.args, 'api_key', None)
           for app in config.applications):
        return
    logger.warning('server binds 0.0.0.0:%s with no api_key configured on any gateway application: '
                   'every route is reachable unauthenticated by any network peer. Set ServerArgs.api_key '
                   '(--api_key) or bind a non-public host.', config.http_options.port)


def launch_server(
    config: ServerConfig | None = None,
    config_path: str | Path | None = None,
    ray_namespace: str | None = None,
    blocking: bool = True,
) -> None:
    """Launch a twinkle server — the single launch choke point.

    Exactly one of ``config`` (a :class:`ServerConfig` instance) or
    ``config_path`` (a YAML file) must be provided. When a path is given the
    config is loaded via :meth:`ServerConfig.from_yaml` here, so there is one
    construction path: load config → ``ServerLauncher(config=...)`` →
    ``launch()``. The call blocks until a SIGINT/SIGTERM signal is received,
    unless ``blocking=False`` (detached deployment: the driver exits and the
    applications stay up on the Ray cluster).

    Raises:
        ValueError: neither ``config`` nor ``config_path`` was provided.
        TypeError: ``config`` is not a :class:`ServerConfig` instance — raw
            dicts are not accepted.

    Examples:
        launch_server(config_path="server_config.yaml")
        launch_server(config=ServerConfig(...))
    """
    if config is None and config_path is None:
        raise ValueError("Either 'config' or 'config_path' must be provided")

    if config is None:
        # ``FileNotFoundError`` / ``ConfigParseError`` / ``pydantic.ValidationError``
        # propagate so the caller can surface a precise message.
        config = ServerConfig.from_yaml(config_path)

    _warn_public_unauthenticated(config)

    launcher = ServerLauncher(
        config=config,
        ray_namespace=ray_namespace or config.ray_namespace,
    )
    launcher.launch(blocking=blocking)
