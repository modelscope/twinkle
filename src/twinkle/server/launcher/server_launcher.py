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
    from twinkle.server.deployment import apply_ray_serve_patches
    from twinkle.infra._ray.ray_helper import worker_setup_hook

    apply_ray_serve_patches()
    worker_setup_hook()


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


def ensure_ray_cluster(*, namespace: str, address: str | None = None, persistent: bool = False) -> None:
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
    if address == 'local':
        address = None
    if persistent and address is None:
        import shutil
        import socket
        import subprocess
        import sys

        # A detached deployment needs a head whose lifetime is independent of this driver.
        try:
            with socket.create_connection(('127.0.0.1', 6379), timeout=1.0):
                pass
        except OSError:
            executable = Path(sys.executable).parent / 'ray'
            ray_cli = str(executable) if executable.is_file() else shutil.which('ray')
            if ray_cli is None:
                raise RuntimeError('The Ray CLI is required to start a persistent local cluster.')
            result = subprocess.run([ray_cli, 'start', '--head'], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f'Ray head startup failed:\n{result.stdout}\n{result.stderr}')
        address = 'auto'
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

    def __init__(self, config: ServerConfig, ray_namespace: str | None = None, *,
                 ray_address: str | None = 'auto', persistent_cluster: bool = False):
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
        self.ray_address = ray_address
        self.persistent_cluster = persistent_cluster
        self._existing_applications: set[str] = set()
        self._owned_applications: set[str] = set()
        self._published_models: set[str] = set()

    @staticmethod
    def _application_details() -> dict[str, Any]:
        """Read the authoritative route table; an unavailable controller is not an empty deployment."""
        from ray.serve.context import _get_global_client

        client = _get_global_client(raise_if_no_controller_running=False)
        return {} if client is None else dict(client.get_serve_details()['applications'])

    @classmethod
    def _model_names(cls) -> list[str]:
        names = []
        for name, details in cls._application_details().items():
            route = details.get('route_prefix') or ''
            if name.startswith('sampler-') and '/sampler/' in route:
                names.append(route.split('/sampler/', 1)[1])
        return names

    @staticmethod
    def _gateway_state(gateway_name: str):
        """Read the live gateway's deployment info, route, user config and model registry."""
        from ray.serve.context import _get_global_client
        client = _get_global_client()
        info, route = client.get_deployment_info('GatewayServer', gateway_name)
        config = deepcopy(info.deployment_config.user_config or {})
        current = config.get('supported_models', info.replica_config.init_kwargs.get('supported_models') or [])
        return info, route, config, current

    @staticmethod
    def _update_gateway_models(gateway_name: str, *, add: list | None = None,
                               remove: list[str] | None = None) -> list[str]:
        """Persist discovery through Serve reconfiguration while preserving all existing deployment settings.

        Returns the names now registered to the given ``add`` entries (newly added or already
        identical). A name registered to a DIFFERENT entry is a cross-deployment collision:
        the registry routes a name to one deployment, so keeping the first writer silently
        would misroute the second -- fail loudly instead.
        """
        from ray import serve
        from ray.serve.deployment import Application, Deployment

        info, route, config, current = ServerLauncher._gateway_state(gateway_name)
        removed = set(remove or ())

        def model_name(item):
            return item if isinstance(item, str) else item['model_name']

        models = {model_name(item): item for item in current if model_name(item) not in removed}
        published = []
        for item in add or ():
            name = model_name(item)
            if name in models:
                if models[name] != item:
                    raise ValueError(f'Model name {name!r} is already registered to a different entry '
                                     f'({models[name]!r}); stop that deployment or rename the adapter.')
            else:
                models[name] = item
            published.append(name)
        registry = list(models.values())
        if registry == current:
            return published
        config['supported_models'] = registry
        deployment = Deployment('GatewayServer', info.deployment_config, info.replica_config,
                                version=info.version, _internal=True).options(user_config=config)
        serve.run(Application(deployment), name=gateway_name, route_prefix=route)
        return published

    def _publish_models(self, base_names: list[str]) -> None:
        """Register this launcher's samplers and their adapter aliases on every gateway.

        A reused gateway kept its original registry, so the freshly built spec is the only
        place the adapter aliases exist: publish the full entries (dict aliases carry
        base_model routing), not just the base names, and own exactly the names that got
        registered so shutdown removes only this launcher's entries.
        """
        for app in self.config.applications:
            if app.import_path != 'server':
                continue
            registry = app.args.supported_models or []
            entries = []
            for base in base_names:
                matched = [item for item in registry
                           if item == base or (isinstance(item, dict) and item.get('base_model') == base)]
                entries.extend(matched or [base])
            if entries:
                self._published_models.update(self._update_gateway_models(app.name, add=entries))

    def _prepare_model_deployment(self, model_name: str) -> None:
        """Share gateway/data-plane applications without replacing a live sampler."""
        applications = self._application_details()
        sampler_name = f'sampler-{model_name.replace("/", "_")}'
        if sampler_name in applications:
            raise ValueError(f'Model {model_name!r} is already deployed; stop it before replacing it.')
        names = set(self._model_names()) | {model_name}
        for app in self.config.applications:
            if app.import_path == 'server':
                registry = list(app.args.supported_models or [])
                registry.extend(name for name in sorted(names) if name not in registry)
                app.args.supported_models = registry
        self.config.applications = [app for app in self.config.applications
                                    if not (app.import_path == 'data_plane' and app.name in applications)]

    @staticmethod
    def _wait_for_application(name: str, timeout: float = 600.0) -> None:
        """Wait for readiness and propagate terminal startup failures."""
        from time import monotonic, sleep
        from ray import serve

        deadline = monotonic() + timeout
        while monotonic() < deadline:
            details = serve.status().applications.get(name)
            if details is not None:
                status = getattr(details.status, 'value', str(details.status))
                if status == 'RUNNING':
                    return
                if status in ('DEPLOY_FAILED', 'UNHEALTHY'):
                    raise RuntimeError(f'Application {name!r} reached {status}: {details.message}')
            sleep(min(2.0, max(0.0, deadline - monotonic())))
        raise TimeoutError(f'Application {name!r} did not reach RUNNING within {timeout}s.')

    def stop_model(self, model_name: str) -> None:
        """Release one deployed sampler and update the gateway without stopping other tenants."""
        from ray import serve
        from twinkle.server.sampler.pool.actor import stop_sampler_controller

        self._init_ray()
        applications = self._application_details()
        sampler_name = f'sampler-{model_name.replace("/", "_")}'
        gateways = [app for app in self.config.applications if app.import_path == 'server']
        if len(gateways) != 1:
            raise ValueError('Stopping a model requires exactly one gateway configuration.')
        gateway = gateways[0]
        route = (applications.get(sampler_name) or {}).get('route_prefix')
        route = route or f'{gateway.route_prefix}/sampler/{model_name}'
        failures = []
        released = False
        try:
            released = stop_sampler_controller(self.ray_namespace, route)
        except Exception as exc:
            failures.append(exc)
            logger.exception('Sampler pool cleanup failed for %s', model_name)
        if sampler_name not in applications and not released and not failures:
            raise ValueError(f'No deployment found for model {model_name!r}.')
        if sampler_name in applications:
            serve.delete(sampler_name)
        remaining = [name for name in self._model_names() if name != model_name]
        if gateway.name in applications:
            try:
                # Adapter aliases route to this sampler through base_model; leaving them
                # behind would advertise a model whose engine is gone. Derive them from
                # the LIVE registry (this launcher may not be the one that published them).
                _, _, _, current = ServerLauncher._gateway_state(gateway.name)
                aliases = [item['model_name'] for item in current
                           if isinstance(item, dict) and item.get('base_model') == model_name]
                self._update_gateway_models(gateway.name, remove=[model_name] + aliases)
            except BaseException as error:
                failures.append(error)
                logger.exception('Gateway discovery cleanup failed for %s', model_name)
            # Other training/processor deployments may still rely on the shared gateway.
            unrelated = set(self._application_details()) - {gateway.name, 'data-plane'}
            if not remaining and not unrelated and not failures:
                serve.delete(gateway.name)
                if 'data-plane' in applications:
                    serve.delete('data-plane')
        if failures:
            raise RuntimeError(f'Sampler cleanup is incomplete for {model_name!r}.') from failures[0]
        logger.info('Stopped deployment for model %s', model_name)

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
        ensure_ray_cluster(namespace=namespace, address=self.ray_address, persistent=self.persistent_cluster)
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

        from ray.serve.context import _get_global_client
        client = _get_global_client(raise_if_no_controller_running=False)
        self._existing_applications = set(self._application_details())
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
        if app.name not in self._existing_applications:
            self._owned_applications.add(app.name)
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
        if app_spec.import_path == 'server' and app_spec.name in self._existing_applications:
            logger.info('Reusing gateway %s with its existing listener, authentication and session settings.',
                        app_spec.name)
            return
        if app_spec.name not in self._existing_applications:
            self._owned_applications.add(app_spec.name)
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
                router_name = f'training-router-{self.ray_namespace}-{model.name}'
                if router_name not in self._existing_applications:
                    self._owned_applications.add(router_name)
                app = build_training_router(config, model, self.ray_namespace, self.config.http_options.model_dump(),
                                            self._training_router_options(model))
                serve.run(
                    app,
                    name=f'training-router-{self.ray_namespace}-{model.name}',
                    route_prefix=model.route_prefix or f'/api/v1/model/{model.name}')
        except BaseException:
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
            except BaseException as exc:
                failures.append((name, exc))
                logger.exception('%s failed', name)
        if failures and raise_errors:
            names = ', '.join(name for name, _ in failures)
            raise BaseExceptionGroup(f'Server cleanup is incomplete: {names}', [error for _, error in failures])

    def _unpublish_models(self) -> None:
        """Remove only this launcher's discovery entries from gateways that remain shared."""
        if not self._published_models:
            return
        applications = self._application_details()
        actions = [(f'{app.name} discovery cleanup',
                    lambda app=app: self._update_gateway_models(app.name, remove=list(self._published_models)))
                   for app in self.config.applications
                   if app.import_path == 'server' and app.name in applications]
        self._run_cleanup(actions, raise_errors=True)
        self._published_models.clear()

    def _delete_owned_applications(self) -> None:
        """Leave shared applications alive while any other deployment still uses them."""
        from ray import serve

        shared = {app.name for app in self.config.applications if app.import_path in ('server', 'data_plane')}
        unrelated = set(self._application_details()) - self._owned_applications - shared
        failures = []
        for name in tuple(self._owned_applications):
            if name in shared and unrelated:
                continue
            try:
                serve.delete(name)
            except BaseException as error:
                failures.append(error)
                logger.exception('Application cleanup failed for %s', name)
            else:
                self._owned_applications.remove(name)
        if failures:
            raise BaseExceptionGroup('Owned application cleanup is incomplete', failures)

    def _shutdown(self, *, raise_errors: bool = True) -> None:
        from functools import partial

        from twinkle.server.telemetry import flush_telemetry_safely
        actions = [('training controller shutdown', self._stop_training_controller)]
        actions.extend(('sampler controller shutdown', partial(self._stop_sampler_controller, controller))
                       for controller in tuple(self._sampler_controllers))
        if self._serve_started:
            actions.append(('gateway discovery cleanup', self._unpublish_models))
            actions.append(('owned application cleanup', self._delete_owned_applications))
        actions.append(('telemetry flush', flush_telemetry_safely))
        self._run_cleanup(actions, raise_errors=raise_errors)
        self._serve_started = bool(self._owned_applications or self._published_models)

    @staticmethod
    def _wait_for_shutdown() -> None:
        stopped = threading.Event()

        def signal_shutdown(signum, frame):
            logger.info('Received %s', signal.Signals(signum).name)
            stopped.set()

        signal.signal(signal.SIGINT, signal_shutdown)
        signal.signal(signal.SIGTERM, signal_shutdown)
        stopped.wait()

    def launch(self, blocking: bool = True, *, model_name: str | None = None) -> None:
        """Launch applications and their controllers, optionally waiting for shutdown."""
        if not self.config.applications:
            logger.warning('No applications configured')
            return
        from twinkle.server.deployment import apply_ray_serve_patches
        apply_ray_serve_patches()
        self._configure_environment()
        self._init_ray()
        if model_name is not None:
            self._prepare_model_deployment(model_name)
        try:
            self._start_serve()
            for app in self.config.applications:
                self._deploy_application(app)
            if self.config.training_config.models:
                self._deploy_training()
            if model_name is not None:
                self._wait_for_application(f'sampler-{model_name.replace("/", "_")}')
                self._publish_models([model_name])
            else:
                # Plain config launches share the same discovery path: a reused gateway
                # never sees this config's args, so its samplers must be published here
                # too -- otherwise they run but no request can route to them.
                self._publish_models([alias for app in self.config.applications
                                      if app.import_path == 'sampler'
                                      and (alias := extract_model_alias(app.route_prefix, 'sampler'))])
            logger.info('All applications started')
            if not blocking:
                return
            self._wait_for_shutdown()
        except BaseException:
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
    *,
    ray_address: str | None = 'auto',
    persistent_cluster: bool = False,
    model_name: str | None = None,
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
        ray_address=ray_address,
        persistent_cluster=persistent_cluster,
    )
    launcher.launch(blocking=blocking, model_name=model_name)
