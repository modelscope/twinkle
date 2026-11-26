# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ray composition for the sampling controller and its owned driver."""

import asyncio
import logging
from copy import deepcopy

from twinkle_server.config.application_spec import SamplerArgs
from twinkle_server.telemetry.worker_init import ensure_telemetry_initialized, shutdown_telemetry_safely
from .controller import SamplerController

logger = logging.getLogger(__name__.replace('twinkle_server', 'twinkle.server', 1))


def create_controller_actor():
    import ray

    from .driver import SamplerDriver

    @ray.remote(num_cpus=0.1, max_restarts=-1, max_task_retries=0)
    class SamplerControllerActor(SamplerController):
        def __init__(self, values, namespace, pool_id, options, budgets, base_model):
            ensure_telemetry_initialized()
            args = SamplerArgs.model_validate(values)
            driver = SamplerDriver(args, namespace, pool_id, options, budgets, base_model=base_model)
            super().__init__(args, driver, budgets)

        async def initialize(self):
            await super().initialize()
            if not self._stopping and self._monitor_task is None:
                self._monitor_task = asyncio.create_task(self._monitor())

        async def shutdown(self):
            try:
                await super().shutdown()
                await self.driver.runtime.wait_for_cleanup()
            finally:
                await asyncio.to_thread(shutdown_telemetry_safely)

    return SamplerControllerActor


def create_sampler_controller(spec, namespace, budgets, env, *, base_model):
    """Driver-only composition. Each compute instance owns its external allocation."""
    from .controller import sampler_controller_name, sampling_pool_id

    options = deepcopy(spec.deployments[0]) if spec.deployments else {}
    options.pop("name", None)
    options.pop("autoscaling_config", None)
    options["num_replicas"] = 1
    actor_options = options.setdefault("ray_actor_options", {})
    actor_options.setdefault("num_cpus", 0.1)
    runtime_env = actor_options.setdefault("runtime_env", {})
    runtime_env["env_vars"] = {**env, **runtime_env.get("env_vars", {}), "TWINKLE_STATE_RAY_NAMESPACE": namespace}
    pool_id = sampling_pool_id(spec.route_prefix)
    name = sampler_controller_name(namespace, pool_id)
    # Detached: a `--detach` driver exits once Serve reports RUNNING, and a named
    # non-detached actor dies with its owner job ("all references removed including
    # lineage ref count") -- the router's get_actor handle cannot keep it alive.
    # The controller must outlive the driver until `stop_sampler_controller` kills it.
    controller = (
        create_controller_actor()
        .options(name=name, namespace=namespace, lifetime="detached", runtime_env=runtime_env)
        .remote(spec.args.model_dump(), namespace, pool_id, options, budgets, base_model)
    )
    return controller, options, pool_id


def stop_sampler_controller(namespace, route_prefix, *, timeout_s=60):
    """Stop the pool controller named for ``route_prefix`` (``--stop`` on a detached deploy).

    ``serve.delete`` only removes the router application. The detached controller owns
    the sampling instances, engine deployments and placement groups -- its ``shutdown``
    releases them, then the actor itself is killed by name. Returns whether a controller
    was found; the resources default ``shutdown_timeout_s`` bounds the wait.
    """
    import ray

    from .controller import sampler_controller_name, sampling_pool_id

    name = sampler_controller_name(namespace, sampling_pool_id(route_prefix))
    try:
        controller = ray.get_actor(name, namespace=namespace)
    except ValueError:
        return False
    try:
        ray.get(controller.shutdown.remote(), timeout=timeout_s)
    except ray.exceptions.ActorDiedError:
        pass  # already dead; the kill below is then just name bookkeeping
    except ray.exceptions.GetTimeoutError:
        # Keep the owner alive: a timeout does not prove its devices or deployments were released.
        logger.warning('Sampler controller %r shutdown timed out after %ss; its owner is retained for retry.',
                       name, timeout_s)
        raise
    ray.kill(controller, no_restart=True)
    return True
