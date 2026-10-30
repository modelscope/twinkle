# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ray composition for the sampling controller and its owned driver."""

import asyncio
from copy import deepcopy

from twinkle.server.config.application_spec import SamplerArgs
from .controller import SamplerController


def create_controller_actor():
    import ray

    from .driver import SamplerDriver

    @ray.remote(num_cpus=0.1, max_restarts=-1, max_task_retries=0)
    class SamplerControllerActor(SamplerController):
        def __init__(self, values, namespace, pool_id, options, budgets):
            args = SamplerArgs.model_validate(values)
            super().__init__(args, SamplerDriver(args, namespace, pool_id, options, budgets), budgets)

        async def initialize(self):
            await super().initialize()
            if not self._stopping and self._monitor_task is None:
                self._monitor_task = asyncio.create_task(self._monitor())

        async def shutdown(self):
            await super().shutdown()
            await self.driver.runtime.wait_for_cleanup()

    return SamplerControllerActor


def create_sampler_controller(spec, namespace, budgets, env):
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
    controller = (
        create_controller_actor()
        .options(name=name, namespace=namespace, runtime_env=runtime_env)
        .remote(spec.args.model_dump(), namespace, pool_id, options, budgets)
    )
    return controller, options, pool_id
