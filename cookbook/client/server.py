import os

import ray
from omegaconf import OmegaConf
from ray import serve
from twinkle.server import build_processor_app, build_sampler_app, build_model_app

ray.init()

file_dir = os.path.dirname(__file__)
config = OmegaConf.load(os.path.join(file_dir, 'server_config.yaml'))

APP_BUILDERS = {
    'main:build_model_app': build_model_app,
    'main:build_sampler_app': build_sampler_app,
    'main:build_processor_app': build_processor_app,
}

for app_config in config.applications:
    print(f"Starting {app_config.name} at {app_config.route_prefix}...")

    builder = APP_BUILDERS[app_config.import_path]
    args = OmegaConf.to_container(app_config.args, resolve=True)

    app = builder(
        device_group=args['device_group'],
        device_mesh=args['device_mesh'],
        **{k: v for k, v in args.items() if k not in ('device_group', 'device_mesh')}
    )

    # 应用 deployment 配置
    for deploy_config in app_config.deployments:
        deploy_options = {}
        if 'autoscaling_config' in deploy_config:
            deploy_options['autoscaling_config'] = OmegaConf.to_container(deploy_config.autoscaling_config)
        if 'ray_actor_options' in deploy_config:
            deploy_options['ray_actor_options'] = OmegaConf.to_container(deploy_config.ray_actor_options)

        if deploy_options:
            app = app.options(**deploy_options)

    serve.run(app, name=app_config.name, route_prefix=app_config.route_prefix)

print("\nAll applications started!")
print("Endpoints:")
for app_config in config.applications:
    print(f"  - http://localhost:8000{app_config.route_prefix}")

input("\nPress Enter to stop the server...")