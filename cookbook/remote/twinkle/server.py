import os
os.environ['RAY_DEBUG'] = '1'
import ray
from omegaconf import OmegaConf
from ray import serve
from twinkle.server import build_processor_app, build_sampler_app, build_model_app, build_server_app

ray.init()
serve.shutdown()
import time
time.sleep(5)

file_dir = os.path.abspath(os.path.dirname(__file__))
config = OmegaConf.load(os.path.join(file_dir, 'server_config.yaml'))

# Start Ray Serve with http_options from config
http_options = OmegaConf.to_container(config.http_options, resolve=True)
serve.start(http_options=http_options)

APP_BUILDERS = {
    'main:model_qwen25_7B': build_model_app,
    # 'main:build_sampler_app': build_sampler_app,
    'main:processor_app': build_processor_app,
    'main:build_server_app': build_server_app,
}

for app_config in config.applications:
    print(f"Starting {app_config.name} at {app_config.route_prefix}...")

    builder = APP_BUILDERS[app_config.import_path]
    args = OmegaConf.to_container(app_config.args, resolve=True) if app_config.args else {}

    deploy_options = {}
    deploy_config = app_config.deployments[0]
    if 'autoscaling_config' in deploy_config:
        deploy_options['autoscaling_config'] = OmegaConf.to_container(deploy_config.autoscaling_config)
    if 'ray_actor_options' in deploy_config:
        deploy_options['ray_actor_options'] = OmegaConf.to_container(deploy_config.ray_actor_options)

    app = builder(
        deploy_options=deploy_options,
        **{k: v for k, v in args.items()}
    )

    serve.run(app, name=app_config.name, route_prefix=app_config.route_prefix)

print("\nAll applications started!")
print("Endpoints:")
for app_config in config.applications:
    print(f"  - http://{config.http_options.host}:{config.http_options.port}{app_config.route_prefix}")

input("\nPress Enter to stop the server...")