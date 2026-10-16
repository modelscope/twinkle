# Copyright (c) ModelScope Contributors. All rights reserved.
"""Translate model application configuration without creating a second runtime."""
from twinkle.hub.model_alias import extract_model_alias
from .application_spec import ApplicationSpec, ModelArgs
from .training import LegacyExpiration, TrainingModelConfig, TrainingModes


def normalize_model_application(app: ApplicationSpec) -> TrainingModelConfig:
    args = app.args
    assert isinstance(args, ModelArgs)
    if args.modes is not None:
        baseline = ModelArgs(model_id=args.model_id, backend=args.backend, device_group={}, device_mesh={})
        common = {
            'model_id', 'backend', 'modes', 'data_plane_url', 'device_group', 'device_mesh', 'device_type', 'env_vars'
        }
        legacy_fields = {
            key
            for key in type(args).model_fields if key not in common and getattr(args, key) != getattr(baseline, key)
        }
        if legacy_fields or args.device_group or args.device_mesh:
            raise ValueError(f'{app.name}: modes cannot be mixed with single-mode training parameters')
        if app.deployments:
            raise ValueError(f'{app.name}: configure instance replicas inside modes, not deployments')
        modes = args.modes
        legacy_expiration = None
        device_types = {mode.device_mesh.device_type for mode in (modes.lora, modes.full) if mode is not None}
        device_types.discard(None)
        if len(device_types) > 1:
            raise ValueError(f'{app.name}: modes must use the same device type')
        device_type = args.device_type or next(iter(device_types), 'cpu' if args.backend == 'mock' else 'cuda').lower()
        env_vars = args.env_vars
    else:
        if len(app.deployments) > 1:
            raise ValueError(f'{app.name}: a model application must have one deployment configuration')
        deployment = app.deployments[0] if app.deployments else {}
        scaling = deployment.get('autoscaling_config') or {}
        ranks = args.device_group.get('ranks', args.nproc_per_node)
        if isinstance(ranks, list):
            raise ValueError(f'{app.name}: use a device count instead of fixed ranks for pooled training')
        device_type = args.device_type or str(
            args.device_group.get('device_type',
                                  args.device_mesh.get('device_type',
                                                       'cpu' if args.backend == 'mock' else 'cuda'))).lower()
        mesh = {**args.device_mesh, 'device_type': device_type}
        mode = {
            'num_devices':
            ranks,
            'device_mesh':
            mesh,
            'min_replicas':
            args.min_replicas if args.min_replicas is not None else scaling.get('min_replicas',
                                                                                deployment.get('num_replicas', 0)),
            'max_replicas':
            args.max_replicas if args.max_replicas is not None else scaling.get('max_replicas',
                                                                                deployment.get('num_replicas')),
            'idle_timeout_s':
            args.idle_timeout_s,
            'strategy':
            args.strategy,
            'queue_config':
            args.queue_config,
            'max_length':
            args.max_length,
        }
        mode_name = 'full' if args.train_mode == 'full' else 'lora'
        if mode_name == 'lora':
            mode.update(
                max_loras=args.max_loras, max_r=args.max_r, target_modules=args.target_modules, hybrid=args.hybrid)
        modes = TrainingModes.model_validate({mode_name: mode})
        legacy_expiration = LegacyExpiration(
            session_timeout_s=(args.adapter_config or {}).get('adapter_timeout', 1800),
            max_lifetime_s=(args.adapter_config or {}).get('adapter_max_lifetime', 36000))
        env_vars = {
            **(deployment.get('ray_actor_options') or {}).get('runtime_env', {}).get('env_vars', {}),
            **args.env_vars
        }
    name = extract_model_alias(app.route_prefix) or app.name
    return TrainingModelConfig(
        name=name,
        model_id=args.model_id,
        backend=args.backend,
        device_type=device_type,
        modes=modes,
        legacy_expiration=legacy_expiration,
        route_prefix=app.route_prefix.rstrip('/') or '/',
        data_plane_url=args.data_plane_url,
        env_vars=env_vars)
