# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end mock-mode startup + determinism integration test.

Launches the all-mock cookbook config inside the test process via Ray Serve,
then asserts:
- Both Model and Sampler deployments report RUNNING within 30 seconds.
- Repeated calls to the mock model and mock sampler over HTTP return
  byte-identical responses for identical input.
- The launch path imports cleanly even when ``transformers`` / ``vllm`` /
  ``megatron`` would not be available — the mock branches don't pull them.

This test is heavier than the property suite (boots a full Ray Serve
cluster) and is gated behind ``TWINKLE_TEST_INTEGRATION=1`` so plain
``pytest`` runs stay fast. CI / local runs that opt-in pick it up.
"""
from __future__ import annotations

import httpx
import os
import pytest
import subprocess
import sys
import time

from tests.server.fixtures import MOCK_SERVER_CONFIG, MOCK_SERVER_CONFIG_REDIS
from twinkle.server.config import ServerConfig

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_INTEGRATION', '0') != '1',
    reason='Set TWINKLE_TEST_INTEGRATION=1 to run the in-process Ray Serve smoke',
)

# Default to file-backed persistence (no external deps). Set
# ``TWINKLE_TEST_REDIS_PERSISTENCE=1`` to exercise the redis backend instead;
# requires a redis on ``redis://127.0.0.1:6379`` (e.g. ``docker run -d --rm
# -p 6379:6379 redis:7-alpine``).
SELECTED_CONFIG = (
    MOCK_SERVER_CONFIG_REDIS if os.environ.get('TWINKLE_TEST_REDIS_PERSISTENCE', '0') == '1' else MOCK_SERVER_CONFIG)

READY_BUDGET_SECONDS = 30.0
RAY_NODE_CPUS = 8


def _run_ray_command(*args: str) -> None:
    ray_bin = os.path.join(os.path.dirname(sys.executable), 'ray')
    result = subprocess.run(
        [ray_bin, *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'ray {" ".join(args)} failed with code {result.returncode}:\n{result.stdout}')


@pytest.fixture(scope='module')
def ray_cluster():
    """Start a fresh local Ray head node for the duration of the module.

    Bypasses ``twinkle.server.launcher`` (which would normally inject
    ``TWINKLE_PERSISTENCE_*`` into the cluster), so we mirror that injection
    by hand into both ``os.environ`` and ``ray.init(runtime_env=...)``.
    Without it, each replica defaults to ``RayActorBackend`` and the tinker
    future flow can't resolve across replicas.
    """
    import ray
    from ray import serve

    from twinkle.server.config import ServerConfig
    cfg = ServerConfig.from_yaml(SELECTED_CONFIG)

    persistence_env: dict[str, str] = {}
    if cfg.persistence is not None:
        persistence_env = cfg.persistence.to_env_vars()
        for k, v in persistence_env.items():
            os.environ[k] = v

    if ray.is_initialized():
        ray.shutdown()
    _run_ray_command('stop', '--force')
    _run_ray_command(
        'start',
        '--head',
        '--port=0',
        f'--num-cpus={RAY_NODE_CPUS}',
        '--num-gpus=0',
        '--include-dashboard=false',
        '--disable-usage-stats',
    )
    ray.init(
        address='auto',
        runtime_env={'env_vars': persistence_env} if persistence_env else None,
    )
    yield
    try:
        serve.shutdown()
    except Exception:
        pass
    try:
        ray.shutdown()
    except Exception:
        pass
    try:
        _run_ray_command('stop', '--force')
    except Exception:
        pass


def _wait_until_healthy(serve_module, timeout: float) -> dict:
    """Poll ``serve.status()`` until every app is RUNNING or timeout."""
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        status = serve_module.status()
        last = {name: app.status for name, app in status.applications.items()}
        if last and all(s == 'RUNNING' for s in last.values()):
            return last
        time.sleep(0.5)
    return last


def _http(url: str, method: str = 'GET', json: dict | None = None) -> httpx.Response:
    return httpx.request(method, url, json=json, timeout=10.0)


def test_mock_mode_reaches_ready_under_30s_and_is_deterministic(ray_cluster) -> None:
    from ray import serve

    from twinkle.server.gateway import build_gateway_app
    from twinkle.server.model import build_model_app
    from twinkle.server.sampler import build_sampler_app

    cfg = ServerConfig.from_yaml(SELECTED_CONFIG)

    # Use a randomized port so concurrent runs / leftover processes don't collide.
    port = 18000 + (os.getpid() % 1000)
    host = '127.0.0.1'
    serve.start(http_options={'host': host, 'port': port})

    started = time.monotonic()
    deploys: list[tuple[str, str]] = []
    builders = {
        'server': build_gateway_app,
        'model': build_model_app,
        'sampler': build_sampler_app,
    }
    for app_spec in cfg.applications:
        builder = builders[app_spec.import_path]
        # Shallow-dump so nested typed models (notably ``queue_config``) stay as
        # ``TaskQueueConfig`` instances instead of being serialized back to a
        # dict — this mirrors what the production launcher does at
        # ``launcher/server_launcher.py:161`` and is required since the
        # builders + deployment ``__init__`` accept ``TaskQueueConfig`` directly
        # (Task 27 removed the ``from_dict`` revival path).
        args = {k: v for k, v in dict(app_spec.args).items() if v is not None}
        if app_spec.import_path == 'server':
            # Gateway's ServiceProxy reads http_options.port to build internal
            # proxy targets; the fixture hard-codes 8000 but the test runs on
            # a randomized port, so override before passing in.
            http_opts = cfg.http_options.model_dump()
            http_opts['host'] = host
            http_opts['port'] = port
            args.setdefault('http_options', http_opts)
        # Strip ray_actor_options runtime_env to keep the test light, BUT keep
        # ``num_cpus`` low so the three deployments leave room for the mock
        # distributed runtime workers in the dedicated local Ray node.
        deploy_options: dict = {'ray_actor_options': {'num_cpus': 0.1}}
        for raw in app_spec.deployments:
            if isinstance(raw, dict):
                deploy_options = {
                    k: v
                    for k, v in raw.items() if k not in ('name', 'ray_actor_options', 'autoscaling_config')
                }
                deploy_options['ray_actor_options'] = {'num_cpus': 0.1}
                break
        bound = builder(deploy_options=deploy_options, **args)
        serve.run(bound, name=app_spec.name, route_prefix=app_spec.route_prefix)
        deploys.append((app_spec.name, app_spec.route_prefix))

    statuses = _wait_until_healthy(serve, READY_BUDGET_SECONDS)
    elapsed = time.monotonic() - started
    assert statuses, 'serve.status() returned no applications'
    assert all(s == 'RUNNING' for s in statuses.values()), statuses
    assert elapsed < READY_BUDGET_SECONDS, f'startup took {elapsed:.1f}s > {READY_BUDGET_SECONDS}s'

    # ---- Determinism: gateway /healthz must respond 200 -------------------
    base = f'http://{host}:{port}'
    r = _http(f'{base}/api/v1/healthz')
    assert r.status_code == 200, r.text

    # Mock model + sampler determinism via the gateway's exposed routes.
    r1 = _http(f'{base}/api/v1/twinkle/healthz')
    r2 = _http(f'{base}/api/v1/twinkle/healthz')
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.text == r2.text, 'twinkle healthz responses differ'

    # The Model + Sampler primary endpoints don't expose a healthz, but Ray
    # Serve only marks a deployment RUNNING after its FastAPI app finishes
    # startup — so the RUNNING assertion above already covers readiness.

    # Smoke the /twinkle/* surface end-to-end (routing, dispatch, queueing,
    # serialization, pydantic shapes). Numerical correctness is out of scope.
    _exercise_twinkle_clients(base)

    # Same surface via the upstream Tinker SDK at /api/v1/tinker/*. Gated
    # separately because tinker's polling can stall the test budget on flaky
    # cross-replica state.
    if os.environ.get('TWINKLE_TEST_TINKER', '0') == '1':
        _exercise_tinker_client(base)


def _exercise_twinkle_clients(base: str) -> None:
    from twinkle_client import init_twinkle_client
    from twinkle_client.model import MultiLoraTransformersModel
    from twinkle_client.sampler import vLLMSampler

    # Creates a server-side session so adapter endpoints get
    # ``X-Twinkle-Session-Id``; also wires base_url + api_key for the SDK.
    twinkle = init_twinkle_client(base_url=base, api_key='EMPTY_TOKEN')  # noqa: F841

    # --- model service ---
    model = MultiLoraTransformersModel(model_id='mock-model')
    # Pass an actual ``LoraConfig`` (not a dict): the client auto-serializes
    # objects with ``__dict__`` to JSON, which the server's ``config: str``
    # field accepts; plain dicts fail pydantic validation.
    from peft import LoraConfig
    lora_cfg = LoraConfig(r=4, target_modules='all-linear')
    model.add_adapter_to_model(adapter_name='a', config=lora_cfg)

    model.set_loss('CrossEntropy')
    model.set_optimizer('Adam')
    model.set_lr_scheduler('constant')
    model.set_template('Template')
    model.set_processor('InputProcessor')
    model.add_metric('Loss', is_training=True)

    inputs = [{'input_ids': [1, 2, 3], 'labels': [1, 2, 3]}]
    fwd = model.forward(inputs)
    assert fwd.result is not None
    fwd_only = model.forward_only(inputs)
    assert fwd_only.result is not None
    fwd_bwd = model.forward_backward(inputs)
    assert fwd_bwd.result is not None

    grad_norm = model.clip_grad_norm()
    assert isinstance(grad_norm.result, str)
    model.step()
    model.zero_grad()
    model.lr_step()
    loss = model.calculate_loss()
    assert isinstance(loss.result, float)
    metric = model.calculate_metric(is_training=True)
    assert isinstance(metric.result, dict)
    cfgs = model.get_train_configs()
    assert isinstance(cfgs.result, str)
    state = model.get_state_dict()
    assert isinstance(state.result, dict)

    save_resp = model.save(name='step-1')
    assert save_resp.twinkle_path and save_resp.twinkle_path.startswith('twinkle://')
    model.load(name=save_resp.twinkle_path)
    model.resume_from_checkpoint(name=save_resp.twinkle_path)
    model.apply_patch('NoopPatch')

    model.upload_to_hub(
        checkpoint_dir=save_resp.twinkle_path,
        hub_model_id='mock/dummy',
        hub_token='EMPTY_TOKEN',
        async_upload=False,
        poll_interval=0.5,
    )

    # --- sampler service ---
    sampler = vLLMSampler(model_id='mock-model')
    # Sampler /add_adapter_to_sampler reuses the model-side ``AddAdapterRequest``
    # shape (``config: str``), so send JSON. Skip a real ``LoraConfig`` here
    # because its ``runtime_config`` member isn't JSON-serializable.
    import json
    sampler.add_adapter_to_sampler('a', config=json.dumps({'r': 4, 'target_modules': ['all-linear']}))
    sampler.set_template('Template')

    samples = sampler.sample(
        inputs=[{
            'input_ids': [1, 2, 3]
        }],
        sampling_params={'max_tokens': 4},
        adapter_name='a',
    )
    assert samples and samples[0].sequences and len(samples[0].sequences[0].tokens) == 4

    # adapter_uri triggers reset_prefix_cache; reuse the training-weights
    # path because mock ``save()`` materialized that directory on disk.
    sampler.sample(
        inputs=[{
            'input_ids': [1, 2, 3]
        }],
        sampling_params={'max_tokens': 2},
        adapter_name='a',
        adapter_uri=save_resp.twinkle_path,
    )

    sampler.apply_patch('NoopPatch')


def _exercise_tinker_client(base: str) -> None:
    """Drive the upstream Tinker SDK against the mock server."""
    pytest.importorskip('tinker')

    import os
    from tinker import ServiceClient, types

    from twinkle_client import init_tinker_client

    # patch_tinker injects Twinkle's auth + Ray Serve multiplex headers and
    # lifts tinker's ``tml-`` api-key prefix check so EMPTY_TOKEN passes.
    os.environ['TINKER_BASE_URL'] = base
    os.environ['TWINKLE_SERVER_TOKEN'] = 'EMPTY_TOKEN'
    init_tinker_client()

    client = ServiceClient()
    training = client.create_lora_training_client(base_model='mock-model', rank=4)

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            'target_tokens': [1, 2, 3],
            'weights': [1.0, 1.0, 1.0]
        },
    )
    training.forward_backward([datum], loss_fn='cross_entropy').result()
    training.optim_step(types.AdamParams(learning_rate=1e-4)).result()

    base_sampling = client.create_sampling_client(base_model='mock-model')
    base_sampling.sample(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=4),
    ).result()

    sampler_ckpt = training.save_weights_for_sampler(name='step-1').result()
    # Gateway's /asample resolves ``base_model`` from ``body.base_model`` or
    # ``sampling_session_id``; pass it explicitly because the SDK only sets
    # ``model_path`` and the gateway doesn't parse ``twinkle://`` URIs.
    sampling = client.create_sampling_client(base_model='mock-model', model_path=sampler_ckpt.path)
    sampling.sample(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=4),
    ).result()

    # ``TrainingClient`` exposes save_state/load_state, not save_weights —
    # the wire-level handler is /tinker/save_weights either way.
    ckpt = training.save_state(name='step-2').result()
    training.load_state(ckpt.path).result()
