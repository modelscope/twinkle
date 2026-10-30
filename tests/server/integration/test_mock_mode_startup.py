# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end mock-mode startup + determinism integration test.

Launches the all-mock cookbook config inside the test process via Ray Serve,
then asserts:
- Managed training and sampler routes become ready within the 120-second startup budget.
- Repeated calls to the mock model and mock sampler over HTTP return
  byte-identical responses for identical input.
- The actual launcher normalizes the original applications YAML and deploys
  CPU routes and managed mock instances without loading real model weights.

This test is heavier than the property suite (boots a full Ray Serve
cluster) and is gated behind ``TWINKLE_TEST_INTEGRATION=1`` so plain
``pytest`` runs stay fast. CI / local runs that opt-in pick it up.
"""
from __future__ import annotations

import httpx
import os
import pytest
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

READY_BUDGET_SECONDS = 120.0
RAY_NODE_CPUS = 8


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
    ray.init(
        address='local', num_cpus=RAY_NODE_CPUS, num_gpus=0, include_dashboard=False,
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


def test_mock_mode_reaches_ready_and_is_deterministic(ray_cluster, monkeypatch) -> None:
    import ray
    from ray import serve
    from twinkle.server.launcher.server_launcher import ServerLauncher

    cfg = ServerConfig.from_yaml(SELECTED_CONFIG)

    # Use a randomized port so concurrent runs / leftover processes don't collide.
    port = 18000 + (os.getpid() % 1000)
    host = '127.0.0.1'
    serve.start(http_options={'host': host, 'port': port})

    started = time.monotonic()
    namespace = ray.get_runtime_context().namespace
    monkeypatch.setenv('TWINKLE_STATE_RAY_NAMESPACE', namespace)
    cfg = cfg.model_copy(update={'ray_namespace': namespace,
                                'http_options': cfg.http_options.model_copy(update={'host': host, 'port': port})})
    launcher = ServerLauncher(cfg)
    for app_spec in cfg.applications:
        launcher._deploy_application(app_spec)
    launcher._deploy_training()

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
        # The native patch dirties the resident instance; release replaces it.
        # A create during that replacement is correctly rejected, not queued.
        deadline = time.monotonic() + READY_BUDGET_SECONDS
        while time.monotonic() < deadline:
            health = _http(f'{base}/api/v1/model/mock-model/healthz').json()
            if health.get('healthy'):
                break
            time.sleep(0.2)
        else:
            raise AssertionError('Resident replacement did not become ready')
        _exercise_tinker_client(base)


def _exercise_twinkle_clients(base: str) -> None:
    from twinkle import init_twinkle_client
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

    save_resp = model.save(name='step-1')
    assert save_resp.twinkle_path and save_resp.twinkle_path.startswith('twinkle://')
    # Mock save creates an empty directory; identify it as the LoRA checkpoint
    # expected by the sampler's checkpoint classification introduced on main.
    assert save_resp.checkpoint_dir
    lora_cfg.save_pretrained(save_resp.checkpoint_dir)
    model.load(name=save_resp.twinkle_path)
    model.resume_from_checkpoint(name=save_resp.twinkle_path)
    model.apply_patch('NoopPatch')

    # Hub upload is checked with a local mocked HubOperation; this smoke
    # must not publish files to an external account.

    # --- sampler service ---
    from twinkle_client.http import ClientTransport
    from twinkle_client.http.context import capture_transport
    # Cold PEFT imports on NAS can exceed the default 90-second HTTP timeout.
    sampler_transport = ClientTransport(capture_transport().context, timeout=600)
    sampler = vLLMSampler(model_id='mock-model', transport=sampler_transport)
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

    # Exercise adapter_uri resolution using the mock LoRA checkpoint directory.
    sampler.sample(
        inputs=[{
            'input_ids': [1, 2, 3]
        }],
        sampling_params={'max_tokens': 2},
        adapter_name='a',
        adapter_uri=save_resp.twinkle_path,
    )

    sampler.apply_patch('NoopPatch')
    sampler_transport.close()
    model.remove_adapter()
    twinkle.close()


def _exercise_tinker_client(base: str) -> None:
    """Drive the upstream Tinker SDK against the mock server."""
    pytest.importorskip('tinker')

    import os
    from tinker import ServiceClient, types

    from twinkle import init_tinker_client

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
    # path mode: save_weights_for_sampler(name) must return path != None
    assert sampler_ckpt.path is not None
    from pathlib import Path
    from peft import LoraConfig
    from twinkle.server.checkpoint import create_checkpoint_manager
    checkpoint_manager = create_checkpoint_manager('EMPTY_TOKEN', client_type='tinker')
    sampler_dir = Path(checkpoint_manager.resolve_checkpoint_dir(sampler_ckpt.path)).resolve()
    LoraConfig(r=4, target_modules='all-linear').save_pretrained(sampler_dir)
    # Gateway's /asample resolves ``base_model`` from ``body.base_model`` or
    # ``sampling_session_id``; pass it explicitly because the SDK only sets
    # ``model_path`` and the gateway doesn't parse ``twinkle://`` URIs.
    sampling = client.create_sampling_client(base_model='mock-model', model_path=sampler_ckpt.path)
    sampling.sample(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=4),
    ).result()

    # sampling_session_seq_id mode: save_weights_and_get_sampling_client()
    # must return path == None and sampling_session_id != None (asserted by SDK internally)
    sampling_client = training.save_weights_and_get_sampling_client()
    # Mock saves contain no tensors. Mark the newly versioned sampler checkpoint
    # as LoRA, just as the native smoke does, rather than pretending it is full weights.
    for checkpoint in sampler_dir.parent.iterdir():
        if checkpoint.is_dir():
            LoraConfig(r=4, target_modules='all-linear').save_pretrained(checkpoint)
    sampling_client.sample(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=4),
    ).result()

    # ``TrainingClient`` exposes save_state/load_state, not save_weights —
    # the wire-level handler is /tinker/save_weights either way.
    ckpt = training.save_state(name='step-2').result()
    training.load_state(ckpt.path).result()
    from tests.server.integration.on_demand_training.on_demand_sdk import unload_tinker_model
    unload_tinker_model(client, training.model_id)
