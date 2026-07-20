# Copyright (c) ModelScope Contributors. All rights reserved.
"""Embedding training E2E test scaffolding (Twinkle client path).

This module provides the reusable scaffolding for validating the embedding
training pipeline end to end through the Twinkle client:

    set_processor('InputProcessor')
      -> set_loss('InfonceLoss', temperature=..., use_batch=True)
      -> add_metric('EmbeddingMetric', is_training=True)
      -> loop: forward_backward(inputs=mb, task='embedding') + clip_grad_and_step(...)
      -> calculate_metric(is_training=True)

It exposes TWO switchable backend entrypoints (fixtures):

  * ``mock_embedding_model`` — local CPU-only entrypoint backed by
    ``TwinkleCompatMockModel`` (src/twinkle/server/model/backends/mock_model.py).
    Requires NO GPU. Boots an in-process Ray Serve cluster from the CPU-only
    mock server config fixture. Consumed by the local CPU protocol/link validation.

  * ``gpu_embedding_model`` — real transformers model entrypoint. Gated behind
    the ``TWINKLE_TEST_GPU_E2E=1`` environment variable (skipped otherwise) and
    connects to an already-running GPU server (see
    tests/server/start_e2e_server.py). Consumed by the real numeric
    validation cases.

Plus two reusable helpers:

  * ``build_synthetic_contrastive_dataset(...)`` — builds a small synthetic
    anchor/positive contrastive-learning dataset (even number of samples, with
    ``labels = [1, 0, 1, 0, ...]`` anchor/positive semantics) in the embedding
    pooling input format consumed by ``InputProcessor``.
  * ``run_embedding_training(...)`` — issues the "verification
    call sequence" against a client model and returns the observed losses and
    the final metric result.

==============================================================================
CONDA ENVIRONMENT REQUIREMENT (MANDATORY)
==============================================================================
ALL test cases in this file MUST be run inside the ``twinkle`` conda env, e.g.:

    conda run -n twinkle pytest tests/server/test_embedding_e2e.py -v

The local, CPU-only mock cases run without a GPU and must pass in
that environment:

    conda run -n twinkle pytest tests/server/test_embedding_e2e.py -v -k mock

The GPU-gated cases additionally require ``TWINKLE_TEST_GPU_E2E=1``
and a running GPU server; they are skipped automatically when the variable is
unset:

    TWINKLE_TEST_GPU_E2E=1 conda run -n twinkle pytest tests/server/test_embedding_e2e.py -v -k gpu
==============================================================================
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is importable for both pytest and direct execution.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

# Reuse the existing GPU e2e helpers (server URL, session init, real model id).
from tests.server.integration.e2e_helpers import (
    BASE_URL,
    MODEL_ID,
    init_twinkle_client_session,
    log,
    wait_for_server,
)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Local CPU-only mock backend model id (see tests/server/fixtures/server_config_mock.yaml).
MOCK_MODEL_ID = 'mock-model'

# InfoNCE temperature used by the verification call sequence.
DEFAULT_TEMPERATURE = 0.07

# Default synthetic-dataset shape (kept tiny for CPU speed).
DEFAULT_NUM_PAIRS = 4
DEFAULT_SEQ_LEN = 8
DEFAULT_VOCAB_SIZE = 32

# Number of training steps used by the default verification loop.
DEFAULT_TRAIN_STEPS = 4


def gpu_e2e_enabled() -> bool:
    """Return True when GPU e2e tests are explicitly enabled."""
    return os.environ.get('TWINKLE_TEST_GPU_E2E', '0') == '1'


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic contrastive-learning dataset
# ═══════════════════════════════════════════════════════════════════════════

def build_synthetic_contrastive_dataset(
    num_pairs: int = DEFAULT_NUM_PAIRS,
    *,
    seq_len: int = DEFAULT_SEQ_LEN,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Build a small synthetic anchor/positive contrastive dataset.

    Produces ``2 * num_pairs`` samples interleaved as
    ``[anchor_0, positive_0, anchor_1, positive_1, ...]`` so that the per-sample
    ``labels`` scalar carries the ``[1, 0, 1, 0, ...]`` anchor/positive semantics
    expected by embedding pooling (anchor -> 1, positive -> 0). Each sample is a
    feature dict in the embedding pooling input format consumed by
    ``InputProcessor`` (``input_ids`` + ``attention_mask`` + ``labels``).

    Anchor and its positive within a pair share correlated token content so a
    real InfoNCE loss has a meaningful contrastive signal to learn from, while a
    fixed ``seed`` keeps the dataset deterministic across runs.

    Args:
        num_pairs: Number of anchor/positive pairs (result has ``2*num_pairs`` samples).
        seq_len: Token sequence length per sample.
        vocab_size: Upper bound (exclusive) for synthetic token ids.
        seed: RNG seed for deterministic generation.

    Returns:
        A list of feature dicts of length ``2 * num_pairs``.
    """
    if num_pairs < 1:
        raise ValueError(f'num_pairs must be >= 1, got {num_pairs}')
    if seq_len < 1:
        raise ValueError(f'seq_len must be >= 1, got {seq_len}')

    import numpy as np

    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []
    for pair_idx in range(num_pairs):
        # Shared "concept" tokens make anchor/positive semantically correlated.
        base = rng.integers(low=1, high=vocab_size, size=seq_len)
        # Anchor: the base sequence.
        anchor_ids = base.copy()
        # Positive: perturb a couple of positions so it is similar but not identical.
        positive_ids = base.copy()
        n_perturb = max(1, seq_len // 4)
        perturb_pos = rng.choice(seq_len, size=n_perturb, replace=False)
        positive_ids[perturb_pos] = rng.integers(low=1, high=vocab_size, size=n_perturb)

        samples.append(_make_embedding_feature(anchor_ids.tolist(), label=1))
        samples.append(_make_embedding_feature(positive_ids.tolist(), label=0))

    return samples


def _make_embedding_feature(input_ids: List[int], *, label: int) -> Dict[str, Any]:
    """Build a single embedding pooling feature dict.

    ``labels`` is a single scalar per sample (``[label]``) — this mirrors the
    anchor/positive grouping used by the bare-library embedding example
    (cookbook/exp/embedding/train_embedding_full_ddp.py) where anchors carry
    ``labels=[1]`` and positives carry ``labels=[0]``.
    """
    seq_len = len(input_ids)
    return {
        'input_ids': list(input_ids),
        'attention_mask': [1] * seq_len,
        'labels': [int(label)],
    }


def iter_minibatches(dataset: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """Split a dataset into contiguous minibatches.

    ``batch_size`` should be even so anchor/positive pairs stay together inside
    each minibatch (InfoNCE assumes paired samples within a batch).
    """
    if batch_size < 2 or batch_size % 2 != 0:
        raise ValueError(f'batch_size must be a positive even number, got {batch_size}')
    return [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]


# ═══════════════════════════════════════════════════════════════════════════
# Verification call sequence
# ═══════════════════════════════════════════════════════════════════════════

def configure_embedding_adapter(
    model: Any,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    use_batch: bool = True,
    hard_negatives: Optional[int] = None,
) -> None:
    """Apply the embedding-training configuration steps to a client model.

    Order matches the verification call sequence:
        set_processor('InputProcessor')
        -> set_loss('InfonceLoss', temperature=..., use_batch=True)
        -> add_metric('EmbeddingMetric', is_training=True)

    Note: ``TransformersEmbeddingPatch`` is applied/rolled back automatically
    inside ``forward`` via ``_resolve_task_context`` when ``task='embedding'`` is
    passed, so there is intentionally no ``apply_patch(...)`` call here.
    """
    model.set_processor('InputProcessor')
    model.set_loss('InfonceLoss', temperature=temperature, use_batch=use_batch, hard_negatives=hard_negatives)
    model.add_metric('EmbeddingMetric', is_training=True)


def extract_loss(fwd_bwd_response: Any) -> float:
    """Best-effort extraction of the scalar loss from a forward_backward response.

    Handles both backend shapes:
      * real transformers backend: ``result`` is a ModelOutput/dict with a
        ``'loss'`` key (see TransformersModel.forward_backward).
      * mock backend: ``result`` is ``[records, loss]`` (see
        TwinkleCompatMockModel.forward_backward).
    """
    result = getattr(fwd_bwd_response, 'result', fwd_bwd_response)
    if isinstance(result, dict) and 'loss' in result:
        return float(result['loss'])
    if isinstance(result, (list, tuple)) and result and isinstance(result[-1], (int, float)):
        return float(result[-1])
    raise AssertionError(f'Could not extract loss from forward_backward response: {result!r}')


def run_embedding_training(
    model: Any,
    minibatches: List[List[Dict[str, Any]]],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_grad_norm: float = 1.0,
    configure: bool = True,
) -> Dict[str, Any]:
    """Run the design-document embedding-training verification call sequence.

    Steps:
        (optional) set_processor -> set_loss -> add_metric
        loop over minibatches: forward_backward(task='embedding') + clip_grad_and_step
        calculate_metric(is_training=True)

    Args:
        model: A configured client model (mock or GPU entrypoint).
        minibatches: List of minibatches, each a list of embedding feature dicts.
        temperature: InfoNCE temperature (used only when ``configure=True``).
        max_grad_norm: Grad clipping threshold passed to ``clip_grad_and_step``.
        configure: When True, apply ``configure_embedding_adapter`` first.

    Returns:
        Dict with keys ``losses`` (list[float]) and ``metric`` (final metric result).
    """
    if configure:
        configure_embedding_adapter(model, temperature=temperature)

    losses: List[float] = []
    for step, mb in enumerate(minibatches):
        fwd_bwd = model.forward_backward(inputs=mb, task='embedding')
        loss = extract_loss(fwd_bwd)
        losses.append(loss)
        model.clip_grad_and_step(max_grad_norm=max_grad_norm)
        log(f'[embedding step {step}] loss={loss:.6f}')

    metric = model.calculate_metric(is_training=True)
    metric_result = getattr(metric, 'result', metric)
    return {'losses': losses, 'metric': metric_result}


# ═══════════════════════════════════════════════════════════════════════════
# Client model builder (shared by both entrypoints)
# ═══════════════════════════════════════════════════════════════════════════

def build_embedding_client_model(model_id: str, *, adapter_name: str = 'emb_adapter') -> Any:
    """Create a ``MultiLoraTransformersModel`` with a LoRA adapter for embedding.

    The InfoNCE loss / EmbeddingMetric / processor configuration is applied
    separately by ``configure_embedding_adapter`` so that callers can inspect or
    reorder the verification call sequence.
    """
    from peft import LoraConfig
    from twinkle_client.model import MultiLoraTransformersModel

    model = MultiLoraTransformersModel(model_id=model_id)
    model.add_adapter_to_model(adapter_name, LoraConfig(target_modules='all-linear'))
    model.set_template('Qwen3_5Template')
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Local CPU-only mock server harness
# ═══════════════════════════════════════════════════════════════════════════

class MockEmbeddingServerHarness:
    """Boots the CPU-only mock server in-process via Ray Serve.

    Mirrors the proven boot sequence in
    tests/server/integration/test_mock_mode_startup.py: it manages its own local
    Ray head node (so it does not depend on the session-scoped Ray fixture),
    starts Ray Serve on a randomized port, and runs every application declared in
    the CPU-only mock server config fixture (server + mock model + mock sampler).

    No GPU is required.
    """

    READY_BUDGET_SECONDS = 60.0
    RAY_NODE_CPUS = 8

    def __init__(self) -> None:
        self.host = '127.0.0.1'
        self.port = 18100 + (os.getpid() % 800)
        self.base_url = f'http://{self.host}:{self.port}'
        self._started = False

    # ----- Ray lifecycle ------------------------------------------------- #

    @staticmethod
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
            raise RuntimeError(f'ray {" ".join(args)} failed ({result.returncode}):\n{result.stdout}')

    def start(self) -> str:
        """Boot Ray + Serve + all mock apps; return the base URL when ready."""
        import ray
        from ray import serve

        from tests.server.fixtures import MOCK_SERVER_CONFIG
        from twinkle.server.config import ServerConfig
        from twinkle.server.gateway import build_gateway_app
        from twinkle.server.model import build_model_app
        from twinkle.server.sampler import build_sampler_app

        cfg = ServerConfig.from_yaml(MOCK_SERVER_CONFIG)

        persistence_env: Dict[str, str] = {}
        if cfg.persistence is not None:
            persistence_env = cfg.persistence.to_env_vars()
            for k, v in persistence_env.items():
                os.environ[k] = v

        if ray.is_initialized():
            ray.shutdown()
        self._run_ray_command('stop', '--force')
        self._run_ray_command(
            'start', '--head', '--port=0', f'--num-cpus={self.RAY_NODE_CPUS}',
            '--num-gpus=0', '--include-dashboard=false', '--disable-usage-stats')
        ray.init(
            address='auto',
            runtime_env={'env_vars': persistence_env} if persistence_env else None,
        )
        self._started = True

        serve.start(http_options={'host': self.host, 'port': self.port})
        builders = {
            'server': build_gateway_app,
            'model': build_model_app,
            'sampler': build_sampler_app,
        }
        for app_spec in cfg.applications:
            builder = builders[app_spec.import_path]
            args = {k: v for k, v in dict(app_spec.args).items() if v is not None}
            if app_spec.import_path == 'server':
                http_opts = cfg.http_options.model_dump()
                http_opts['host'] = self.host
                http_opts['port'] = self.port
                args.setdefault('http_options', http_opts)
            deploy_options: Dict[str, Any] = {'ray_actor_options': {'num_cpus': 0.1}}
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

        self._wait_until_healthy(serve, self.READY_BUDGET_SECONDS)
        return self.base_url

    def _wait_until_healthy(self, serve_module: Any, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            status = serve_module.status()
            last = {name: app.status for name, app in status.applications.items()}
            if last and all(s == 'RUNNING' for s in last.values()):
                return
            time.sleep(0.5)
        raise TimeoutError(f'Mock server not RUNNING within {timeout}s: {last}')

    def stop(self) -> None:
        if not self._started:
            return
        try:
            import ray
            from ray import serve
            try:
                serve.shutdown()
            except Exception:
                pass
            try:
                ray.shutdown()
            except Exception:
                pass
        finally:
            try:
                self._run_ray_command('stop', '--force')
            except Exception:
                pass
            self._started = False


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: the two switchable backend entrypoints
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def mock_embedding_model():
    """Local CPU-only mock entrypoint (no GPU required).

    Boots the mock server in-process and yields a client model configured with a
    LoRA adapter, ready for the embedding verification call sequence. Consumed by
    the local CPU protocol/link validation cases.
    """
    harness = MockEmbeddingServerHarness()
    base_url = harness.start()
    try:
        from twinkle_client import init_twinkle_client
        init_twinkle_client(base_url=base_url, api_key='EMPTY_TOKEN')
        model = build_embedding_client_model(MOCK_MODEL_ID)
        yield model
    finally:
        harness.stop()


@pytest.fixture(scope='module')
def gpu_embedding_model():
    """GPU real-model entrypoint (gated by TWINKLE_TEST_GPU_E2E=1).

    Connects to an already-running GPU server (see start_e2e_server.py) and
    yields a client model configured with a LoRA adapter. Consumed by tasks
    4.3-4.5. Skipped automatically when GPU e2e is not enabled.
    """
    if not gpu_e2e_enabled():
        pytest.skip('Set TWINKLE_TEST_GPU_E2E=1 to run real GPU embedding E2E tests (requires running server)')

    wait_for_server()
    init_twinkle_client_session()
    model = build_embedding_client_model(MODEL_ID)
    yield model


# ═══════════════════════════════════════════════════════════════════════════
# Smoke tests — verify the scaffolding collects/imports and helpers behave.
# (Real protocol/numeric assertions live in tasks 4.2-4.5.)
# ═══════════════════════════════════════════════════════════════════════════

def test_synthetic_dataset_shape_and_labels():
    """Synthetic dataset is even-length with [1,0,1,0,...] anchor/positive labels."""
    num_pairs = 3
    dataset = build_synthetic_contrastive_dataset(num_pairs, seq_len=6, vocab_size=16, seed=123)

    assert len(dataset) == 2 * num_pairs
    labels = [sample['labels'][0] for sample in dataset]
    assert labels == [1, 0] * num_pairs

    for sample in dataset:
        assert set(sample.keys()) == {'input_ids', 'attention_mask', 'labels'}
        assert len(sample['input_ids']) == 6
        assert len(sample['attention_mask']) == 6
        assert all(m == 1 for m in sample['attention_mask'])


def test_synthetic_dataset_is_deterministic():
    """Same seed -> identical dataset (keeps mock/GPU comparisons reproducible)."""
    a = build_synthetic_contrastive_dataset(2, seq_len=5, seed=7)
    b = build_synthetic_contrastive_dataset(2, seq_len=5, seed=7)
    assert a == b


def test_iter_minibatches_keeps_pairs_together():
    """Minibatching requires an even batch size and preserves order."""
    dataset = build_synthetic_contrastive_dataset(4, seq_len=4, seed=1)
    batches = iter_minibatches(dataset, batch_size=4)
    assert [len(b) for b in batches] == [4, 4]
    assert batches[0] + batches[1] == dataset

    with pytest.raises(ValueError):
        iter_minibatches(dataset, batch_size=3)


def test_extract_loss_handles_both_backend_shapes():
    """extract_loss understands mock ([records, loss]) and real ({'loss': ...})."""

    class _Resp:
        def __init__(self, result):
            self.result = result

    # Mock backend shape: [records, loss]
    assert extract_loss(_Resp([[{'logprobs': [0.1]}], 0.42])) == pytest.approx(0.42)
    # Real backend shape: dict with 'loss'
    assert extract_loss(_Resp({'loss': 1.5, 'logits': None})) == pytest.approx(1.5)

    with pytest.raises(AssertionError):
        extract_loss(_Resp({'no_loss_here': 1}))


def test_gpu_fixture_gating_env_flag():
    """The GPU gate reflects TWINKLE_TEST_GPU_E2E without side effects."""
    assert gpu_e2e_enabled() == (os.environ.get('TWINKLE_TEST_GPU_E2E', '0') == '1')


# ═══════════════════════════════════════════════════════════════════════════
# Local CPU protocol/link validation against the mock backend.
#
# These cases boot the CPU-only mock server (module-scoped ``mock_embedding_model``
# fixture — booted ONCE, never rebooted per example) and drive the full embedding
# verification call sequence over HTTP:
#
#     set_processor('InputProcessor')
#       -> set_loss('InfonceLoss', temperature=..., use_batch=True)
#       -> add_metric('EmbeddingMetric', is_training=True)
#       -> forward_backward(inputs=mb, task='embedding')
#       -> calculate_metric(is_training=True)
#
# They assert each HTTP request/response hop is well-formed, that ``task='embedding'``
# is transmitted through the protocol layer to the mock backend without protocol
# errors, and that the loss returned by
# ``forward_backward(task='embedding')`` is a finite number (math.isfinite: not NaN,
# not Inf). Here the mock loss layer validates the numeric-finiteness contract at the
# protocol-link level; real-model numeric semantics live in the GPU-gated cases.
#
# Run locally (no GPU) via:
#     conda run -n twinkle pytest tests/server/test_embedding_e2e.py -v -k mock
# ═══════════════════════════════════════════════════════════════════════════

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def test_mock_embedding_protocol_link_full_sequence(mock_embedding_model):
    """Full ordered call sequence runs over HTTP against the mock backend.

    Validates that every hop of the verification call sequence
    (set_processor -> set_loss -> add_metric -> forward_backward(task='embedding')
    -> calculate_metric) completes without any protocol-layer exception, that
    ``task='embedding'`` is accepted through the /twinkle/* protocol and reaches
    the mock backend, and that every returned loss is finite.
    """
    model = mock_embedding_model

    dataset = build_synthetic_contrastive_dataset(DEFAULT_NUM_PAIRS, seq_len=DEFAULT_SEQ_LEN, seed=0)
    minibatches = iter_minibatches(dataset, batch_size=2 * DEFAULT_NUM_PAIRS)

    result = run_embedding_training(model, minibatches)

    losses = result['losses']
    # One loss per minibatch, and the sequence actually issued forward_backward calls.
    assert len(losses) == len(minibatches)
    assert losses, 'expected at least one forward_backward loss'

    # Every embedding forward_backward loss is a finite number.
    for step, loss in enumerate(losses):
        assert math.isfinite(loss), f'loss at step {step} is not finite: {loss!r}'

    # calculate_metric(is_training=True) returned a well-formed metric mapping.
    metric = result['metric']
    assert isinstance(metric, dict), f'expected metric dict, got {type(metric)!r}'


def test_mock_embedding_task_embedding_is_passed_through(mock_embedding_model):
    """``task='embedding'`` traverses the HTTP protocol layer to the mock backend.

    The mock backend intentionally ignores ``task`` semantically (it only exercises
    dispatch), so we assert the protocol contract instead: a ``forward_backward``
    carrying ``task='embedding'`` is accepted end-to-end and returns a well-formed
    response whose extracted loss is finite. This confirms the extra kwarg is
    transported (via ``ForwardRequest.model_extra``) rather than rejected by the
    /twinkle/forward_backward endpoint.
    """
    model = mock_embedding_model
    configure_embedding_adapter(model)

    dataset = build_synthetic_contrastive_dataset(2, seq_len=DEFAULT_SEQ_LEN, seed=42)
    minibatch = dataset  # single minibatch of 4 samples (2 pairs)

    response = model.forward_backward(inputs=minibatch, task='embedding')

    # Response is the ForwardBackwardResponse pydantic model with a ``result`` field.
    assert hasattr(response, 'result'), f'malformed forward_backward response: {response!r}'
    loss = extract_loss(response)
    assert math.isfinite(loss), f'forward_backward(task="embedding") loss not finite: {loss!r}'


@settings(
    max_examples=12,
    deadline=None,
    # ``mock_embedding_model`` is module-scoped (booted once and reused across all
    # generated examples); suppress the health check so hypothesis does not object
    # to the shared fixture and — critically — does NOT reboot the server per example.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    num_pairs=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=2, max_value=12),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_mock_embedding_loss_is_finite_property(mock_embedding_model, num_pairs, seq_len, seed):
    """forward_backward(task='embedding') loss is always finite.

    Varied contrastive dataset shapes (num_pairs, seq_len) and seeds are generated
    within a SINGLE module-scoped mock server instance — the server is never
    rebooted per example. For every generated minibatch, the loss returned by the
    mock backend over the HTTP protocol link must be a finite number
    (math.isfinite: not NaN, not Inf).
    """
    model = mock_embedding_model
    # Idempotent no-op configuration on the mock backend; keeps the case self-contained.
    configure_embedding_adapter(model)

    dataset = build_synthetic_contrastive_dataset(num_pairs, seq_len=seq_len, seed=seed)
    # 2*num_pairs is a positive even batch size, so every anchor/positive pair stays
    # together inside a single minibatch (InfoNCE assumes paired samples per batch).
    minibatches = iter_minibatches(dataset, batch_size=2 * num_pairs)

    for step, mb in enumerate(minibatches):
        response = model.forward_backward(inputs=mb, task='embedding')
        loss = extract_loss(response)
        assert math.isfinite(loss), (
            f'non-finite loss {loss!r} at step {step} '
            f'(num_pairs={num_pairs}, seq_len={seq_len}, seed={seed})')


# ═══════════════════════════════════════════════════════════════════════════
# Single-GPU real-model loss finiteness (GPU-gated).
#
# This case exercises the REAL transformers model backend through the Twinkle
# client HTTP path (module-scoped ``gpu_embedding_model`` fixture, gated behind
# TWINKLE_TEST_GPU_E2E=1 and an already-running GPU server; skipped otherwise).
# For a non-empty contrastive minibatch, the real-model
# ``forward_backward(task='embedding')`` loss must be a finite number
# (math.isfinite: not NaN, not Inf). Unlike the mock cases (which
# validate the numeric-finiteness contract only at the protocol-link level), this
# asserts finiteness against real InfoNCE numeric semantics on real model weights.
#
# Run on GPU via:
#     TWINKLE_TEST_GPU_E2E=1 conda run -n twinkle pytest \
#         tests/server/test_embedding_e2e.py -v -k gpu
# ═══════════════════════════════════════════════════════════════════════════


def test_gpu_embedding_real_model_loss_is_finite(gpu_embedding_model):
    """Real-model forward_backward(task='embedding') loss is finite.

    Drives the embedding verification call sequence
    (set_processor -> set_loss('InfonceLoss', ...) -> add_metric('EmbeddingMetric')
    -> forward_backward(task='embedding') + clip_grad_and_step) against the REAL
    transformers backend and asserts that every loss returned by
    ``forward_backward(task='embedding')`` over a non-empty contrastive minibatch
    is a finite number (math.isfinite: not NaN, not Inf).

    GPU-gated (TWINKLE_TEST_GPU_E2E=1 + running GPU server): skipped automatically
    on machines without a GPU, so this asserts real numeric semantics only on GPU
    CI while collecting/skipping cleanly locally.
    """
    model = gpu_embedding_model

    dataset = build_synthetic_contrastive_dataset(DEFAULT_NUM_PAIRS, seq_len=DEFAULT_SEQ_LEN, seed=0)
    minibatches = iter_minibatches(dataset, batch_size=2 * DEFAULT_NUM_PAIRS)

    # A non-empty minibatch is required to actually exercise the loss path.
    assert minibatches, 'expected at least one minibatch'
    assert all(mb for mb in minibatches), 'expected every minibatch to be non-empty'

    # Configure the embedding pipeline and attach an optimizer before training:
    # run_embedding_training issues clip_grad_and_step, which requires a set optimizer.
    configure_embedding_adapter(model, temperature=DEFAULT_TEMPERATURE)
    model.set_optimizer('AdamW', lr=1e-4)
    result = run_embedding_training(model, minibatches, configure=False)

    losses = result['losses']
    # One loss per minibatch, and the sequence actually issued forward_backward calls.
    assert len(losses) == len(minibatches)
    assert losses, 'expected at least one forward_backward loss'

    # Every real-model embedding forward_backward loss is finite.
    for step, loss in enumerate(losses):
        assert math.isfinite(loss), f'real-model loss at step {step} is not finite: {loss!r}'


# ═══════════════════════════════════════════════════════════════════════════
# Single-GPU: TransformersEmbeddingPatch auto-rollback.
#
# GPU-gated (TWINKLE_TEST_GPU_E2E=1). Skipped automatically on this GPU-less dev
# machine, and on any environment where the variable is unset.
#
# TransformersEmbeddingPatch is applied
# *and rolled back automatically* inside a single forward call via
# ``_resolve_task_context(model, task='embedding')`` (src/twinkle/model/transformers/
# transformers.py). While the patch is active, ``lm_head`` is swapped for identity
# and a forward hook replaces the model output with per-token hidden states
# (src/twinkle/patch/transformers_emb.py::_output_features_hook), i.e. the "logits"
# would carry the HIDDEN-STATE dimension. After the embedding forward_backward
# returns, the patch MUST be reverted, so the very next ``forward_only`` WITHOUT a
# ``task`` argument (defaults to 'causal_lm') must produce real language-model
# logits whose trailing dimension is the VOCABULARY size — never the leftover
# identity hidden states.
#
# This property depends on the real model's patch/unpatch semantics, which the
# CPU-only mock backend cannot exhibit, so it lives behind the GPU gate.
#
# Run on GPU (with a running GPU server) in the ``twinkle`` conda env:
#     TWINKLE_TEST_GPU_E2E=1 conda run -n twinkle pytest tests/server/test_embedding_e2e.py -v -k gpu
# ═══════════════════════════════════════════════════════════════════════════


def _innermost_dim(nested: Any) -> Optional[int]:
    """Return the length of the innermost (last-axis) list of a nested list.

    ``forward_only`` returns logits that ``to_cpu_safe_output`` has converted from
    a torch tensor of shape ``[B, T, V]`` (or a per-sample list of ``[T, V]``) into
    plain nested Python lists. Descending to the innermost list of scalars yields
    the trailing dimension ``V`` (the vocabulary size for real logits, or the hidden
    size if the embedding patch had leaked). Returns ``None`` when the structure is
    not a nested list (e.g. logits were suppressed to ``None``).
    """
    cur = nested
    while isinstance(cur, list) and cur and isinstance(cur[0], list):
        cur = cur[0]
    return len(cur) if isinstance(cur, list) else None


def _build_causal_sample(seq_len: int = DEFAULT_SEQ_LEN) -> Dict[str, Any]:
    """Build a well-formed causal-LM feature (input_ids/attention_mask/labels aligned).

    Distinct from the embedding features (which carry a single scalar ``labels``):
    here ``labels`` has the same length as ``input_ids`` so a causal ``forward_only``
    can compute logps without shape mismatch, and we can read back real vocab-dim
    logits. Small, deterministic token ids keep it valid for any real tokenizer.
    """
    input_ids = [(i % 7) + 1 for i in range(seq_len)]
    return {
        'input_ids': list(input_ids),
        'attention_mask': [1] * seq_len,
        'labels': list(input_ids),
    }


def test_gpu_embedding_patch_auto_rollback_property(gpu_embedding_model):
    """TransformersEmbeddingPatch auto-rolls back after an embedding step.

    Flow (single real GPU adapter, via the ``gpu_embedding_model`` fixture):

      1. Configure the adapter for embedding training (set_processor -> set_loss
         'InfonceLoss' -> add_metric 'EmbeddingMetric').
      2. Establish a baseline: call ``forward_only(return_logits=True)`` WITHOUT a
         ``task`` argument BEFORE any embedding step. Since the patch is never
         applied outside a ``task='embedding'`` forward, these baseline logits are
         genuine vocabulary-dimension logits — the ground-truth "vocab dim".
      3. Run one real ``forward_backward(inputs=mb, task='embedding')``. This applies
         TransformersEmbeddingPatch for the duration of that forward and must revert
         it on the way out.
      4. Immediately call ``forward_only(return_logits=True)`` again WITHOUT ``task``.

    Assertions:
      * The post-embedding logits exist and are 3D nested lists (a leaked patch
        would instead make ``forward_only`` fail — the feature hook returns only
        ``{'features': ...}`` with no ``logits`` key — or yield the hidden-state
        dimension).
      * The post-embedding logits' trailing dimension equals the baseline vocabulary
        dimension, proving ``lm_head``/the forward hook were restored (not left as
        identity emitting hidden states).
    """
    model = gpu_embedding_model
    configure_embedding_adapter(model)

    causal_sample = _build_causal_sample(seq_len=DEFAULT_SEQ_LEN)
    # Use two identical samples so the baseline/post forward_only can be split across
    # a multi-DP model server (dp>=2); the assertions only inspect the logits' trailing
    # vocab dimension, which is invariant to batch size.
    causal_batch = [causal_sample, causal_sample]

    # (2) Baseline vocab-dim logits with the patch NEVER applied.
    baseline_resp = model.forward_only(inputs=causal_batch, return_logits=True)
    baseline_result = getattr(baseline_resp, 'result', baseline_resp)
    assert isinstance(baseline_result, dict), f'malformed forward_only response: {baseline_result!r}'
    baseline_logits = baseline_result.get('logits')
    assert baseline_logits is not None, 'baseline forward_only returned no logits (return_logits was set)'
    vocab_dim = _innermost_dim(baseline_logits)
    assert isinstance(vocab_dim, int) and vocab_dim > 1, (
        f'baseline logits trailing dim is not a valid vocab size: {vocab_dim!r}')

    # (3) One real embedding step: applies + must auto-rollback the patch.
    emb_dataset = build_synthetic_contrastive_dataset(2, seq_len=DEFAULT_SEQ_LEN, seed=0)
    fwd_bwd = model.forward_backward(inputs=emb_dataset, task='embedding')
    emb_loss = extract_loss(fwd_bwd)
    assert math.isfinite(emb_loss), f'embedding forward_backward loss not finite: {emb_loss!r}'

    # (4) Post-embedding causal forward_only WITHOUT task: patch must be gone.
    post_resp = model.forward_only(inputs=causal_batch, return_logits=True)
    post_result = getattr(post_resp, 'result', post_resp)
    assert isinstance(post_result, dict), (
        f'post-embedding forward_only returned malformed result (patch may have leaked): {post_result!r}')
    post_logits = post_result.get('logits')
    assert post_logits is not None, (
        'post-embedding forward_only returned no logits — the embedding patch likely '
        'did NOT roll back (feature hook suppresses the logits key)')
    post_dim = _innermost_dim(post_logits)

    # Trailing dim is the vocab dim (identical to baseline), NOT the
    # leftover identity hidden-state dim.
    assert post_dim == vocab_dim, (
        f'TransformersEmbeddingPatch did NOT auto-rollback: post-embedding logits '
        f'trailing dim {post_dim!r} != baseline vocab dim {vocab_dim!r} — logits appear '
        f'to reuse identity hidden states from the embedding task')


# ═══════════════════════════════════════════════════════════════════════════
# Single-GPU: bare-library parity + pos_sim upward trend (GPU-gated).
#
# These cases exercise the REAL transformers model backend and assert real
# numeric semantics that the mock backend cannot express, so they live behind
# the GPU gate (module-scoped ``gpu_embedding_model`` fixture — gated by
# TWINKLE_TEST_GPU_E2E=1 and a running GPU server; skipped automatically
# otherwise). Two properties are validated:
#
#   * Twinkle client HTTP path vs. bare-library path parity:
#     the same small synthetic contrastive dataset is trained for the same number
#     of steps through (a) the Twinkle client HTTP path and
#     (b) the bare-library training path used by
#     ``cookbook/exp/embedding/train_embedding_full_ddp.py``. The two losses must
#     stay within the SAME ORDER OF MAGNITUDE (digit-for-digit equality is NOT
#     required).
#
#   * pos_sim upward trend: after several real training steps
#     on a dataset with a clear contrastive signal, ``calculate_metric(is_training=
#     True)`` must report an increasing anchor-positive cosine similarity
#     (``pos_sim``) relative to the untrained baseline.
#
# NOTE on the "bare-library path": the published script
# ``cookbook/exp/embedding/train_embedding_full_ddp.py`` orchestrates an 8-GPU
# online-compression pipeline (Ray, vLLM condenser, real datasets) that cannot be
# executed verbatim inside a single-GPU test. This module instead reproduces the
# script's ESSENTIAL bare-library embedding-training primitives IN-PROCESS with
# the very same core-library classes it uses —
#     TransformersModel/MultiLoraTransformersModel
#       + set_processor(InputProcessor)
#       + set_loss(InfonceLoss, temperature=..., use_batch=True, hard_negatives=None)
#       + set_optimizer('AdamW', ...)
#       + add_metric(EmbeddingMetric, is_training=True)
#       + loop: forward_backward(inputs=mb, task='embedding') + clip_grad_and_step(...)
#       + calculate_metric(is_training=True)
# — running on the SAME synthetic dataset and the SAME number of steps as the HTTP
# path. A LoRA adapter is used (instead of full fine-tuning) purely to keep the
# in-process GPU memory footprint tractable next to the running server; the loss
# numerics being compared come from the identical InfoNCE + embedding-pooling code
# path exercised by the bare script.
#
# Run on GPU via:
#     TWINKLE_TEST_GPU_E2E=1 conda run -n twinkle pytest \
#         tests/server/test_embedding_e2e.py -v -k gpu
# ═══════════════════════════════════════════════════════════════════════════

# Dataset / training shape for the parity + trend cases (kept small for GPU speed
# while still giving InfoNCE a real contrastive signal to learn from).
CMP_NUM_PAIRS = 6          # -> 12 samples
CMP_BATCH_SIZE = 4         # -> 3 minibatches == 3 training steps for BOTH paths
TREND_NUM_PAIRS = 6        # -> 12-sample fixed eval minibatch
TREND_STEPS = 20           # real training steps to expose the pos_sim trend
TREND_LR = 1e-3            # aggressive LR so the trend is visible within few steps


def _metric_dict(metric_response: Any) -> Dict[str, Any]:
    """Normalize a calculate_metric result to a plain dict.

    Handles the client HTTP shape (``CalculateMetricResponse`` with a ``result``
    attribute) and the bare-library local shape (a plain dict returned directly).
    """
    result = getattr(metric_response, 'result', metric_response)
    assert isinstance(result, dict), f'expected metric dict, got {type(result)!r}: {result!r}'
    return result


def _parse_metric_float(metric_response: Any, key: str) -> float:
    """Parse a single float metric value (EmbeddingMetric formats values as strings)."""
    result = _metric_dict(metric_response)
    assert key in result, f'metric {key!r} missing from result: {result!r}'
    return float(result[key])


def _order_of_magnitude(value: float) -> int:
    """Return floor(log10(|value|)); 0 is treated as magnitude 0."""
    if value == 0:
        return 0
    return int(math.floor(math.log10(abs(value))))


def _assert_same_order_of_magnitude(a: float, b: float, *, label: str) -> None:
    """Assert two positive losses share the same order of magnitude.

    "Same order of magnitude" is interpreted as a bounded ratio in [0.1, 10]
    (equivalently, their base-10 exponents differ by at most 1). Digit-for-digit
    equality is intentionally NOT required.
    """
    assert math.isfinite(a) and math.isfinite(b), f'[{label}] non-finite losses: {a!r}, {b!r}'
    assert a > 0 and b > 0, f'[{label}] expected positive InfoNCE losses, got {a!r}, {b!r}'
    ratio = a / b
    assert 0.1 <= ratio <= 10.0, (
        f'[{label}] losses differ by more than one order of magnitude: '
        f'client={a:.6f}, bare={b:.6f}, ratio={ratio:.4f} '
        f'(oom client={_order_of_magnitude(a)}, bare={_order_of_magnitude(b)})')
    log(f'[{label}] same order of magnitude: client={a:.6f}, bare={b:.6f}, ratio={ratio:.4f}')


def run_bare_library_embedding_training(
    minibatches: List[List[Dict[str, Any]]],
    *,
    model_id: str = MODEL_ID,
    temperature: float = DEFAULT_TEMPERATURE,
    lr: float = 1e-4,
    max_grad_norm: float = 1.0,
    adapter_name: str = 'bare_emb_adapter',
) -> Dict[str, Any]:
    """Run the bare-library embedding-training path IN-PROCESS on a single GPU.

    Reproduces the essential core-library primitives used by
    ``cookbook/exp/embedding/train_embedding_full_ddp.py`` (see the module section
    comment for why the full multi-GPU script cannot be executed verbatim):

        model = MultiLoraTransformersModel(model_id=...)
        model.add_adapter_to_model(adapter, LoraConfig(target_modules='all-linear'))
        model.set_processor(InputProcessor)
        model.set_loss(InfonceLoss, temperature=..., use_batch=True, hard_negatives=None)
        model.set_optimizer('AdamW', lr=...)
        model.add_metric(EmbeddingMetric, is_training=True)
        for mb in minibatches:
            model.forward_backward(inputs=mb, task='embedding')
            model.clip_grad_and_step(...)
        model.calculate_metric(is_training=True)

    The model runs in ``local`` mode (the default when ``TWINKLE_MODE`` is unset),
    so the ``@remote_function`` decorators call straight through and every method
    requires an explicit ``adapter_name`` kwarg (unlike the client wrapper which
    tracks it internally).

    Args:
        minibatches: The SAME minibatches trained by the HTTP path (same order/steps).
        model_id: Base model id (defaults to the shared real-model id).
        temperature: InfoNCE temperature (matches the client path).
        lr: AdamW learning rate.
        max_grad_norm: Grad clipping threshold.
        adapter_name: LoRA adapter name for the in-process model.

    Returns:
        Dict with ``losses`` (list[float]) and ``metric`` (final metric dict).
    """
    import gc

    from peft import LoraConfig

    from twinkle.loss import InfonceLoss
    from twinkle.metric import EmbeddingMetric
    from twinkle.model import MultiLoraTransformersModel
    from twinkle.processor import InputProcessor

    model = None
    try:
        model = MultiLoraTransformersModel(model_id=model_id)
        model.add_adapter_to_model(adapter_name, LoraConfig(target_modules='all-linear'))
        model.set_processor(InputProcessor, adapter_name=adapter_name)
        model.set_loss(
            InfonceLoss, temperature=temperature, use_batch=True, hard_negatives=None,
            adapter_name=adapter_name)
        model.set_optimizer(optimizer_cls='AdamW', lr=lr, adapter_name=adapter_name)
        model.add_metric(EmbeddingMetric, is_training=True, adapter_name=adapter_name)

        losses: List[float] = []
        for step, mb in enumerate(minibatches):
            outputs = model.forward_backward(inputs=mb, task='embedding', adapter_name=adapter_name)
            loss = extract_loss(outputs)
            losses.append(loss)
            model.clip_grad_and_step(max_grad_norm=max_grad_norm, adapter_name=adapter_name)
            log(f'[bare embedding step {step}] loss={loss:.6f}')

        metric = model.calculate_metric(is_training=True, adapter_name=adapter_name)
        return {'losses': losses, 'metric': _metric_dict(metric)}
    finally:
        # Free the in-process model's GPU memory so it does not linger next to the
        # running server (the test process holds a full second copy of the weights).
        try:
            del model
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def test_gpu_embedding_loss_matches_bare_library_order_of_magnitude(gpu_embedding_model):
    """Client HTTP path and bare-library path losses match in magnitude.

    Trains the SAME small synthetic contrastive dataset for the SAME number of
    steps through two independent paths:

      * the Twinkle client HTTP path, against the real GPU
        server, and
      * the bare-library training path reproduced in-process from
        ``cookbook/exp/embedding/train_embedding_full_ddp.py``.

    Both paths use a fresh, untrained LoRA adapter, the same InfoNCE temperature,
    the same AdamW learning rate, and the same minibatch sequence, so the observed
    losses are directly comparable. The assertion requires only that the mean
    losses stay within the SAME ORDER OF MAGNITUDE — digit-for-digit equality is
    explicitly NOT required (LoRA weights are randomly initialized independently on
    each path, so the two runs are not bit-identical).

    GPU-gated (TWINKLE_TEST_GPU_E2E=1 + running GPU server): skipped automatically
    on machines without a GPU, so it collects/skips cleanly locally and asserts
    real numeric semantics only on GPU CI.
    """
    # ``gpu_embedding_model`` guarantees the GPU gate + a live server/session; use a
    # dedicated fresh client adapter so the comparison is order-independent of other
    # GPU cases that may have already trained the shared fixture adapter.
    client_model = build_embedding_client_model(MODEL_ID, adapter_name='emb_cmp_adapter')

    dataset = build_synthetic_contrastive_dataset(CMP_NUM_PAIRS, seq_len=DEFAULT_SEQ_LEN, seed=0)
    minibatches = iter_minibatches(dataset, batch_size=CMP_BATCH_SIZE)
    assert minibatches, 'expected at least one minibatch'
    assert all(mb for mb in minibatches), 'expected every minibatch to be non-empty'

    # ---- Client HTTP path -------------------------------------------------- #
    # Configure explicitly (processor + loss + metric) and add an optimizer with a
    # matching LR, then train WITHOUT re-configuring inside run_embedding_training.
    configure_embedding_adapter(client_model, temperature=DEFAULT_TEMPERATURE)
    client_model.set_optimizer('AdamW', lr=1e-4)
    client_result = run_embedding_training(client_model, minibatches, configure=False)
    client_losses = client_result['losses']

    # ---- Bare-library path (in-process, same dataset + steps) -------------- #
    bare_result = run_bare_library_embedding_training(
        minibatches, temperature=DEFAULT_TEMPERATURE, lr=1e-4)
    bare_losses = bare_result['losses']

    # Both paths issued exactly one loss per (identical) minibatch.
    assert len(client_losses) == len(minibatches), (
        f'client produced {len(client_losses)} losses for {len(minibatches)} minibatches')
    assert len(bare_losses) == len(minibatches), (
        f'bare produced {len(bare_losses)} losses for {len(minibatches)} minibatches')
    assert client_losses and bare_losses, 'both paths must produce at least one loss'

    for step, loss in enumerate(client_losses):
        assert math.isfinite(loss), f'client loss at step {step} not finite: {loss!r}'
    for step, loss in enumerate(bare_losses):
        assert math.isfinite(loss), f'bare loss at step {step} not finite: {loss!r}'

    # Same order of magnitude on the mean loss across the identical step sequence.
    client_mean = sum(client_losses) / len(client_losses)
    bare_mean = sum(bare_losses) / len(bare_losses)
    _assert_same_order_of_magnitude(client_mean, bare_mean, label='embedding-loss-parity')


def test_gpu_embedding_pos_sim_increases_after_training(gpu_embedding_model):
    """pos_sim rises after several real training steps.

    Uses a fresh, untrained LoRA adapter on the real GPU model and repeatedly
    trains on a fixed synthetic contrastive minibatch (anchors and their positives
    share correlated tokens, so InfoNCE has a clear signal). ``pos_sim`` (the
    anchor-positive cosine similarity reported by ``EmbeddingMetric``) is measured
    on the SAME fixed minibatch before training and after each step, and must show
    an upward trend: the average of the last few measurements exceeds the average
    of the first few.

    This is a real numeric-semantics property that the mock backend cannot express
    (mock embeddings carry no contrastive signal), hence the GPU gate.

    GPU-gated (TWINKLE_TEST_GPU_E2E=1 + running GPU server): skipped automatically
    on machines without a GPU.
    """
    # Fresh dedicated adapter -> clean untrained baseline, independent of other cases.
    model = build_embedding_client_model(MODEL_ID, adapter_name='emb_trend_adapter')
    configure_embedding_adapter(model, temperature=DEFAULT_TEMPERATURE)
    model.set_optimizer('AdamW', lr=TREND_LR)

    dataset = build_synthetic_contrastive_dataset(TREND_NUM_PAIRS, seq_len=DEFAULT_SEQ_LEN, seed=0)
    eval_mb = dataset  # single fixed minibatch reused for every measurement
    assert eval_mb, 'expected a non-empty eval minibatch'
    # Sanity: the dataset must contain anchors (label==1) for pos_sim to be defined.
    assert any(sample['labels'][0] == 1 for sample in eval_mb), 'eval minibatch has no anchors'

    pos_sims: List[float] = []
    for step in range(TREND_STEPS + 1):
        # Measure pos_sim on the CURRENT (pre-step) weights, then take a train step.
        model.forward_backward(inputs=eval_mb, task='embedding')
        metric = model.calculate_metric(is_training=True)
        pos_sims.append(_parse_metric_float(metric, 'pos_sim'))
        model.clip_grad_and_step(max_grad_norm=1.0)

    # Every measurement must be finite and a valid cosine similarity in [-1, 1].
    for step, ps in enumerate(pos_sims):
        assert math.isfinite(ps), f'pos_sim at measurement {step} not finite: {ps!r}'
        assert -1.0001 <= ps <= 1.0001, f'pos_sim at measurement {step} out of range: {ps!r}'

    # Upward trend: average of the last 3 measurements exceeds the first 3
    # (mirrors the robust first-avg/last-avg comparison used elsewhere in the
    # e2e helpers, tolerating per-step noise).
    assert len(pos_sims) >= 6, f'need >=6 pos_sim measurements, got {len(pos_sims)}'
    first_avg = sum(pos_sims[:3]) / 3
    last_avg = sum(pos_sims[-3:]) / 3
    assert last_avg > first_avg, (
        f'pos_sim did NOT increase after training: '
        f'first_3_avg={first_avg:.4f} >= last_3_avg={last_avg:.4f} '
        f'(full trace: {[round(p, 4) for p in pos_sims]})')
    log(f'[embedding pos_sim trend] {first_avg:.4f} -> {last_avg:.4f} '
        f'(baseline={pos_sims[0]:.4f}, final={pos_sims[-1]:.4f})')
