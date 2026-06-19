from __future__ import annotations

import os
import pytest
import uuid

EXPECTED_BASE_MODEL = 'Qwen/Qwen3.6-27B'


def _online_enabled() -> bool:
    return os.environ.get('TWINKLE_NOTEBOOK_ONLINE') == '1'


def _server_root_url() -> str:
    base_url = os.environ.get('TWINKLE_NOTEBOOK_BASE_URL', 'http://www.modelscope.cn/twinkle').rstrip('/')
    if base_url.endswith('/api/v1'):
        base_url = base_url[:-len('/api/v1')]
    return base_url


def _token() -> str:
    token = os.environ.get('MODELSCOPE_TOKEN') or os.environ.get('TWINKLE_SERVER_TOKEN')
    if not token:
        pytest.skip('Set MODELSCOPE_TOKEN or TWINKLE_SERVER_TOKEN for online training smoke tests')
    return token


def _tiny_cross_entropy_datum():
    import numpy as np
    from tinker import types

    tokens = [1, 2, 3, 4]
    labels = np.asarray(tokens, dtype=np.int64)
    weights = np.ones_like(labels, dtype=np.float32)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            'target_tokens': types.TensorData.from_numpy(labels),
            'weights': types.TensorData.from_numpy(weights),
        },
    )


def _future_result(future, label: str, timeout: int = 180):
    print(f'{label}...')
    result = future.result(timeout=timeout)
    print(f'{label}: ok')
    return result


@pytest.mark.skipif(not _online_enabled(), reason='Set TWINKLE_NOTEBOOK_ONLINE=1 to hit the live service')
def test_online_tinker_sft_training_smoke():
    from tinker import ServiceClient, types

    from twinkle import init_tinker_client

    init_tinker_client()
    client = ServiceClient(base_url=_server_root_url(), api_key=_token())
    run_id = f'sft-smoke-{uuid.uuid4().hex[:8]}'

    training = client.create_lora_training_client(
        base_model=EXPECTED_BASE_MODEL,
        rank=8,
        user_metadata={
            'source': 'tests.cookbook.tinker_sft_smoke',
            'run_id': run_id
        },
    )
    batch = [_tiny_cross_entropy_datum() for _ in range(4)]

    fwdbwd = _future_result(training.forward_backward(batch, loss_fn='cross_entropy'), 'sft.forward_backward')
    assert fwdbwd.loss_fn_outputs

    optim = _future_result(training.optim_step(types.AdamParams(learning_rate=1e-4)), 'sft.optim_step')
    assert optim is not None

    saved = _future_result(training.save_state(run_id, ttl_seconds=600), 'sft.save_state')
    assert saved.path


@pytest.mark.skipif(not _online_enabled(), reason='Set TWINKLE_NOTEBOOK_ONLINE=1 to hit the live service')
def test_online_tinker_dpo_training_smoke():
    from tinker import ServiceClient, types

    from twinkle import init_tinker_client

    init_tinker_client()
    client = ServiceClient(base_url=_server_root_url(), api_key=_token())
    run_id = f'dpo-smoke-{uuid.uuid4().hex[:8]}'

    training = client.create_lora_training_client(
        base_model=EXPECTED_BASE_MODEL,
        rank=8,
        user_metadata={
            'source': 'tests.cookbook.tinker_dpo_smoke',
            'run_id': run_id
        },
    )
    batch = [_tiny_cross_entropy_datum() for _ in range(4)]

    ref = _future_result(
        training.forward(
            batch,
            loss_fn='cross_entropy',
            loss_fn_config={'disable_lora': True},
        ),
        'dpo.reference_forward',
    )
    assert len(ref.loss_fn_outputs) == len(batch)

    for datum, ref_out in zip(batch, ref.loss_fn_outputs):
        datum.loss_fn_inputs['ref_logps'] = ref_out['logprobs']

    fwdbwd = _future_result(
        training.forward_backward(
            batch,
            loss_fn='importance_sampling',
            loss_fn_config={
                'dpo_beta': 0.1,
                'dpo_sft_weight': 1.0,
            },
        ),
        'dpo.forward_backward',
    )
    assert fwdbwd.loss_fn_outputs

    optim = _future_result(training.optim_step(types.AdamParams(learning_rate=1e-4)), 'dpo.optim_step')
    assert optim is not None

    saved = _future_result(training.save_state(run_id, ttl_seconds=600), 'dpo.save_state')
    assert saved.path
