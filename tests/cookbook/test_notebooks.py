from __future__ import annotations

import ast
import json
import os
import pytest
import requests
import time
import uuid
from pathlib import Path
from typing import Any

EXPECTED_BASE_MODEL = 'Qwen/Qwen3.6-27B'
NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / 'notebook'

NOTEBOOKS = {
    'dpo.ipynb': {
        'paths': {'tinker_training', 'tinker_sampling', 'twinkle_model'},
    },
    'multi_modal.ipynb': {
        'paths': {'tinker_sampling', 'twinkle_model'},
    },
    'sample.ipynb': {
        'paths': {'tinker_sampling'},
    },
    'self_cognition.ipynb': {
        'paths': {'tinker_training', 'tinker_sampling', 'twinkle_model'},
    },
    'short_math_grpo.ipynb': {
        'paths': {'tinker_sampling', 'twinkle_model', 'twinkle_sampler'},
    },
}


def _notebook_path(name: str) -> Path:
    return NOTEBOOK_DIR / name


def _code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    return [''.join(cell.get('source') or []) for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']


def _python_source_for_ast(cell_source: str) -> str:
    lines: list[str] = []
    for line in cell_source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(('!', '%')):
            continue
        lines.append(line)
    return '\n'.join(lines)


def _all_code(path: Path) -> str:
    return '\n'.join(_code_cells(path))


@pytest.mark.parametrize('notebook_name', sorted(NOTEBOOKS))
def test_notebook_code_cells_parse(notebook_name: str):
    path = _notebook_path(notebook_name)
    assert path.exists(), f'Missing notebook: {path}'

    for index, source in enumerate(_code_cells(path)):
        python_source = _python_source_for_ast(source)
        if not python_source.strip():
            continue
        try:
            ast.parse(python_source)
        except SyntaxError as exc:
            raise AssertionError(f'{notebook_name} cell {index} does not parse: {exc}') from exc


@pytest.mark.parametrize('notebook_name', sorted(NOTEBOOKS))
def test_notebook_models_match_online_base_model(notebook_name: str):
    source = _all_code(_notebook_path(notebook_name))

    assert EXPECTED_BASE_MODEL in source
    assert 'Qwen/Qwen3.6-35B-A3B' not in source
    assert 'Qwen/Qwen3.5-35B-A3B' not in source


@pytest.mark.parametrize('notebook_name, spec', sorted(NOTEBOOKS.items()))
def test_notebook_expected_client_paths_are_present(notebook_name: str, spec: dict[str, Any]):
    source = _all_code(_notebook_path(notebook_name))

    if 'tinker_training' in spec['paths']:
        assert 'create_lora_training_client' in source
    if 'tinker_sampling' in spec['paths']:
        assert 'create_sampling_client' in source
    if 'twinkle_model' in spec['paths']:
        assert 'MultiLoraTransformersModel' in source
    if 'twinkle_sampler' in spec['paths']:
        assert 'vLLMSampler' in source


def _online_enabled() -> bool:
    return os.environ.get('TWINKLE_NOTEBOOK_ONLINE') == '1'


def _online_config() -> tuple[str, str, int]:
    token = os.environ.get('MODELSCOPE_TOKEN') or os.environ.get('TWINKLE_SERVER_TOKEN')
    if not token:
        pytest.skip('Set MODELSCOPE_TOKEN or TWINKLE_SERVER_TOKEN for online notebook smoke tests')
    base_url = os.environ.get('TWINKLE_NOTEBOOK_BASE_URL', 'http://www.modelscope.cn/twinkle').rstrip('/')
    if not base_url.endswith('/api/v1'):
        base_url = f'{base_url}/api/v1'
    timeout = int(os.environ.get('TWINKLE_NOTEBOOK_TIMEOUT', '45'))
    return base_url, token, timeout


def _headers(token: str, request_id: str | None = None) -> dict[str, str]:
    request_id = request_id or f'notebook-smoke-{uuid.uuid4().hex[:8]}'
    return {
        'Authorization': f'Bearer {token}',
        'Twinkle-Authorization': f'Bearer {token}',
        'x-request-id': request_id,
        'serve_multiplexed_model_id': request_id,
        'Serve-Multiplexed-Model-Id': request_id,
    }


def _post_json(url: str, token: str, payload: dict[str, Any], timeout: int, request_id: str | None = None):
    response = requests.post(url, json=payload, headers=_headers(token, request_id), timeout=timeout)
    response.raise_for_status()
    return response.json()


@pytest.mark.skipif(not _online_enabled(), reason='Set TWINKLE_NOTEBOOK_ONLINE=1 to hit the live service')
def test_online_capabilities_match_notebook_model():
    base_url, token, timeout = _online_config()
    response = requests.get(f'{base_url}/get_server_capabilities', headers=_headers(token), timeout=timeout)
    response.raise_for_status()

    models = [item['model_name'] for item in response.json()['supported_models']]
    assert models == [EXPECTED_BASE_MODEL]


@pytest.mark.skipif(not _online_enabled(), reason='Set TWINKLE_NOTEBOOK_ONLINE=1 to hit the live service')
def test_online_twinkle_model_and_sampler_create_paths():
    base_url, token, timeout = _online_config()

    session = _post_json(
        f'{base_url}/twinkle/create_session',
        token,
        {'metadata': {
            'source': 'tests.cookbook.notebook_smoke'
        }},
        timeout,
    )
    headers = _headers(token)
    headers['X-Twinkle-Session-Id'] = session['session_id']

    for service in ('model', 'sampler'):
        url = f'{base_url}/{service}/{EXPECTED_BASE_MODEL}/twinkle/create'
        response = requests.post(url, json={}, headers=headers, timeout=timeout)
        response.raise_for_status()


@pytest.mark.skipif(not _online_enabled(), reason='Set TWINKLE_NOTEBOOK_ONLINE=1 to hit the live service')
def test_online_tinker_base_model_sampling_without_model_path():
    from tinker import types

    base_url, token, timeout = _online_config()
    request_id = f'notebook-smoke-{uuid.uuid4().hex[:8]}'

    session_body = types.CreateSessionRequest(
        tags=['tests.cookbook'],
        user_metadata={
            'source': 'notebook_smoke'
        },
        sdk_version='tests.cookbook',
    ).model_dump(mode='json')
    session = _post_json(f'{base_url}/create_session', token, session_body, timeout, request_id)

    sampling_session_body = types.CreateSamplingSessionRequest(
        session_id=session['session_id'],
        sampling_session_seq_id=0,
        base_model=EXPECTED_BASE_MODEL,
        model_path=None,
    ).model_dump(mode='json')
    sampling_session = _post_json(
        f'{base_url}/create_sampling_session',
        token,
        sampling_session_body,
        timeout,
        request_id,
    )

    sample_body = types.SampleRequest(
        sampling_session_id=sampling_session['sampling_session_id'],
        seq_id=0,
        num_samples=1,
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        sampling_params=types.SamplingParams(max_tokens=4),
        base_model=EXPECTED_BASE_MODEL,
    ).model_dump(mode='json')
    future = _post_json(f'{base_url}/asample', token, sample_body, timeout, request_id)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        retrieve_body = types.FutureRetrieveRequest(request_id=future['request_id']).model_dump(mode='json')
        result = _post_json(f'{base_url}/retrieve_future', token, retrieve_body, timeout, request_id)
        if result.get('type') != 'try_again':
            assert 'error' not in result, result
            assert result['sequences']
            return
        time.sleep(1)

    raise AssertionError(f'Tinker base-model sampling did not finish within {timeout}s')
