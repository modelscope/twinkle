import concurrent.futures
import importlib.util
import sys
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock


class _NoOpProcessPoolExecutor:

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        raise RuntimeError('Process pool is disabled in this test environment.')


concurrent.futures.ProcessPoolExecutor = _NoOpProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if 'twinkle.server.common.checkpoint_factory' not in sys.modules:
    fake_checkpoint_factory = ModuleType('twinkle.server.common.checkpoint_factory')
    fake_checkpoint_factory.create_checkpoint_manager = lambda *args, **kwargs: None
    fake_checkpoint_factory.create_training_run_manager = lambda *args, **kwargs: None
    sys.modules['twinkle.server.common.checkpoint_factory'] = fake_checkpoint_factory

from twinkle_client.types.checkpoint import ResolvedLoadPath

_HANDLERS_PATH = SRC / 'twinkle' / 'server' / 'model' / 'twinkle_handlers.py'
_HANDLERS_SPEC = importlib.util.spec_from_file_location('twinkle_resume_test_handlers', _HANDLERS_PATH)
handlers = importlib.util.module_from_spec(_HANDLERS_SPEC)
sys.modules[_HANDLERS_SPEC.name] = handlers
_HANDLERS_SPEC.loader.exec_module(handlers)


class _FakeCheckpointManager:

    def __init__(self, checkpoint_dir='./resolved/weights'):
        self._checkpoint_dir = checkpoint_dir

    def resolve_load_path(self, path: str) -> ResolvedLoadPath:
        return ResolvedLoadPath(
            checkpoint_name='ckpt-1',
            checkpoint_dir=self._checkpoint_dir,
            is_twinkle_path=True,
            training_run_id='run-1',
            checkpoint_id='weights/ckpt-1',
        )


class _FakeModelManagement:

    def __init__(self):
        self.model = Mock()

    async def _on_request_start(self, request):
        request.state.request_id = 'req-1'
        return 'token-1'

    def assert_resource_exists(self, adapter_name):
        return None

    async def schedule_task_and_wait(self, task, task_type=''):
        return await task()


def _build_test_client(monkeypatch, checkpoint_manager=None):
    management = _FakeModelManagement()
    checkpoint_manager = checkpoint_manager or _FakeCheckpointManager()
    monkeypatch.setattr(handlers, 'create_checkpoint_manager', lambda token, client_type='twinkle': checkpoint_manager)

    app = FastAPI()
    handlers._register_twinkle_routes(app, lambda: management)
    return TestClient(app), management


def test_load_training_state_route_resolves_checkpoint_path_and_calls_model(monkeypatch):
    client, management = _build_test_client(monkeypatch)

    response = client.post(
        '/twinkle/load_training_state',
        json={
            'name': 'twinkle://training_runs/run-1/checkpoints/weights/ckpt-1',
            'adapter_name': ''
        },
    )

    assert response.status_code == 200
    management.model.load_training_state.assert_called_once_with(
        'resolved/weights/ckpt-1',
        adapter_name=None,
    )
    management.model.read_training_progress.assert_not_called()


def test_load_training_state_route_prefixes_non_empty_adapter_name(monkeypatch):
    client, management = _build_test_client(monkeypatch)

    response = client.post(
        '/twinkle/load_training_state',
        json={
            'name': 'twinkle://training_runs/run-1/checkpoints/weights/ckpt-1',
            'adapter_name': 'adapter-a'
        },
    )

    assert response.status_code == 200
    management.model.load_training_state.assert_called_once_with(
        'resolved/weights/ckpt-1',
        adapter_name='req-1-adapter-a',
    )


def test_load_training_state_route_uses_raw_name_when_checkpoint_dir_missing(monkeypatch):
    client, management = _build_test_client(monkeypatch, checkpoint_manager=_FakeCheckpointManager(checkpoint_dir=None))

    response = client.post(
        '/twinkle/load_training_state',
        json={
            'name': 'local-checkpoint-dir',
            'adapter_name': ''
        },
    )

    assert response.status_code == 200
    management.model.load_training_state.assert_called_once_with(
        'local-checkpoint-dir',
        adapter_name=None,
    )


def test_read_training_progress_route_returns_progress_and_calls_model(monkeypatch):
    client, management = _build_test_client(monkeypatch)
    management.model.read_training_progress.return_value = {'cur_step': 6, 'consumed_train_samples': 12}

    response = client.post(
        '/twinkle/read_training_progress',
        json={
            'name': 'twinkle://training_runs/run-1/checkpoints/weights/ckpt-1',
            'adapter_name': ''
        },
    )

    assert response.status_code == 200
    assert response.json()['result']['consumed_train_samples'] == 12
    management.model.read_training_progress.assert_called_once_with(
        'resolved/weights/ckpt-1',
        adapter_name=None,
    )
    management.model.load_training_state.assert_not_called()
