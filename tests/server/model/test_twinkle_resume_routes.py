import concurrent.futures
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient


class _NoOpProcessPoolExecutor:

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args, **kwargs):
        raise RuntimeError('Process pool is disabled in this test environment.')


concurrent.futures.ProcessPoolExecutor = _NoOpProcessPoolExecutor

if 'tinker' not in sys.modules:
    tinker_module = types.ModuleType('tinker')
    tinker_types_module = types.ModuleType('tinker.types')

    class _TinkerPlaceholder:
        pass

    for name in (
            'CreateModelRequest',
            'TrainingRun',
            'TrainingRunsResponse',
            'Cursor',
            'Checkpoint',
            'CheckpointsListResponse',
            'ParsedCheckpointTinkerPath',
            'WeightsInfoResponse',
    ):
        setattr(tinker_types_module, name, _TinkerPlaceholder)
    tinker_module.types = tinker_types_module
    sys.modules['tinker'] = tinker_module
    sys.modules['tinker.types'] = tinker_types_module

if 'twinkle.server.common' not in sys.modules:
    common_module = types.ModuleType('twinkle.server.common')
    checkpoint_factory_module = types.ModuleType('twinkle.server.common.checkpoint_factory')
    checkpoint_factory_module.create_checkpoint_manager = lambda token, client_type='twinkle': None
    checkpoint_factory_module.create_training_run_manager = lambda token, client_type='twinkle': None
    common_module.checkpoint_factory = checkpoint_factory_module
    sys.modules['twinkle.server.common'] = common_module
    sys.modules['twinkle.server.common.checkpoint_factory'] = checkpoint_factory_module

from twinkle_client.types.checkpoint import ResolvedLoadPath

_HANDLERS_PATH = Path(__file__).resolve().parents[3] / 'src' / 'twinkle' / 'server' / 'model' / 'twinkle_handlers.py'
_HANDLERS_SPEC = importlib.util.spec_from_file_location('twinkle_resume_test_handlers', _HANDLERS_PATH)
handlers = importlib.util.module_from_spec(_HANDLERS_SPEC)
sys.modules[_HANDLERS_SPEC.name] = handlers
_HANDLERS_SPEC.loader.exec_module(handlers)


class _FakeCheckpointManager:

    def resolve_load_path(self, path: str) -> ResolvedLoadPath:
        return ResolvedLoadPath(
            checkpoint_name='ckpt-1',
            checkpoint_dir='D:/resolved/weights',
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


def _build_test_client(monkeypatch):
    management = _FakeModelManagement()
    checkpoint_manager = _FakeCheckpointManager()
    monkeypatch.setattr(handlers, 'create_checkpoint_manager', lambda token, client_type='twinkle': checkpoint_manager)

    app = FastAPI()
    handlers._register_twinkle_routes(app, lambda: management)
    return TestClient(app), management


def _run_remote_resume_case(client: TestClient, *, resume_from_checkpoint, resume_only_model, ignore_data_skip):
    if resume_from_checkpoint is None:
        return None
    if not resume_only_model:
        return client.post('/twinkle/load_training_state', json={'name': resume_from_checkpoint, 'adapter_name': ''})
    if not ignore_data_skip:
        return client.post('/twinkle/read_training_progress', json={'name': resume_from_checkpoint, 'adapter_name': ''})
    return None


def test_case_1_no_resume_call_leaves_remote_resume_helpers_unused(monkeypatch):
    """Case 1: resume_from_checkpoint is None, so no remote resume helper should be called."""
    client, management = _build_test_client(monkeypatch)

    response = _run_remote_resume_case(
        client,
        resume_from_checkpoint=None,
        resume_only_model=False,
        ignore_data_skip=False,
    )

    assert response is None
    management.model.load_training_state.assert_not_called()
    management.model.read_training_progress.assert_not_called()


def test_case_2_resume_only_model_false_uses_load_training_state_route(monkeypatch):
    """Case 2: resume_only_model=False should use load_training_state()."""
    client, management = _build_test_client(monkeypatch)

    response = _run_remote_resume_case(
        client,
        resume_from_checkpoint='twinkle://training_runs/run-1/checkpoints/weights/ckpt-1',
        resume_only_model=False,
        ignore_data_skip=False,
    )

    assert response.status_code == 200
    management.model.load_training_state.assert_called_once_with(
        'D:/resolved/weights/ckpt-1',
        adapter_name=None,
    )
    management.model.read_training_progress.assert_not_called()


def test_case_3_resume_only_model_true_without_ignore_data_skip_reads_progress_only(monkeypatch):
    """Case 3: resume_only_model=True and ignore_data_skip=False should use read_training_progress() only."""
    client, management = _build_test_client(monkeypatch)
    management.model.read_training_progress.return_value = {'cur_step': 6, 'consumed_train_samples': 12}

    response = _run_remote_resume_case(
        client,
        resume_from_checkpoint='twinkle://training_runs/run-1/checkpoints/weights/ckpt-1',
        resume_only_model=True,
        ignore_data_skip=False,
    )

    assert response.status_code == 200
    assert response.json()['result']['consumed_train_samples'] == 12
    management.model.read_training_progress.assert_called_once_with(
        'D:/resolved/weights/ckpt-1',
        adapter_name=None,
    )
    management.model.load_training_state.assert_not_called()


def test_case_4_resume_only_model_true_with_ignore_data_skip_uses_neither_helper(monkeypatch):
    """Case 4: resume_only_model=True and ignore_data_skip=True should call neither remote helper."""
    client, management = _build_test_client(monkeypatch)

    response = _run_remote_resume_case(
        client,
        resume_from_checkpoint='twinkle://training_runs/run-1/checkpoints/weights/ckpt-1',
        resume_only_model=True,
        ignore_data_skip=True,
    )

    assert response is None
    management.model.load_training_state.assert_not_called()
    management.model.read_training_progress.assert_not_called()
