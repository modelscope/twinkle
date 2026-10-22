"""Real filesystem save transactions shared by Tinker and Twinkle handlers."""
import asyncio
from pathlib import Path

import pytest

from twinkle.server.checkpoint import create_checkpoint_manager
from twinkle.server.checkpoint.tinker import TinkerCheckpointManager, TinkerTrainingRunManager
from twinkle.server.checkpoint.twinkle import TwinkleCheckpointManager


@pytest.fixture
def manager(request, monkeypatch, tmp_path):
    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    manager = create_checkpoint_manager('test-token', client_type=getattr(request, 'param', 'twinkle'))
    manager.training_run_manager._write_info('run', {'base_model': 'test-model', 'is_lora': True})
    return manager


def _writer(value, fail=False):
    async def save(name, output_dir):
        checkpoint = Path(output_dir) / name
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / 'weights').write_text(value)
        if fail:
            raise RuntimeError('injected save failure')
        return str(checkpoint)
    return save


@pytest.mark.asyncio
@pytest.mark.parametrize('manager', ['tinker', 'twinkle'], indirect=True)
async def test_named_and_live_versions_have_separate_lifetimes(manager):
    first, first_dir = await manager.save_sampler('run', 'x', _writer('x'))
    second, _ = await manager.save_sampler('run', 'y', _writer('y'))
    live, _ = await manager.save_sampler('run', None, _writer('live-1'))
    latest, _ = await manager.save_sampler('run', None, _writer('live-2'))
    assert first.endswith('/x') and second.endswith('/y')
    assert Path(first_dir, 'weights').read_text() == 'x'
    assert manager.resolve_load_path(first).checkpoint_name == 'x'
    assert manager.get('run', manager.parse_path(live).checkpoint_id) is None
    assert manager.get('run', manager.parse_path(latest).checkpoint_id) is not None
    listed = manager.list_checkpoints('run').checkpoints
    assert {ckpt.checkpoint_id for ckpt in listed} == {
        'sampler_weights/x', 'sampler_weights/y', manager.parse_path(latest).checkpoint_id}
    assert all(ckpt.size_bytes > 0 for ckpt in listed)
    assert 'last_checkpoint' not in manager.training_run_manager._read_info('run')
    assert manager.delete('run', manager.parse_path(latest).checkpoint_id)
    assert Path(first_dir).exists()
    assert manager.delete('run', 'sampler_weights/x')
    assert not Path(first_dir).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('manager', ['tinker', 'twinkle'], indirect=True)
async def test_overwrite_keeps_uri_but_changes_backend_path(manager):
    uri, directory = await manager.save_sampler('run', 'x', _writer('first'))
    previous = Path(manager.parse_adapter_uri(uri)[1])
    after_uri, after_dir = await manager.save_sampler('run', 'x', _writer('second'))
    assert (after_uri, after_dir) == (uri, directory)
    assert Path(directory, 'weights').read_text() == 'second'
    current = Path(manager.parse_adapter_uri(uri)[1])
    assert current != previous and current.is_dir()
    assert not previous.exists()
    assert not list(Path(directory).parent.glob('.pending-*'))


@pytest.mark.asyncio
async def test_same_model_saves_are_serialized(manager):
    active = 0
    max_active = 0
    async def save(name, output_dir):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        await _writer('same')(name, output_dir)
        active -= 1
    other_cls = TwinkleCheckpointManager if isinstance(manager, TinkerCheckpointManager) else TinkerCheckpointManager
    other = other_cls(manager.token, manager.training_run_manager)
    results = await asyncio.gather(*(owner.save_sampler('run', 'x', save) for owner in (manager, other, manager)))
    assert max_active == 1
    assert len(set(results)) == 1


@pytest.mark.asyncio
async def test_cancelling_lock_waiter_does_not_block_following_saves(manager, monkeypatch):
    import twinkle.server.checkpoint.checkpoint_manager as module
    started, finish, waiting_for_lock = asyncio.Event(), asyncio.Event(), asyncio.Event()
    acquire = module.acquire_lock

    def observe_acquire(lock, blocking=False):
        acquired = acquire(lock, blocking=blocking)
        if not acquired:
            waiting_for_lock.set()
        return acquired

    monkeypatch.setattr(module, 'acquire_lock', observe_acquire)

    async def held_save(name, output_dir):
        started.set()
        await finish.wait()
        return await _writer('first')(name, output_dir)

    first = asyncio.create_task(manager.save_sampler('run', 'x', held_save))
    await started.wait()
    waiting = asyncio.create_task(manager.save_sampler('run', 'x', _writer('cancelled')))
    try:
        await asyncio.wait_for(waiting_for_lock.wait(), timeout=2)
        assert not waiting.done()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        finish.set()
        await first
    uri, directory = await manager.save_sampler('run', 'x', _writer('last'))
    assert uri.endswith('/x') and Path(directory, 'weights').read_text() == 'last'


@pytest.mark.asyncio
@pytest.mark.parametrize('manager', ['tinker', 'twinkle'], indirect=True)
async def test_legacy_latest_and_alias_remain_readable(manager):
    latest = Path(manager.get_save_dir('run', is_sampler=True)) / 'latest'
    latest.mkdir(parents=True)
    (latest / 'weights').write_text('legacy')
    manager.save('run', 'latest', is_sampler=True)
    alias = latest.parent / '20261009_000000'
    alias.symlink_to(latest)
    legacy_uri = f'{manager.path_prefix}run/sampler_weights/{alias.name}'
    assert manager.resolve_load_path(legacy_uri).checkpoint_name == alias.name
    await manager.save_sampler('run', 'named', _writer('new'))
    assert alias.exists()
    await manager.save_sampler('run', None, _writer('live'))
    assert not alias.exists()
    assert (latest.parent / 'named').exists()


@pytest.mark.parametrize('name', ['x', None])
@pytest.mark.parametrize('failure', ['backend', 'write', 'flush', 'replace'])
@pytest.mark.asyncio
async def test_save_failure_preserves_run_and_checkpoint(manager, monkeypatch, name, failure):
    import twinkle.server.checkpoint.training_run_manager as run_module
    runs = manager.training_run_manager
    old_uri, directory = await manager.save_sampler('run', name, _writer('first'))
    info_path = runs.get_model_dir('run') / runs.train_run_info_filename
    before = info_path.read_bytes()
    before_checkpoint = Path(directory, 'checkpoint_metadata.json').read_bytes()
    versions = set(Path(directory).parent.joinpath('.versions').iterdir())
    replace = run_module.os.replace

    def fail(*args, **kwargs):
        if failure == 'replace' and Path(args[1]) != info_path:
            return replace(*args, **kwargs)
        if failure == 'write':
            stream = args[1]
            stream.write('{')
            stream.flush()
        raise OSError('injected metadata failure')

    if failure != 'backend':
        target, attr = {'write': (run_module.json, 'dump'), 'flush': (run_module.os, 'fsync'),
                        'replace': (run_module.os, 'replace')}[failure]
        monkeypatch.setattr(target, attr, fail)
    with pytest.raises((OSError, RuntimeError), match='injected'):
        await manager.save_sampler('run', name, _writer('second', fail=failure == 'backend'))
    assert Path(directory, 'weights').read_text() == 'first'
    assert Path(directory, 'checkpoint_metadata.json').read_bytes() == before_checkpoint
    assert info_path.read_bytes() == before
    assert manager.resolve_load_path(old_uri).checkpoint_name == old_uri.rsplit('/', 1)[1]
    assert not list(info_path.parent.glob(f'.{info_path.name}-*'))
    assert not list(Path(directory).parent.glob('.pending-*'))
    assert set(Path(directory).parent.joinpath('.versions').iterdir()) == versions


@pytest.mark.asyncio
async def test_publish_failure_preserves_existing_version(manager, monkeypatch):
    uri, directory = await manager.save_sampler('run', 'x', _writer('first'))
    previous = manager.parse_adapter_uri(uri)[1]
    replace = Path.replace

    def fail(path, target):
        if path.name == '.new':
            raise OSError('injected publish failure')
        return replace(path, target)

    monkeypatch.setattr(Path, 'replace', fail)
    with pytest.raises(OSError, match='publish failure'):
        await manager.save_sampler('run', 'x', _writer('second'))
    assert manager.parse_adapter_uri(uri)[1] == previous
    assert Path(directory, 'weights').read_text() == 'first'
    assert list(Path(directory).parent.joinpath('.versions').iterdir()) == [Path(previous)]
    assert not list(Path(directory).parent.glob('.pending-*'))


@pytest.mark.parametrize('name', ['x', None])
@pytest.mark.asyncio
async def test_cancelling_active_save_keeps_previous_checkpoint(manager, name):
    uri, directory = await manager.save_sampler('run', name, _writer('first'))
    runs = manager.training_run_manager
    info_path = runs.get_model_dir('run') / runs.train_run_info_filename
    before = info_path.read_bytes()
    started = asyncio.Event()

    async def partial_save(name, output_dir):
        checkpoint = Path(output_dir) / name
        checkpoint.mkdir()
        (checkpoint / 'weights').write_text('unfinished')
        started.set()
        await asyncio.Event().wait()

    pending = asyncio.create_task(manager.save_sampler('run', name, partial_save))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert Path(manager.parse_adapter_uri(uri)[1], 'weights').read_text() == 'first'
    assert info_path.read_bytes() == before
    assert not list(Path(directory).parent.glob('.pending-*'))

    # Cancellation must also release the shared lock for the next save.
    _, current = await asyncio.wait_for(manager.save_sampler('run', name, _writer('next')), timeout=2)
    assert Path(current, 'weights').read_text() == 'next'


@pytest.mark.parametrize('name', ['x', None])
@pytest.mark.asyncio
async def test_metadata_failure_restores_legacy_checkpoint_directory(manager, monkeypatch, name):
    checkpoint_name = name or 'latest'
    directory = Path(manager.get_save_dir('run', is_sampler=True)) / checkpoint_name
    directory.mkdir(parents=True)
    (directory / 'weights').write_text('legacy')
    uri = manager.save('run', checkpoint_name, is_sampler=True)
    runs = manager.training_run_manager
    info_path = runs.get_model_dir('run') / runs.train_run_info_filename
    before = info_path.read_bytes()
    checkpoint_before = (directory / 'checkpoint_metadata.json').read_bytes()

    def fail(*args, **kwargs):
        raise OSError('injected metadata failure')

    monkeypatch.setattr(runs, 'update', fail)
    with pytest.raises(OSError, match='metadata failure'):
        await manager.save_sampler('run', name, _writer('new'))
    assert directory.is_dir() and not directory.is_symlink()
    assert Path(manager.parse_adapter_uri(uri)[1], 'weights').read_text() == 'legacy'
    assert (directory / 'checkpoint_metadata.json').read_bytes() == checkpoint_before
    assert info_path.read_bytes() == before
    assert not list(directory.parent.glob('.pending-*'))
    assert not list(directory.parent.joinpath('.versions').iterdir())


@pytest.mark.parametrize('name', ['x', None])
@pytest.mark.asyncio
async def test_failed_first_save_does_not_publish_checkpoint(manager, monkeypatch, name):
    runs = manager.training_run_manager
    info_path = runs.get_model_dir('run') / runs.train_run_info_filename
    before = info_path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError('injected metadata failure')

    monkeypatch.setattr(runs, 'update', fail)
    with pytest.raises(OSError, match='metadata failure'):
        await manager.save_sampler('run', name, _writer('uncommitted'))
    save_dir = Path(manager.get_save_dir('run', is_sampler=True))
    assert info_path.read_bytes() == before
    assert not manager.list_checkpoints('run').checkpoints
    assert not list(save_dir.glob('.pending-*'))
    assert not list(save_dir.joinpath('.versions').iterdir())
    assert not any(path.is_symlink() for path in save_dir.iterdir())


@pytest.mark.asyncio
async def test_missing_backend_weights_does_not_publish_checkpoint(manager):
    runs = manager.training_run_manager
    info_path = runs.get_model_dir('run') / runs.train_run_info_filename
    before = info_path.read_bytes()

    async def save_without_weights(*, name, output_dir):
        return str(Path(output_dir) / name)

    with pytest.raises(RuntimeError, match='did not materialize'):
        await manager.save_sampler('run', 'x', save_without_weights)
    assert info_path.read_bytes() == before
    assert not manager.list_checkpoints('run').checkpoints
    assert not list(Path(manager.get_save_dir('run', is_sampler=True)).glob('.pending-*'))


@pytest.mark.asyncio
@pytest.mark.parametrize('manager', ['tinker', 'twinkle'], indirect=True)
async def test_training_and_sampler_checkpoints_can_be_read_and_deleted_by_other_dialect(manager):
    manager.training_run_manager.update('run', {
        'training_run_id': 'run', 'model_owner': manager.token, 'last_request_time': '2026-10-11T00:00:00'})
    training_uri, training_dir = await manager.save_training('run', 'step', _writer('training'))
    sampler_uri, sampler_dir = await manager.save_sampler('run', 'sample', _writer('sampler'))
    peer_type = 'twinkle' if isinstance(manager, TinkerCheckpointManager) else 'tinker'
    peer = create_checkpoint_manager(manager.token, client_type=peer_type)

    assert peer.resolve_load_path(training_uri).checkpoint_name == 'step'
    assert peer.resolve_checkpoint_dir(training_uri) == training_dir
    assert peer.resolve_checkpoint_dir(sampler_uri) == sampler_dir
    assert peer.get_weights_info(training_uri).base_model == 'test-model'
    assert {ckpt.checkpoint_id for ckpt in peer.list_checkpoints('run').checkpoints} == {
        'weights/step', 'sampler_weights/sample'}
    assert peer.get('run', 'weights/step').size_bytes > 0
    run = peer.training_run_manager.get('run').model_dump(mode='json')
    assert run['last_checkpoint']['checkpoint_id'] == 'weights/step'
    assert run['last_sampler_checkpoint']['checkpoint_id'] == 'sampler_weights/sample'

    assert peer.delete('run', 'weights/step')
    info = manager.training_run_manager._read_info('run')
    assert info['last_checkpoint'] is None
    assert info['last_sampler_checkpoint']['checkpoint_id'] == 'sampler_weights/sample'
    assert not Path(training_dir).exists()
    assert Path(sampler_dir, 'weights').read_text() == 'sampler'
    assert not peer.delete('run', 'weights/step')
    assert peer.delete('run', 'sampler_weights/sample')
    assert manager.training_run_manager._read_info('run')['last_sampler_checkpoint'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('manager', ['tinker', 'twinkle'], indirect=True)
async def test_load_and_resume_use_public_directory_while_sampler_cache_uses_version(manager):
    uri, public_directory = await manager.save_sampler('run', 'sample', _writer('sampler'))
    load = manager.resolve_load_path(uri)
    assert Path(load.checkpoint_dir, load.checkpoint_name) == Path(public_directory)
    assert manager.resolve_checkpoint_dir(uri) == public_directory
    physical_directory = Path(manager.parse_adapter_uri(uri)[1])
    assert physical_directory == Path(public_directory).resolve()
    assert physical_directory != Path(public_directory)

    other = type(manager)('another-token', type(manager.training_run_manager)('another-token'))
    with pytest.raises(ValueError, match='not found or access denied'):
        other.resolve_checkpoint_dir(uri)


@pytest.mark.parametrize('path', ['org/model', '/external/checkpoint', './relative/checkpoint'])
def test_external_load_paths_are_preserved(manager, path):
    load = manager.resolve_load_path(path)
    assert load.checkpoint_name == path and load.checkpoint_dir is None
    assert not load.is_twinkle_path
    assert manager.resolve_checkpoint_dir(path) == path


@pytest.mark.asyncio
async def test_overwrite_invalidates_existing_vllm_path_cache(manager):
    """Exercise the real cache method with file-backed adapter loading, not GPU generation."""
    from types import SimpleNamespace
    from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._lora_request_cache = {}
    engine._lora_load_tasks = {}
    loaded = []

    async def load(path):
        request = SimpleNamespace(weights=Path(path, 'weights').read_text())
        loaded.append(request)
        return request

    engine._load_lora = load
    uri, _ = await manager.save_sampler('run', 'x', _writer('first'))

    async def cached_adapter():
        return await engine._get_or_load_lora(manager.parse_adapter_uri(uri)[1])

    first = await cached_adapter()
    assert await cached_adapter() is first
    after, _ = await manager.save_sampler('run', 'x', _writer('second'))
    assert after == uri
    second = await cached_adapter()
    assert first.weights == 'first' and second.weights == 'second'
    assert await cached_adapter() is second and len(loaded) == 2


@pytest.mark.asyncio
async def test_reusing_live_alias_name_does_not_delete_latest_weights(manager):
    live, _ = await manager.save_sampler('run', None, _writer('live'))
    previous = Path(manager.parse_adapter_uri(live)[1])
    name = manager.parse_path(live).checkpoint_id.rsplit('/', 1)[1]
    named, _ = await manager.save_sampler('run', name, _writer('named'))
    assert named == live
    assert Path(manager.parse_adapter_uri(named)[1], 'weights').read_text() == 'named'
    assert Path(manager.get_save_dir('run', is_sampler=True), 'latest', 'weights').read_text() == 'live'
    await manager.save_sampler('run', None, _writer('new-live'))
    assert not previous.exists()
    assert Path(manager.parse_adapter_uri(named)[1], 'weights').read_text() == 'named'


@pytest.mark.asyncio
async def test_sampler_update_does_not_rewrite_existing_save_dir_pointer(monkeypatch, tmp_path):
    import twinkle.server.checkpoint.training_run_manager as run_module
    monkeypatch.setattr(run_module, 'TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path / 'default'))
    runs = TinkerTrainingRunManager('test-token')
    (tmp_path / 'external').mkdir()
    runs._write_info('run', {'base_model': 'test-model', 'save_dir': str(tmp_path / 'external')})
    pointer = runs._default_model_dir('run') / runs.train_run_info_filename
    original = pointer.read_bytes()
    write_atomic = runs._write_json_atomic

    def write(path, data):
        if path == pointer:
            raise PermissionError('Existing pointer is now read-only')
        return write_atomic(path, data)

    monkeypatch.setattr(runs, '_write_json_atomic', write)
    manager = TinkerCheckpointManager('test-token', runs)
    uri, directory = await manager.save_sampler('run', 'x', _writer('first'))
    assert pointer.read_bytes() == original
    assert Path(directory, 'weights').read_text() == 'first'
    assert runs._read_info('run')['last_sampler_checkpoint']['tinker_path'] == uri


@pytest.mark.asyncio
async def test_reserved_sampler_name_is_rejected_before_backend_or_file_io(manager):
    from unittest.mock import AsyncMock
    from twinkle.server.task_queue.types import UserTaskError
    save = AsyncMock()
    with pytest.raises(UserTaskError, match='reserved'):
        await manager.save_sampler('run', 'latest', save)
    save.assert_not_awaited()
    assert not Path(manager.get_save_dir('run', is_sampler=True)).exists()
