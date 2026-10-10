"""Real filesystem save transactions shared by Tinker and Twinkle handlers."""
import asyncio
import json
from pathlib import Path

import pytest

from twinkle.server.checkpoint.tinker import TinkerCheckpointManager, TinkerTrainingRunManager
from twinkle.server.checkpoint.twinkle import TwinkleCheckpointManager, TwinkleTrainingRunManager


class _Runs:
    def __init__(self, root):
        self.root = root
        self.info = {'base_model': 'test-model', 'is_lora': True}

    def get_model_dir(self, model_id):
        return self.root / model_id

    def _read_info(self, model_id):
        return self.info.copy()

    def update(self, model_id, changes):
        self.info.update(changes)


@pytest.fixture(params=[TinkerCheckpointManager, TwinkleCheckpointManager])
def manager(request, tmp_path):
    return request.param('test-token', _Runs(tmp_path))


def _writer(value, fail=False):
    async def save(name, output_dir):
        checkpoint = Path(output_dir) / name
        checkpoint.mkdir()
        (checkpoint / 'weights').write_text(value)
        await asyncio.sleep(.01)
        if fail:
            raise RuntimeError('injected save failure')
        return str(checkpoint)
    return save


@pytest.mark.asyncio
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
    assert 'last_checkpoint' not in manager.training_run_manager.info
    assert manager.delete('run', manager.parse_path(latest).checkpoint_id)
    assert Path(first_dir).exists()
    assert manager.delete('run', 'sampler_weights/x')
    assert not Path(first_dir).exists()


@pytest.mark.asyncio
async def test_overwrite_failure_preserves_weights_and_revision(manager):
    uri, directory = await manager.save_sampler('run', 'x', _writer('first'))
    metadata = Path(directory, 'checkpoint_metadata.json')
    before = json.loads(metadata.read_text())
    with pytest.raises(RuntimeError, match='injected'):
        await manager.save_sampler('run', 'x', _writer('broken', fail=True))
    assert Path(directory, 'weights').read_text() == 'first'
    assert json.loads(metadata.read_text()) == before
    after_uri, after_dir = await manager.save_sampler('run', 'x', _writer('second'))
    assert (after_uri, after_dir) == (uri, directory)
    assert Path(directory, 'weights').read_text() == 'second'
    assert json.loads(metadata.read_text())['weights_revision'] != before['weights_revision']
    assert not list(Path(directory).parent.glob('.pending-*'))


@pytest.mark.asyncio
async def test_metadata_failure_rolls_back_published_weights(manager, monkeypatch):
    _, directory = await manager.save_sampler('run', 'x', _writer('first'))
    def fail(*args):
        raise RuntimeError('metadata failure')
    monkeypatch.setattr(manager.training_run_manager, 'update', fail)
    with pytest.raises(RuntimeError, match='metadata failure'):
        await manager.save_sampler('run', 'x', _writer('second'))
    assert Path(directory, 'weights').read_text() == 'first'


@pytest.mark.asyncio
async def test_same_model_saves_are_serialized(manager):
    active = 0
    max_active = 0
    async def save(name, output_dir):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await _writer('same')(name, output_dir)
        active -= 1
    results = await asyncio.gather(*(manager.save_sampler('run', 'x', save) for _ in range(3)))
    assert max_active == 1
    assert len(set(results)) == 1


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_reserved_name_does_not_write(manager):
    with pytest.raises(ValueError, match='reserved'):
        await manager.save_sampler('run', 'latest', _writer('unused'))


@pytest.mark.parametrize('client', ['tinker', 'twinkle'])
@pytest.mark.parametrize('name', ['x', None])
@pytest.mark.parametrize('failure', ['write', 'flush', 'replace'])
@pytest.mark.asyncio
async def test_real_metadata_failure_preserves_run_and_checkpoint(monkeypatch, tmp_path, client, name, failure):
    import twinkle.server.checkpoint.training_run_manager as run_module
    monkeypatch.setattr(run_module, 'TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    run_cls, ckpt_cls = ((TinkerTrainingRunManager, TinkerCheckpointManager) if client == 'tinker' else
                        (TwinkleTrainingRunManager, TwinkleCheckpointManager))
    runs = run_cls('test-token')
    runs._write_info('run', {'base_model': 'test-model', 'is_lora': True})
    manager = ckpt_cls('test-token', runs)
    old_uri, directory = await manager.save_sampler('run', name, _writer('first'))
    info_path = runs.get_model_dir('run') / runs.train_run_info_filename
    before = info_path.read_bytes()
    before_checkpoint = Path(directory, 'checkpoint_metadata.json').read_bytes()

    def fail(*args, **kwargs):
        if failure == 'write':
            stream = args[1]
            stream.write('{')
            stream.flush()
        raise OSError('injected metadata failure')

    target, attr = {'write': (run_module.json, 'dump'), 'flush': (run_module.os, 'fsync'),
                    'replace': (run_module.os, 'replace')}[failure]
    monkeypatch.setattr(target, attr, fail)
    with pytest.raises(OSError, match='injected metadata failure'):
        await manager.save_sampler('run', name, _writer('second'))
    assert Path(directory, 'weights').read_text() == 'first'
    assert Path(directory, 'checkpoint_metadata.json').read_bytes() == before_checkpoint
    assert info_path.read_bytes() == before
    assert manager.resolve_load_path(old_uri).checkpoint_name == old_uri.rsplit('/', 1)[1]
    assert not list(info_path.parent.glob(f'.{info_path.name}-*'))
    assert not list(Path(directory).parent.glob('.pending-*'))


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
async def test_reserved_sampler_name_is_a_user_failure_in_queue(tmp_path):
    from tests.server.utils.test_task_queue_mixin import _DummyQueue
    manager = TinkerCheckpointManager('test-token', _Runs(tmp_path))
    queue = _DummyQueue()
    queue.enable_compute_worker()

    async def save():
        return await manager.save_sampler('run', 'latest', _writer('unused'))

    try:
        await queue.schedule_task(save, model_id='run', token='test-token')
        for _ in range(100):
            failures = [kwargs['failure'] for args, kwargs in queue.state.records if args[1] == 'failed']
            if failures:
                break
            await asyncio.sleep(0)
        failure = failures[-1]
        assert failure.reason_code == 'request_rejected'
        assert failure.attribution == 'user'
        assert failure.diagnostic is None
        assert not (tmp_path / 'run').exists()
    finally:
        await queue._compute_worker.stop()
