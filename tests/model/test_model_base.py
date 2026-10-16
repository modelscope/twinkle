# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for the checkpoint helpers in ``twinkle.model.base``.

``rotate_checkpoints`` / ``copy_checkpoint_args`` run on the master rank inside the
transformers and megatron ``save`` path, but had no test at all. They are pure
filesystem logic, so these cases drive them against real temporary directories on CPU
(with ``Platform.is_master()`` true in a single-process run) and pin the retention
ordering, the current-checkpoint protection, the name filter and the copy guard.

``_should_bind_device_id_for_process_group`` is the tiny backend predicate that decides
whether ``init_process_group`` gets an explicit ``device_id``; it is covered here too.
"""
import os

import pytest

from twinkle.model.base import (TrainableModel, copy_checkpoint_args, rotate_checkpoints)


def _mkdirs(root, names):
    """Create sub-directories under root and return {name: path}."""
    paths = {}
    for name in names:
        path = os.path.join(str(root), name)
        os.makedirs(path, exist_ok=True)
        paths[name] = path
    return paths


def _set_mtime_ns(path, ns):
    os.utime(path, ns=(ns, ns))


# ---------------------------------------------------------------------------
# rotate_checkpoints
# ---------------------------------------------------------------------------

class TestRotateCheckpoints:

    def test_none_limit_keeps_everything(self, tmp_path):
        paths = _mkdirs(tmp_path, [f'checkpoint-{i}' for i in range(1, 6)])
        rotate_checkpoints(str(tmp_path), paths['checkpoint-5'], None)
        assert all(os.path.isdir(p) for p in paths.values())

    @pytest.mark.parametrize('limit', [0, -1])
    def test_limit_below_one_raises(self, tmp_path, limit):
        with pytest.raises(ValueError, match='save_total_limit must be'):
            rotate_checkpoints(str(tmp_path), str(tmp_path / 'checkpoint-1'), limit)

    def test_missing_output_dir_is_noop(self, tmp_path):
        # A non-existent output_dir must not raise (master + isdir guard).
        rotate_checkpoints(str(tmp_path / 'does-not-exist'), 'whatever', 2)

    def test_keeps_newest_by_mtime(self, tmp_path):
        paths = _mkdirs(tmp_path, [f'checkpoint-{i}' for i in range(1, 6)])
        for i, name in enumerate([f'checkpoint-{i}' for i in range(1, 6)]):
            _set_mtime_ns(paths[name], (i + 1) * 1_000_000_000)
        rotate_checkpoints(str(tmp_path), paths['checkpoint-5'], 2)
        assert os.path.isdir(paths['checkpoint-4'])
        assert os.path.isdir(paths['checkpoint-5'])
        for gone in ('checkpoint-1', 'checkpoint-2', 'checkpoint-3'):
            assert not os.path.exists(paths[gone])

    def test_current_is_protected_even_when_oldest(self, tmp_path):
        # The running checkpoint sorts last regardless of mtime, so it survives even when
        # it is the oldest directory -- this is the whole point of the is_current key.
        paths = _mkdirs(tmp_path, [f'checkpoint-{i}' for i in range(1, 5)])
        for i, name in enumerate([f'checkpoint-{i}' for i in range(1, 5)]):
            _set_mtime_ns(paths[name], (i + 1) * 1_000_000_000)  # checkpoint-1 oldest
        rotate_checkpoints(str(tmp_path), paths['checkpoint-1'], 2)
        assert os.path.isdir(paths['checkpoint-1'])  # protected current
        assert os.path.isdir(paths['checkpoint-4'])  # newest
        assert not os.path.exists(paths['checkpoint-2'])
        assert not os.path.exists(paths['checkpoint-3'])

    def test_matches_final_and_ignores_non_checkpoint_entries(self, tmp_path):
        paths = _mkdirs(tmp_path, ['checkpoint-1', 'checkpoint-2', 'checkpoint-final', 'checkpoint-abc', 'notes'])
        _set_mtime_ns(paths['checkpoint-1'], 1_000_000_000)
        _set_mtime_ns(paths['checkpoint-2'], 2_000_000_000)
        _set_mtime_ns(paths['checkpoint-final'], 3_000_000_000)
        # A regular file that looks like a checkpoint must also be left alone.
        stray_file = os.path.join(str(tmp_path), 'checkpoint-9')
        with open(stray_file, 'w') as handle:
            handle.write('not a dir')
        rotate_checkpoints(str(tmp_path), paths['checkpoint-final'], 1)
        assert os.path.isdir(paths['checkpoint-final'])  # current, kept
        assert not os.path.exists(paths['checkpoint-1'])
        assert not os.path.exists(paths['checkpoint-2'])
        # Non-matching name, non-numeric suffix, and the stray file are never candidates.
        assert os.path.isdir(paths['checkpoint-abc'])
        assert os.path.isdir(paths['notes'])
        assert os.path.isfile(stray_file)


# ---------------------------------------------------------------------------
# copy_checkpoint_args
# ---------------------------------------------------------------------------

class TestCopyCheckpointArgs:

    def test_copies_args_json_into_checkpoint(self, tmp_path):
        output_dir = tmp_path / 'output'
        checkpoint_dir = tmp_path / 'output' / 'checkpoint-1'
        output_dir.mkdir()
        checkpoint_dir.mkdir()
        (output_dir / 'args.json').write_text('{"model": "qwen"}')
        copy_checkpoint_args(str(output_dir), str(checkpoint_dir))
        assert (checkpoint_dir / 'args.json').read_text() == '{"model": "qwen"}'

    def test_missing_source_is_noop(self, tmp_path):
        output_dir = tmp_path / 'output'
        checkpoint_dir = tmp_path / 'output' / 'checkpoint-1'
        output_dir.mkdir()
        checkpoint_dir.mkdir()
        copy_checkpoint_args(str(output_dir), str(checkpoint_dir))
        assert not (checkpoint_dir / 'args.json').exists()

    def test_same_directory_skips_self_copy(self, tmp_path):
        output_dir = tmp_path / 'output'
        output_dir.mkdir()
        (output_dir / 'args.json').write_text('{"a": 1}')
        # source and target resolve to the same file -> the realpath guard skips the copy
        # (shutil.copy2 onto itself would otherwise raise SameFileError).
        copy_checkpoint_args(str(output_dir), str(output_dir))
        assert (output_dir / 'args.json').read_text() == '{"a": 1}'


# ---------------------------------------------------------------------------
# _should_bind_device_id_for_process_group
# ---------------------------------------------------------------------------

class TestShouldBindDeviceId:

    @pytest.mark.parametrize('backend, expected', [
        ('nccl', True),
        ('hccl', True),
        ('gloo', False),
        ('cpu', False),
        ('', False),
    ])
    def test_backend_predicate(self, backend, expected):
        # The method uses only its backend argument, so it is exercised unbound.
        assert TrainableModel._should_bind_device_id_for_process_group(None, backend) is expected
