import json
import random
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import twinkle.model.transformers.transformers as model_module
from twinkle.model.transformers.transformers import TransformersModel


def capture_rng():
    return {
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'device_type': 'cpu',
        'device_rng_state': None,
    }


def draw_random_values():
    return random.random(), np.random.random(), torch.rand(4)


def make_wrapper():
    config = SimpleNamespace(
        cur_step=3,
        gradient_accumulation_steps=1,
        scaler=None,
    )
    wrapper = SimpleNamespace(
        optimizer_group={'': config},
        _save_optimizer=lambda *args, **kwargs: None,
        _load_optimizer=lambda *args, **kwargs: None,
        _get_default_group=lambda: '',
        _get_training_rng_state=capture_rng,
    )
    wrapper._load_rng_state = partial(
        TransformersModel._load_rng_state, wrapper
    )
    return wrapper


def set_rank(monkeypatch, rank):
    monkeypatch.setattr(model_module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(model_module.dist, 'get_rank', lambda: rank)
    monkeypatch.setattr(
        model_module.Platform, 'is_master', lambda: rank == 0
    )


def write_metadata(folder):
    (folder / 'trainer_state.json').write_text(json.dumps({
        'cur_step': 3,
        'gradient_accumulation_steps': 1,
        'consumed_train_samples': 8,
    }))


@pytest.mark.parametrize('rank', [0, 1])
def test_each_rank_saves_and_restores_rng(tmp_path, monkeypatch, rank):
    set_rank(monkeypatch, rank)
    wrapper = make_wrapper()
    before = capture_rng()
    try:
        random.seed(100 + rank)
        np.random.seed(100 + rank)
        torch.manual_seed(100 + rank)

        TransformersModel._save_training_state(wrapper, str(tmp_path))
        assert (tmp_path / f'rng_state_rank_{rank}.pt').is_file()
        assert not (tmp_path / 'rng_state.pt').exists()

        expected = draw_random_values()
        draw_random_values()
        write_metadata(tmp_path)

        TransformersModel._restore_training_state(wrapper, str(tmp_path))
        actual = draw_random_values()

        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    finally:
        random.setstate(before['python_rng_state'])
        np.random.set_state(before['numpy_rng_state'])
        torch.set_rng_state(before['torch_rng_state'])


@pytest.mark.parametrize('rank_file_exists', [True, False])
def test_rank_file_priority_and_legacy_fallback(
    tmp_path, monkeypatch, rank_file_exists
):
    set_rank(monkeypatch, 1)
    wrapper = make_wrapper()
    write_metadata(tmp_path)

    torch.save({'source': 'legacy'}, tmp_path / 'rng_state.pt')
    if rank_file_exists:
        torch.save({'source': 'rank-1'}, tmp_path / 'rng_state_rank_1.pt')

    loaded = []
    wrapper._load_rng_state = lambda path: loaded.append(
        torch.load(path, map_location='cpu', weights_only=True)['source']
    )

    TransformersModel._restore_training_state(wrapper, str(tmp_path))

    assert loaded == ['rank-1' if rank_file_exists else 'legacy']


@pytest.mark.parametrize('legacy_file_exists', [True, False])
def test_new_format_missing_rank_rng_is_rejected(
    tmp_path, monkeypatch, legacy_file_exists
):
    set_rank(monkeypatch, 1)
    wrapper = make_wrapper()
    (tmp_path / 'trainer_state.json').write_text(json.dumps({
        'checkpoint_version': 1,
        'rng_state_format': 'per_rank',
        'cur_step': 3,
        'gradient_accumulation_steps': 1,
        'consumed_train_samples': 8,
    }))
    if legacy_file_exists:
        torch.save(capture_rng(), tmp_path / 'rng_state.pt')

    optimizer_loads = []
    wrapper._load_optimizer = lambda *a, **k: optimizer_loads.append(True)

    with pytest.raises(FileNotFoundError, match='rank 1'):
        TransformersModel._restore_training_state(wrapper, str(tmp_path))

    assert optimizer_loads == []


def test_save_marks_per_rank_rng_format(tmp_path, monkeypatch):
    set_rank(monkeypatch, 0)
    wrapper = make_wrapper()
    TransformersModel._save_training_state(wrapper, str(tmp_path))

    state = json.loads((tmp_path / 'trainer_state.json').read_text())
    assert state['rng_state_format'] == 'per_rank'
    assert (tmp_path / 'rng_state_rank_0.pt').is_file()


@pytest.mark.parametrize('rank', [0, 1])
def test_save_creates_training_directory_on_each_rank(tmp_path, monkeypatch, rank):
    set_rank(monkeypatch, rank)
    wrapper = make_wrapper()
    folder = tmp_path / 'new-run' / 'checkpoint'
    assert not folder.exists()

    optimizer_calls = []

    def save_optimizer(output_dir, **kwargs):
        assert folder.is_dir(), 'Directory must exist before optimizer saving'
        optimizer_calls.append(str(output_dir))

    wrapper._save_optimizer = save_optimizer
    TransformersModel._save_training_state(wrapper, str(folder))

    assert optimizer_calls == [str(folder)]
    assert (folder / f'rng_state_rank_{rank}.pt').is_file()
    assert (folder / 'trainer_state.json').exists() == (rank == 0)
