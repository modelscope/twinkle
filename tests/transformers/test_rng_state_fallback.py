# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU tests for the legacy single-master RNG-state fallback in _restore_training_state.

A checkpoint written before RNG state became per rank carries one ``rng_state.pt`` instead of
``rng_state_rank_{rank}.pt``. The restore must fall back to that master file (replaying rank 0's
stream on every rank) and NAME the consequence with a per-rank warning, rather than silently
skipping RNG replay or crashing on the missing per-rank file. defect33 exercised this end to end
(DEFECT33_fallback_ctrl); these lock the path-selection + warning contract against regression.

The unit under test is the RNG path selection and warning, so the heavy collaborators
(_load_optimizer / _load_rng_state / _load_scaler_state) are isolated; the assertions read which
path was handed to _load_rng_state and whether the warning fired, against real file existence.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from twinkle.model.transformers.transformers import TransformersModel


def _wrapper(tmp_path, *, rank_files=(), master=False, rank=0):
    ckpt = tmp_path / 'ckpt'
    ckpt.mkdir()
    (ckpt / 'trainer_state.json').write_text(json.dumps({'cur_step': 15, 'gradient_accumulation_steps': 4}))
    for name in rank_files:
        (ckpt / name).write_bytes(b'rank-stream')
    if master:
        (ckpt / 'rng_state.pt').write_bytes(b'master-stream')

    wrapper = object.__new__(TransformersModel)
    wrapper.optimizer_group = {'default': SimpleNamespace(scaler=None, cur_step=0, gradient_accumulation_steps=1)}
    wrapper._get_default_group = lambda: 'default'
    wrapper._load_optimizer = lambda *a, **k: None
    wrapper._load_scaler_state = lambda *a, **k: None
    loaded = []
    wrapper._load_rng_state = lambda path: loaded.append(path)

    module = 'twinkle.model.transformers.transformers'
    with patch(f'{module}.dist.is_initialized', return_value=(rank != 0)), \
            patch(f'{module}.dist.get_rank', return_value=rank):
        state = wrapper._restore_training_state(str(ckpt))
    return state, loaded, ckpt


def test_master_fallback_loads_master_and_warns_rank0(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        state, loaded, ckpt = _wrapper(tmp_path, master=True, rank=0)
    assert loaded == [str(ckpt / 'rng_state.pt')]
    assert state['cur_step'] == 15
    text = caplog.text
    assert 'single master-written RNG state' in text
    assert 'rank 0 replays' in text


def test_per_rank_file_preferred_over_master_without_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _, loaded, ckpt = _wrapper(tmp_path, rank_files=('rng_state_rank_0.pt',), master=True, rank=0)
    assert loaded == [str(ckpt / 'rng_state_rank_0.pt')]
    assert 'single master-written RNG state' not in caplog.text


def test_no_rng_state_proceeds_without_load_or_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        state, loaded, _ = _wrapper(tmp_path, rank=0)
    assert loaded == []
    assert state['cur_step'] == 15
    assert 'single master-written RNG state' not in caplog.text


def test_rank1_fallback_warning_names_rank1(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _, loaded, ckpt = _wrapper(tmp_path, master=True, rank=1)
    assert loaded == [str(ckpt / 'rng_state.pt')]
    assert 'rank 1 replays' in caplog.text