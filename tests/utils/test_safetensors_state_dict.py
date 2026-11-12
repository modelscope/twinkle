# Copyright (c) ModelScope Contributors. All rights reserved.
import random

import numpy as np
import torch

from twinkle.utils.safetensors import load_state_dict, save_state_dict


def test_state_dict_safetensors_round_trip(tmp_path, monkeypatch):
    state = {
        'optimizer': {
            'state': {0: {'exp_avg': torch.tensor([1.0, 2.0]), 'step': torch.tensor(3)}},
            'param_groups': [{'lr': 1e-4, 'betas': (0.9, 0.999)}],
        },
        'rng_state': {
            'random_rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.tensor([1, 2], dtype=torch.uint8),
        },
        'iteration': 3,
    }

    path = tmp_path / 'optimizer_rank_0.safetensors'
    save_state_dict(state, str(path))

    def fail_if_called(*args, **kwargs):
        raise AssertionError('torch.load must not be used for optimizer checkpoints')

    monkeypatch.setattr(torch, 'load', fail_if_called)
    loaded = load_state_dict(str(path))

    assert path.exists()
    assert (tmp_path / 'optimizer_rank_0.safetensors.json').exists()
    assert torch.equal(loaded['optimizer']['state'][0]['exp_avg'], state['optimizer']['state'][0]['exp_avg'])
    assert loaded['optimizer']['param_groups'] == state['optimizer']['param_groups']
    assert loaded['rng_state']['random_rng_state'] == state['rng_state']['random_rng_state']
    assert np.array_equal(loaded['rng_state']['np_rng_state'][1], state['rng_state']['np_rng_state'][1])
