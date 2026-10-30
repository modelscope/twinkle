# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint helpers for synchronous RL with a shared filesystem."""

import os
import pickle
import random
from pathlib import Path

STATE_FILE = 'rl_state.pkl'


def save_rl_checkpoint(model, dataloader, *, name, output_dir, adapter_name, optim_step, pending, config):
    """Save at an optimizer-step boundary using synchronous model calls.

    pending contains the remaining rollout inputs, old logprobs and advantages.
    The driver and workers must share the checkpoint filesystem.
    """
    checkpoint_dir = Path(output_dir).resolve() / name
    if checkpoint_dir.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {checkpoint_dir}')

    data_state = dataloader.get_state()
    state = {
        'version': 1,
        'optim_step': optim_step,
        'data_state': data_state,
        'pending': pending,
        'config': config,
        'python_rng_state': random.getstate(),
    }

    model.save(
        name,
        output_dir=str(checkpoint_dir.parent),
        adapter_name=adapter_name,
        save_optimizer=True,
        **data_state,
    )

    # Commit controller state only after the synchronous model save succeeds.
    temporary = checkpoint_dir / (STATE_FILE + '.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, checkpoint_dir / STATE_FILE)
    return str(checkpoint_dir)


def load_rl_checkpoint(model, dataloader, checkpoint_dir, *, adapter_name, config):
    """Restore a trusted checkpoint; pickle files must come from trusted runs."""
    checkpoint_dir = str(Path(checkpoint_dir).resolve())
    with (Path(checkpoint_dir) / STATE_FILE).open('rb') as handle:
        state = pickle.load(handle)

    if state.get('version') != 1:
        raise ValueError('Unsupported RL checkpoint version')
    if state['config'] != config:
        raise ValueError('RL checkpoint configuration differs from this run')

    progress = model.resume_from_checkpoint(checkpoint_dir, adapter_name=adapter_name)
    if progress['cur_step'] != state['optim_step']:
        raise ValueError('Model and RL controller steps do not match')
    if (progress['consumed_train_samples'] != state['data_state']['consumed_train_samples']):
        raise ValueError('Model and dataloader progress do not match')

    dataloader.resume_from_checkpoint(**state['data_state'])
    random.setstate(state['python_rng_state'])
    return state['optim_step'], state['pending']
