# Unified `resume_from_checkpoint` API Design

## Problem

The current checkpoint resume API on the `resume_from_ckpt` branch exposes two similar methods (`load_training_state` and `read_training_progress`) that are hard to distinguish. The caller must manually orchestrate state restoration across model and dataloader, acting as a data courier between components. Additionally, the Megatron backend lacks these methods entirely, creating an asymmetric API surface.

## Design Principle

Each component is responsible for its own state restoration. The caller only orchestrates — it does not transport data between components.

## Target API

```python
progress = model.resume_from_checkpoint(checkpoint_path)
dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
```

Two lines. Both backends. No `resume_utils.py` helper needed.

## Return Value Contract

`model.resume_from_checkpoint()` returns a dict with exactly these keys:

```python
{
    'cur_step': int,                    # optimizer step count
    'consumed_train_samples': int,      # total samples consumed
    'gradient_accumulation_steps': int, # GAS value at save time
}
```

Backend-specific state (optimizer tensors, scaler, RNG, mcore sharded state) is restored internally and not exposed.

## Component Changes

### 1. TwinkleModel Base Class (`src/twinkle/model/base.py`)

Add abstract method:

```python
@abstractmethod
def resume_from_checkpoint(
    self,
    checkpoint_dir: str,
    *,
    resume_only_model: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    ...
```

Parameters:
- `checkpoint_dir`: Path to the checkpoint directory.
- `resume_only_model`: If True, load weights only — skip optimizer/scheduler/RNG restoration. Useful for fine-tuning with a different optimizer config.
- `**kwargs`: Backend-specific args (e.g., `adapter_name`).

### 2. TransformersModel (`src/twinkle/model/transformers/transformers.py`)

Delete public methods: `load_training_state()`, `read_training_progress()`.

Retain private helpers: `_save_training_state()`, `_load_optimizer()`, `_load_scaler_state()`, `_load_rng_state()`, `_get_training_rng_state()`.

New implementation:

```python
@remote_function()
def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
    adapter_name = kwargs.get('adapter_name', '')

    # Load adapter weights if checkpoint contains adapter files.
    has_adapter = (
        os.path.exists(os.path.join(checkpoint_dir, 'adapter_model.safetensors'))
        or os.path.exists(os.path.join(checkpoint_dir, 'adapter_model.bin'))
    )
    if has_adapter:
        self.load(checkpoint_dir, adapter_name=adapter_name)

    # Read trainer_state.json.
    trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)

    # Full restore: optimizer, scheduler, scaler, RNG.
    if not resume_only_model:
        optimizer_group = self._get_optimizer_group(adapter_name)
        self._load_optimizer(checkpoint_dir, optimizer_group, adapter_name)
        self._load_scaler_state(checkpoint_dir)
        self._load_rng_state(checkpoint_dir)
        optimizer_group.cur_step = trainer_state['cur_step']
        optimizer_group.gradient_accumulation_steps = trainer_state['gradient_accumulation_steps']

    return {
        'cur_step': trainer_state['cur_step'],
        'consumed_train_samples': trainer_state['consumed_train_samples'],
        'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
    }
```

Full-parameter training: weights are loaded at model initialization time, so `has_adapter` is False and `self.load()` is skipped. Only training state is restored.

### 3. MegatronModel (`src/twinkle/model/megatron/megatron.py`)

**save() change:** When `save_optimizer=True`, also write `trainer_state.json`:

```python
if save_optimizer:
    self._save_mcore_optimizer(checkpoint_dir, optimizer_config=optimizer_config, **kwargs)
    trainer_state = {
        'checkpoint_version': 1,
        'cur_step': optimizer_config.cur_step,
        'consumed_train_samples': kwargs.get('consumed_train_samples', 0),
        'gradient_accumulation_steps': optimizer_config.gradient_accumulation_steps,
    }
    state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
    if self.device_mesh.rank == 0:
        with open(state_path, 'w') as f:
            json.dump(trainer_state, f, indent=2)
```

**New resume_from_checkpoint():**

```python
@remote_function(dispatch='all')
def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
    adapter_name = kwargs.get('adapter_name', self._get_default_group())

    trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)

    self.load(checkpoint_dir, load_optimizer=not resume_only_model,
              adapter_name=adapter_name, **kwargs)

    return {
        'cur_step': trainer_state['cur_step'],
        'consumed_train_samples': trainer_state['consumed_train_samples'],
        'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
    }
```

Megatron's `load(load_optimizer=True)` already restores optimizer/scheduler/RNG/cur_step via `_load_mcore_optimizer`. The `resume_from_checkpoint` wrapper adds `trainer_state.json` reading for `consumed_train_samples`.

### 4. DataLoader (`src/twinkle/dataloader/dataloader.py`)

New method:

```python
def resume_from_checkpoint(self, consumed_train_samples, **kwargs):
    self.skip_consumed_samples(consumed_train_samples)
```

`skip_consumed_samples` is retained as-is (not renamed) for backward compatibility. `resume_from_checkpoint` is the recommended public API going forward.

### 5. Server Endpoints (`src/twinkle/server/model/twinkle_handlers.py`)

- Delete: `/twinkle/load_training_state`, `/twinkle/read_training_progress`
- Add: `/twinkle/resume_from_checkpoint` accepting `checkpoint_dir` and `resume_only_model` parameters

### 6. Client SDK (`src/twinkle_client/`, `client_tools/client_generator.py`)

- Delete: `load_training_state()`, `read_training_progress()` client methods
- Add: `resume_from_checkpoint()` client method

### 7. Cookbook Changes

- Delete `resume_from_checkpoint()` helper from `cookbook/transformers/resume_utils.py` (functionality now lives in the model)
- Update all cookbook examples to use the new two-line API

### 8. Documentation

Update `docs/source_en/Components/Model/TransformersModel.md` and corresponding Chinese docs to reflect the new API.

## Migration Summary

| Before | After |
|--------|-------|
| `model.load(path)` | `progress = model.resume_from_checkpoint(path)` |
| `model.load_training_state(path)` | (merged into above) |
| `model.read_training_progress(path)` | `progress = model.resume_from_checkpoint(path, resume_only_model=True)` |
| `dataloader.skip_consumed_samples(n)` | `dataloader.resume_from_checkpoint(n)` |
| `resume_from_checkpoint(model, dataloader, ...)` (cookbook util) | Two-line inline call |
