# Transformers Strict Resume Design

## Summary

This design adds real checkpoint resumption support for `TransformersModel` without introducing a new trainer class.

The design supports both full-parameter training and LoRA training:

- full-parameter training restores weights during model initialization
- LoRA training restores adapter weights through the existing load path
- both modes share the same training-state resume contract
- strict model-state resume does not silently fall back to weight-only loading when required state is missing

Because Twinkle keeps the training loop explicit in user code, the design extends existing model, dataloader, server, and client interfaces rather than adding a central trainer abstraction.

## Goals

- Support true checkpoint resume for `TransformersModel`
- Support both full-parameter and LoRA training resume
- Restore optimizer state, scheduler state, scaler state, RNG state, and step counters
- Support dataset progress skipping for map-style datasets
- Expose Swift-like resume controls without adding a new trainer class
- Preserve existing weight-only loading and saving behavior

## Non-Goals

- Do not introduce a new `Trainer` class or resume manager class
- Do not guarantee exact sample-by-sample replay when retry-based sampling changes sample order
- Do not support exact data-progress resume for `IterableDataset` or streaming datasets
- Do not attempt to persist transient runtime state such as in-flight batch tensors, current loss tensors, or metric caches

## User-Facing Resume Controls

Resume behavior is controlled by existing training entrypoints through three new parameters:

- `resume_from_checkpoint: Optional[str] = None`
- `resume_only_model: bool = False`
- `ignore_data_skip: bool = False`

### Parameter semantics

#### `resume_from_checkpoint`

- Specifies the checkpoint directory or checkpoint path to resume from
- When unset, training starts normally from scratch
- When set, the training entrypoint reads the checkpoint and restores model state through existing model APIs

#### `resume_only_model`

- Defaults to `False`
- When `False`, resume restores full training state
- When `True`, resume restores only model weights

#### `ignore_data_skip`

- Only meaningful when `resume_from_checkpoint` is set and `resume_only_model=True`
- Defaults to `False`
- When `False`, the system still restores training progress metadata needed for data skipping and step/epoch continuation, but does not restore optimizer, scheduler, scaler, or RNG
- When `True`, the system restores only model weights and does not restore training progress or skip consumed data

### Effective behavior matrix

#### Case 1: `resume_from_checkpoint is None`

- Start a new training run

#### Case 2: `resume_from_checkpoint is not None` and `resume_only_model=False`

- Restore model weights
- Restore optimizer state
- Restore scheduler state
- Restore scaler state
- Restore RNG state
- Restore step counters
- Attempt to skip already consumed training data
- If required model training state is missing, fail without fallback

#### Case 3: `resume_from_checkpoint is not None` and `resume_only_model=True` and `ignore_data_skip=False`

- Restore model weights only
- Do not restore optimizer, scheduler, scaler, or RNG
- Restore step/progress metadata needed for data skipping
- Attempt to skip already consumed training data

#### Case 4: `resume_from_checkpoint is not None` and `resume_only_model=True` and `ignore_data_skip=True`

- Restore model weights only
- Do not restore optimizer, scheduler, scaler, RNG, step counters, or data progress
- Restart the training loop from step 0 with no skipping

## Checkpoint Layout

Existing weight layouts remain valid. New training-state files are added alongside current checkpoint contents.

### Existing files preserved

- full-model weights saved by `save_pretrained`
- LoRA weights saved as `adapter_model.safetensors`
- tokenizer artifacts
- `optimizer.pt`
- `scheduler.pt`

### New training-state files

- `scaler.pt`
- `trainer_state.json`
- `rng_state.pt`

### `trainer_state.json` contents

`trainer_state.json` stores lightweight training metadata:

- `checkpoint_version`
- `cur_step`
- `gradient_accumulation_steps`
- `consumed_train_samples`
- optionally `consumed_batches`

The design prefers storing `consumed_train_samples` as the canonical progress value and deriving batch skipping from it where needed.

### `scaler.pt` contents

- AMP scaler state dict
- optional scaler-related flags such as `scaler_has_nan`

### `rng_state.pt` contents

- Python `random` state
- NumPy RNG state
- PyTorch CPU RNG state
- CUDA RNG state

## Restore Paths

## Full-Parameter Training

For full-parameter training, model weights are restored during initialization.

### Full-parameter restore flow

1. Construct `TransformersModel(model_id=ckpt_dir, ...)`
2. `__init__` uses `from_pretrained(ckpt_dir, ...)` to restore weights
3. Create optimizer, scheduler, and scaler objects
4. Call `load_training_state(ckpt_dir)` to restore training state
5. If data skipping is enabled, rebuild dataloader with skip arguments derived from `trainer_state.json`

This means full-parameter resume does not need a separate model-weight loading method after initialization. It only needs explicit training-state restoration.

## LoRA Training

For LoRA training, the existing adapter-weight load path remains in place.

### LoRA restore flow

1. Construct the model and adapter objects as today
2. Restore adapter weights through the existing `load()` path
3. Create optimizer, scheduler, and scaler objects
4. Call the same `load_training_state(ckpt_dir)` method to restore training state
5. If data skipping is enabled, rebuild dataloader with skip arguments derived from `trainer_state.json`

## Unified training-state method

The model layer gains a shared helper such as `load_training_state(ckpt_dir)`.

This method restores:

- `optimizer.pt`
- `scheduler.pt`
- `scaler.pt`
- `trainer_state.json`
- `rng_state.pt`

It assumes the corresponding optimizer, scheduler, and scaler objects have already been created before invocation.

## Model Save and Load Semantics

## Save behavior

When saving with optimizer state enabled, the checkpoint includes:

- weights in the existing full-model or LoRA format
- tokenizer artifacts
- `optimizer.pt`
- `scheduler.pt`
- `scaler.pt`
- `trainer_state.json`
- `rng_state.pt`

When optimizer save is disabled, save remains weight-only and does not produce strict resume metadata.

## Strict training-state restore

Strict model-state resume restores:

- optimizer state
- scheduler state
- scaler state
- RNG state
- `cur_step`
- `gradient_accumulation_steps`
- data-progress metadata

### Failure behavior

When strict training-state restore is requested, missing required model training state is an error:

- missing `trainer_state.json` -> fail
- missing `optimizer.pt` when optimizer restore is required -> fail
- missing `scheduler.pt` when scheduler restore is required -> fail
- missing `scaler.pt` when scaler restore is required -> fail
- missing `rng_state.pt` when RNG restore is required -> fail
- malformed required fields -> fail

This intentionally does not fall back to weight-only loading, to avoid falsely signaling successful strict resume.

## Training Progress and Data Skipping

Twinkle does not currently have a central trainer abstraction. Because of that, data skipping must be driven by existing training entrypoints and dataloader arguments.

## Dataloader extensions

Existing dataloader and sampler code is extended rather than replaced:

- `twinkle.dataloader.DataLoader`
- `twinkle.dataloader.DeviceMeshSampler`
- retry-aware sampler flow

The dataloader gains resume-oriented arguments:

- `skip_samples: int = 0`
- optionally `skip_batches: int = 0`

Map-style datasets use this progress to skip already consumed data before yielding new training batches.

## Map-style dataset behavior

For datasets with `__len__`, Twinkle attempts to skip previously consumed data using sampler or batch-sampler level skipping.

Preferred behavior:

- preserve existing sharding logic
- apply skip before data is yielded to the training loop
- keep the solution compatible with current `DeviceMeshSampler` wrapping

## Iterable and streaming behavior

`IterableDataset` and streaming datasets do not support exact progress skipping in this design.

Behavior for these datasets:

- restore model state according to the selected resume mode
- log a clear warning that consumed-data skipping is not supported
- continue training without skipping historical samples

This is the only fallback allowed in the design. It applies only to dataset progress skipping, not to model-state resume.

## Entry Point Integration

No new trainer class is introduced.

Resume parameters are threaded through existing training entrypoints:

- direct local training loops using `TwinkleModel` / `TransformersModel`
- current client/server training flows that already support checkpoint save and load

The practical integration model is:

1. Parse or receive the three resume parameters
2. If `resume_from_checkpoint` is unset, construct dataloader normally
3. Construct model weights through the appropriate path
   - full-parameter: restore through `__init__`
   - LoRA: restore through existing adapter load logic
4. If `resume_only_model=False`, call `load_training_state(ckpt_dir)`
5. If `resume_only_model=True` and `ignore_data_skip=False`, read `trainer_state.json` for progress only
6. Recreate the dataloader with skip arguments applied when skipping is enabled

This keeps the training loop explicit and compatible with current Twinkle examples.

## Server and Client Behavior

Server-side checkpoint save/load behavior should preserve current APIs while adding richer metadata.

### Save path

When server-side save endpoints request optimizer save:

- save the model checkpoint as today
- save `optimizer.pt`, `scheduler.pt`, `scaler.pt`, `trainer_state.json`, and `rng_state.pt`
- persist checkpoint metadata through the existing checkpoint manager

### Load path

Current model load APIs remain the weight-loading trigger.

The new resume parameters are primarily a training-entrypoint concern. They orchestrate whether to:

- restore full training state
- restore weight only
- request data skipping

The underlying server model APIs do not need a new trainer object to support this.

## Compatibility Strategy

### Existing checkpoints

Existing checkpoints remain loadable in weight-only mode.

Examples:

- weight-only initialization for full-parameter checkpoints continues to work
- existing LoRA weight loading continues to work
- inference-only consumers remain unaffected

### Old checkpoints under strict resume

Old checkpoints that lack the new training-state files are not valid for strict resume.

Expected behavior:

- strict resume fails clearly
- weight-only load continues to work when requested explicitly

### `resume_only_model=True`

For `resume_only_model=True`, old checkpoints may still be usable if weight files are present.

If data skipping is requested but no progress metadata exists, the entrypoint should fail clearly rather than silently train from the beginning while claiming resumed progress.

## Risks and Constraints

### RetrySampler interaction

`RetrySampler` may retry or replace failed samples, including random backfill behavior at the tail of an epoch.

Because of that:

- progress skipping can preserve approximate data position
- exact sample-for-sample replay is not guaranteed when retry or backfill paths are exercised

This limitation should be documented explicitly.

### Dataset shape changes

If dataset definition, slicing, filtering, or shuffle configuration changes between save and resume, data skipping semantics may become invalid.

The user guidance should state that resume should be done with unchanged training parameters and unchanged dataset configuration.

### Distributed consistency

Skip logic must be compatible with current device-mesh sharding. The implementation should ensure skip is applied consistently before per-rank slicing causes divergence.

## Testing Strategy

Tests should cover:

### Full-parameter training resume

- initializing with `model_id=ckpt_dir` restores weights
- `load_training_state(ckpt_dir)` restores optimizer, scheduler, scaler, RNG, and step metadata

### LoRA training resume

- adapter-weight restore continues to work
- `load_training_state(ckpt_dir)` restores shared training state correctly

### Strict restore failures

- strict resume fails when required files are missing
- malformed state files fail clearly

### Weight-only compatibility

- legacy checkpoints still load in weight-only mode
- `resume_only_model=True` restores weights without optimizer, scheduler, scaler, or RNG

### Data progress skipping

- map-style datasets skip consumed data correctly
- skip behavior remains correct with device-mesh sharding
- iterable and streaming datasets emit warnings and continue without skipping

## Implementation Outline

1. Add model helpers for saving and loading split training-state files
2. Implement `load_training_state(ckpt_dir)` with shared behavior for full-parameter and LoRA training
3. Keep full-parameter weight restore in `__init__`
4. Keep LoRA weight restore in the existing adapter load path
5. Extend dataloader and sampler stack to support skip arguments for map-style datasets
6. Thread `resume_from_checkpoint`, `resume_only_model`, and `ignore_data_skip` through existing training entrypoints
7. Add warnings for unsupported iterable and streaming data skipping
8. Update docs and examples to show the new resume contract

## User Guidance

Recommended guidance text:

- To resume training, keep other parameters unchanged and provide `resume_from_checkpoint`
- `resume_only_model=False` performs full resume
- `resume_only_model=True` restores only model weights
- `ignore_data_skip=True` disables progress restore and starts from step 0
- Full-parameter checkpoints restore weights during model initialization and restore training state afterward
- Iterable and streaming datasets do not support consumed-data skipping and will resume without skipping data
