# Transformers Strict Resume Design

## Summary

This design adds real checkpoint resumption support for `TransformersModel` without introducing a new trainer class.

The implementation aligns the resume semantics of `TransformersModel` with the existing `MegatronModel` behavior:

- normal weight loading remains available
- strict resume restores model weights and training state together
- strict resume does not silently fall back to weight-only loading when required state is missing

Because Twinkle keeps the training loop explicit in user code, the design extends existing model, dataloader, server, and client interfaces rather than adding a central trainer abstraction.

## Goals

- Support true checkpoint resume for `TransformersModel`
- Restore model weights, optimizer state, scheduler state, RNG state, and step counters
- Support dataset progress skipping for map-style datasets
- Expose Swift-like resume controls without adding a new trainer class
- Preserve existing weight-only loading and saving behavior
- Keep backward compatibility for existing checkpoints where possible

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
- When `False`, the system still restores training progress metadata needed for data skipping and step/epoch continuation, but does not restore optimizer, scheduler, or RNG
- When `True`, the system restores only model weights and does not restore training progress or skip consumed data

### Effective behavior matrix

#### Case 1: `resume_from_checkpoint is None`

- Start a new training run

#### Case 2: `resume_from_checkpoint is not None` and `resume_only_model=False`

- Restore model weights
- Restore optimizer state
- Restore scheduler state
- Restore RNG state
- Restore step counters
- Attempt to skip already consumed training data
- If required model training state is missing, fail without fallback

#### Case 3: `resume_from_checkpoint is not None` and `resume_only_model=True` and `ignore_data_skip=False`

- Restore model weights only
- Do not restore optimizer, scheduler, or RNG
- Restore step/progress metadata needed for data skipping
- Attempt to skip already consumed training data

#### Case 4: `resume_from_checkpoint is not None` and `resume_only_model=True` and `ignore_data_skip=True`

- Restore model weights only
- Do not restore optimizer, scheduler, RNG, step counters, or data progress
- Restart the training loop from step 0 with no skipping

## Checkpoint Layout

Existing checkpoint layout remains valid. New resume metadata is added alongside current files.

### Existing files preserved

- model weights saved by `save_pretrained`
- LoRA weights saved as `adapter_model.safetensors`
- tokenizer artifacts
- `optimizer.pt`
- `scheduler.pt`

### New file

- `training_state.pt`

### `training_state.pt` contents

`training_state.pt` stores a small dictionary with the following fields:

- `checkpoint_version`
- `cur_step`
- `gradient_accumulation_steps`
- `scaler_state_dict`
- `scaler_has_nan`
- `rng_state`
- `data_progress`

### `rng_state` contents

- Python `random` state
- NumPy RNG state
- PyTorch CPU RNG state
- CUDA RNG state

### `data_progress` contents

First version stores progress in a compact form:

- `consumed_train_samples`
- optionally `consumed_batches` when this is easier to compute reliably in a given entrypoint

The design prefers storing `consumed_train_samples` as the canonical progress value and deriving batch skipping from it where needed.

## Model Save and Load Semantics

## `TransformersModel.save`

`TransformersModel.save(..., save_optimizer=True)` is extended to:

1. Save weights exactly as today
2. Save tokenizer exactly as today
3. Save `optimizer.pt` and `scheduler.pt` exactly as today
4. Save `training_state.pt`

When `save_optimizer=False`, save remains weight-only and does not produce strict resume metadata.

## `TransformersModel.load`

`TransformersModel.load(..., load_optimizer=False)` keeps current behavior:

- load model weights only

`TransformersModel.load(..., load_optimizer=True)` becomes strict model-state resume:

1. Resolve checkpoint directory
2. Load model weights
3. Load optimizer and scheduler state
4. Load `training_state.pt`
5. Restore scaler state
6. Restore RNG state
7. Restore `cur_step` and `gradient_accumulation_steps`

### Failure behavior

When `load_optimizer=True`, missing required model training state is an error:

- missing `training_state.pt` -> fail
- missing `optimizer.pt` when optimizer restore is required -> fail
- missing `scheduler.pt` when scheduler restore is required -> fail
- malformed required fields in `training_state.pt` -> fail

This intentionally does not fall back to weight-only loading, to avoid falsely signaling successful strict resume.

This matches the `MegatronModel` contract more closely than the current `TransformersModel` behavior.

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
3. If `resume_only_model=False`, call existing model load with strict restore semantics
4. If `resume_only_model=True`, call weight-only model load
5. If data skipping is enabled, read progress metadata from `training_state.pt`
6. Recreate the dataloader with skip arguments applied

This keeps the training loop explicit and compatible with current Twinkle examples.

## Server and Client Behavior

Server-side checkpoint save/load behavior should preserve current APIs while adding richer metadata.

### Save path

When server-side save endpoints request optimizer save:

- save the model checkpoint as today
- save `optimizer.pt`, `scheduler.pt`, and `training_state.pt`
- persist checkpoint metadata through the existing checkpoint manager

### Load path

Current `load_optimizer=True` behavior is retained as the trigger for strict model-state restore.

The new resume parameters are primarily a training-entrypoint concern. They orchestrate whether to:

- call strict resume
- call weight-only resume
- request data skipping

The underlying server model APIs do not need a new trainer object to support this.

## Compatibility Strategy

### Existing checkpoints

Existing checkpoints remain loadable in weight-only mode.

Examples:

- `model.load(path, load_optimizer=False)` continues to work
- inference-only consumers remain unaffected

### Old checkpoints under strict resume

Old checkpoints that lack `training_state.pt` are not valid for strict `TransformersModel` resume.

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

### Model-state save/load

- `training_state.pt` is written when optimizer save is enabled
- scaler, RNG, `cur_step`, and accumulation settings are restored
- strict resume fails when required files are missing

### Weight-only compatibility

- legacy checkpoints still load in weight-only mode
- `resume_only_model=True` restores weights without optimizer and RNG

### Data progress skipping

- map-style datasets skip consumed data correctly
- skip behavior remains correct with device-mesh sharding
- iterable and streaming datasets emit warnings and continue without skipping

### Failure cases

- missing progress metadata when data skipping is requested
- malformed `training_state.pt`
- mismatch between requested strict resume and available checkpoint contents

## Implementation Outline

1. Extend `TransformersModel.save/load` to persist and restore `training_state.pt`
2. Add helper methods for RNG save/load and training-state serialization
3. Extend dataloader and sampler stack to support skip arguments for map-style datasets
4. Thread `resume_from_checkpoint`, `resume_only_model`, and `ignore_data_skip` through existing training entrypoints
5. Add warnings for unsupported iterable/streaming data skipping
6. Update docs and examples to prefer trainer-level resume parameters over ad hoc `model.load(..., load_optimizer=True)` logic

## User Guidance

Recommended guidance text:

- To resume training, keep other parameters unchanged and provide `resume_from_checkpoint`
- `resume_only_model=False` performs full resume
- `resume_only_model=True` restores only model weights
- `ignore_data_skip=True` disables progress restore and starts from step 0
- Iterable and streaming datasets do not support consumed-data skipping and will resume without skipping data
