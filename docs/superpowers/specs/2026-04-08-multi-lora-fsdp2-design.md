# Multi-LoRA Transformers FSDP2 Support Design

## Summary

Enable `MultiLoraTransformersModel` to run with FSDP2 model sharding while keeping `AccelerateStrategy` as the default strategy. The initial scope is intentionally narrow: support the training critical path and LoRA weight persistence for the transformers multi-LoRA backend, without expanding into sampler sync or broader checkpoint compatibility work.

## Goal

Make the following flow work when `device_mesh.fsdp_world_size > 1`:

- construct `MultiLoraTransformersModel`
- add a LoRA adapter
- run `forward`, `calculate_loss`, `backward`, `step`, and `zero_grad`
- save and load LoRA adapter weights
- remove an adapter and restore the slot to its initial state

## Non-Goals

- No new sampler or checkpoint-engine synchronization behavior
- No dedicated `native_fsdp` path for multi-LoRA in this iteration
- No broader refactor to merge `MultiLoraTransformersModel` into the single-adapter transformers stack
- No new guarantees around optimizer state migration across sharding layouts
- No changes to megatron multi-LoRA behavior

## Current Problem

`MultiLoraTransformersModel` currently blocks FSDP usage and bypasses the normal transformers wrapping lifecycle:

- it asserts that FSDP is unsupported during construction
- it always uses `AccelerateStrategy(device_mesh=None)`
- it eagerly wraps the model in `__init__`
- multi-LoRA save/load helpers assume local tensors and perform direct `parameter.data.copy_` writes

That combination prevents Accelerate FSDP2 from sharding the model and makes adapter state handling unsafe once LoRA parameters become sharded tensors.

## Proposed Approach

Keep the existing class and multi-slot adapter design, but make it FSDP2-compatible under the default Accelerate path.

### 1. Strategy and wrapping lifecycle

Update `MultiLoraTransformersModel` so it no longer hard-disables FSDP.

- remove the constructor assert that rejects FSDP
- instantiate `AccelerateStrategy` with the real `device_mesh`
- keep `multi_adapter.patch(self.model)` before any wrapping so the sharded model includes all LoRA slots
- stop eager wrapping in `__init__`
- implement `_lazy_wrap_model()` by reusing the parent lifecycle so wrapping happens after optimizers are created

This preserves the current default strategy choice while allowing Accelerate's FSDP2 plugin to own sharding.

### 2. DTensor-safe multi-LoRA weight access

Adjust `MultiLora` helper methods so they work when LoRA parameters are represented as sharded tensors.

Methods that need FSDP2-aware handling:

- `save_initial_weights`
- `_load_initial_weights`
- `set_state_dict`
- `get_state_dict`
- `save_lora_converter`

Design rules:

- reading weights should operate on a local or reconstructed tensor view rather than assuming a plain parameter tensor
- writing weights should detect DTensor-like parameters and transform incoming checkpoint tensors to the target layout before copy
- LoRA rank slicing rules remain unchanged for `lora_A`, `lora_B`, `lora_embedding_A`, and `lora_embedding_B`

The intent is not to create a fully generic distributed checkpoint layer, only to make the current LoRA slot persistence logic safe for FSDP2.

### 3. Multi-LoRA load path

Extend `MultiLoraTransformersModel.load()` to mirror the existing single-adapter transformers FSDP2 behavior.

Current single-adapter transformers code already converts CPU adapter weights into the destination distributed layout before applying them. The multi-LoRA path should reuse the same idea, but route tensors into the tenant-owned slot inside `MultiLora` instead of using `set_peft_model_state_dict` directly.

Expected behavior:

- checkpoint weights load on CPU first
- keys are mapped into the real internal adapter slot
- tensors are distributed as needed to match the wrapped parameter layout
- values are copied into the correct LoRA slot for the tenant adapter

### 4. Test coverage

Add the smallest set of tests that proves the supported scope.

Required checks:

- constructing `MultiLoraTransformersModel` with an FSDP-enabled mesh no longer fails
- adding an adapter and running one training step succeeds under Accelerate FSDP2
- `get_state_dict` followed by `load` round-trips LoRA weights correctly
- removing an adapter restores the slot to its initial values

Test strategy:

- prefer a focused regression test near the transformers model tests
- keep the model and world size small
- assert behavior on LoRA slot tensors, not just that no exception was raised

## Implementation Outline

1. Update `MultiLoraTransformersModel.__init__` to keep the default Accelerate path but pass through `device_mesh`
2. Move model wrapping out of construction and back into `_lazy_wrap_model`
3. Add FSDP2-safe tensor conversion helpers in `MultiLora`
4. Update multi-LoRA load and slot-reset code to use those helpers
5. Add regression tests for FSDP2 construction, train step, save/load, and remove

## Risks

- LoRA parameter layouts under Accelerate FSDP2 may differ from the assumptions used by direct slicing in patched LoRA forward code
- some helper methods may reconstruct full tensors when only local shards are needed, which can increase test-time memory usage
- eager assumptions about adapter activation order may be exposed once wrapping becomes lazy again

## Deferred Work

These should stay out of this change unless the minimal support cannot be made correct without them:

- sampler LoRA sync compatibility
- optimizer state load/save guarantees under FSDP2
- memory-efficient initialization specific to multi-LoRA
- a deeper unification with `TransformersModel` adapter management

## Acceptance Criteria

This design is complete when:

- `MultiLoraTransformersModel` can run the narrow training flow under Accelerate FSDP2
- multi-LoRA adapter save/load works for the supported LoRA tensor types
- regression tests cover the new supported behavior
- unsupported areas remain explicitly unchanged
