# Multi-LoRA Transformers FSDP2 SFT Support Design

## Summary

Enable `MultiLoraTransformersModel` to run under FSDP2 for SFT across both `AccelerateStrategy` and `NativeFSDPStrategy`, with staged delivery. Implementation will proceed in this order:

1. `AccelerateStrategy + SFT`
2. `native_fsdp + SFT`

The design centers on a shared FSDP2 compatibility layer for transformers multi-LoRA, so the native FSDP stage builds on the same model lifecycle, adapter-slot semantics, and distributed weight handling established by the accelerate stage.

## Goals

### Final Goal

Make `MultiLoraTransformersModel` work under FSDP2 for SFT across both supported transformers strategies:

- `AccelerateStrategy`
- `NativeFSDPStrategy`

### Delivery Goal

Ship the capability in two stages:

1. `AccelerateStrategy + SFT`
2. `native_fsdp + SFT`

Each stage must leave the shared foundations in a state that later stages can reuse without strategy-specific rewrites.

## Non-Goals

- No megatron multi-LoRA changes in this workstream
- No sampler or checkpoint-engine LoRA sync expansion in this workstream
- No attempt to solve all distributed checkpoint migration cases across arbitrary sharding layouts
- No requirement to support multi-adapter concurrent training in the initial stages
- No GRPO support in this workstream
- No large-scale performance tuning as part of correctness work
- No deep rewrite that merges `MultiLoraTransformersModel` into the single-adapter transformers implementation

## Current Problem

`MultiLoraTransformersModel` currently does not participate correctly in the transformers FSDP2 lifecycle.

Current blockers:

- construction hard-rejects FSDP via an assert
- the class always uses `AccelerateStrategy(device_mesh=None)`
- the model is eagerly wrapped in `__init__` instead of participating in lazy wrap
- multi-LoRA weight helpers assume local tensors and directly mutate `parameter.data`
- save, load, and slot-reset behavior is not strategy-aware

Because of that:

- `AccelerateStrategy` cannot shard the model for multi-LoRA training
- `NativeFSDPStrategy` is not wired into multi-LoRA at all
- adapter slot persistence is unsafe once LoRA parameters become DTensors or other sharded parameter forms
- SFT under FSDP2 is currently blocked for both strategy paths

## Design Principles

- Keep the existing multi-slot adapter model: tenants bind to preallocated internal LoRA slots
- Build one shared FSDP2 compatibility layer instead of separate accelerate-only and native-only implementations
- Let strategy differences stay in strategy code paths, not in duplicated multi-LoRA business logic
- Stage rollout by strategy, but avoid temporary patches that block later phases
- Prefer the smallest regression tests that prove correctness at each phase

## Proposed Architecture

The work is split into two layers:

1. A shared FSDP2 compatibility foundation for transformers multi-LoRA
2. A staged rollout over SFT for accelerate and native FSDP2

### Shared Foundation

The shared foundation must be correct before later phases can be added safely.

#### 1. Unified strategy selection and lazy wrap lifecycle

`MultiLoraTransformersModel` should stop managing wrapping as a special case.

Required changes:

- remove the constructor assert that blocks FSDP
- stop forcing `AccelerateStrategy(device_mesh=None)`
- let the class honor the requested strategy:
  - default remains `AccelerateStrategy`
  - `strategy='native_fsdp'` must instantiate `NativeFSDPStrategy`
- keep `multi_adapter.patch(self.model)` before wrapping so LoRA slots exist in the wrapped model graph
- move wrapping back into `_lazy_wrap_model()` so optimizer creation and strategy wrapping follow the same lifecycle as transformers models

This is the main prerequisite for supporting both accelerate and native FSDP2 without forking the class.

#### 2. Strategy-aware multi-LoRA parameter access

`MultiLora` needs a small internal abstraction for reading and writing LoRA slot tensors under both unsharded and sharded parameter representations.

This abstraction should support:

- reading a saveable tensor view from LoRA slot parameters
- writing checkpoint tensors into LoRA slot parameters with the correct target layout
- restoring initial slot values when an adapter is removed
- preserving existing rank slicing rules for:
  - `lora_A`
  - `lora_B`
  - `lora_embedding_A`
  - `lora_embedding_B`

The goal is not a generic distributed checkpoint layer. The goal is to make the current multi-LoRA slot logic safe under FSDP2.

#### 3. Stable adapter-slot state machine after wrapping

The tenant-to-slot model is part of the multi-LoRA contract and should remain unchanged.

The following behaviors must stay valid after wrapping:

- `activate_adapter`
- `deactivate_adapter`
- `save_context`
- `remove_adapter`

This is important even in SFT because adapter activation, save/load, and slot reset must behave the same before and after wrapping.

#### 4. Unified save/load/remove semantics

The same slot-aware semantics should apply regardless of strategy.

Required behaviors:

- `get_state_dict` returns the tenant adapter's LoRA state with correct rank slicing
- `load` maps checkpoint weights into the correct internal slot
- `remove_adapter` restores the slot to its initial weights
- save/load logic works under both wrapped and unwrapped model states

Single-adapter transformers already has FSDP2-aware load behavior. Multi-LoRA should reuse that idea, but route tensors through tenant-owned slots instead of direct single-adapter PEFT application.

#### 5. Reusable test scaffolding

Even though rollout is staged, the test base should be reusable from the start.

Shared fixtures/helpers should cover:

- a minimal FSDP2-capable device mesh
- a minimal multi-LoRA transformers model builder
- adapter-slot inspection helpers
- minimal SFT input samples

This avoids rebuilding test infrastructure at every phase.

## Staged Rollout

### Phase 1: `AccelerateStrategy + SFT`

This is the first delivery milestone and the narrowest supported training loop.

Supported flow:

- construct `MultiLoraTransformersModel` with accelerate FSDP2
- add a single LoRA adapter
- run SFT training:
  - `forward`
  - `calculate_loss(CrossEntropyLoss)`
  - `backward`
  - `clip_grad_norm`
  - `step`
  - `zero_grad`
- save and load LoRA adapter state
- remove the adapter and restore the slot

Implementation focus:

- strategy/lazy-wrap integration
- FSDP2-safe slot state persistence
- SFT regression coverage

Explicitly out of scope for this phase:

- GRPO
- multi-adapter concurrent training
- sampler sync

### Phase 2: `native_fsdp + SFT`

Add native FSDP2 support by reusing the shared foundation.

Supported flow:

- construct `MultiLoraTransformersModel` with `strategy='native_fsdp'`
- add a single LoRA adapter
- run the same minimal SFT loop as Phase 1
- save, load, and remove LoRA adapters correctly

Implementation focus:

- compatibility with `NativeFSDPStrategy.wrap_model`
- parameter layout handling after `fully_shard`
- optimizer rebinding and lazy-wrap correctness
- native SFT regression coverage

Key risk areas:

- wrapped parameter representation under native FSDP2
- optimizer param-group rebinding after wrapping
- LoRA forward patch assumptions when parameters are sharded

## Test Plan

Tests should expand phase by phase, but stay narrow and behavior-oriented.

### Shared test requirements

- use the smallest model and mesh that still exercises the target strategy
- assert adapter-slot tensor behavior, not only absence of exceptions
- validate save/load round-trip and slot reset where relevant

### Phase-specific checks

#### Phase 1

- accelerate FSDP2 construction succeeds
- SFT forward-backward-step succeeds
- LoRA state round-trips through save/load
- `remove_adapter` restores initial slot values

#### Phase 2

- native FSDP2 construction succeeds
- native SFT forward-backward-step succeeds
- native save/load/remove semantics are correct

## Risks

- patched LoRA forward code may assume local tensor access patterns that do not hold after sharding
- accelerate and native FSDP2 may expose LoRA parameters through different tensor representations
- some helper logic may accidentally reconstruct full tensors when only local shards are needed
- lazy wrap may surface ordering issues around optimizer creation, adapter activation, or template hooks

## Deferred Work

These should remain outside this design unless later stages prove them necessary for correctness:

- GRPO support for either strategy
- sampler or checkpoint-engine LoRA synchronization enhancements
- memory-efficient-init customization specific to transformers multi-LoRA
- megatron multi-LoRA parity work
- broader adapter lifecycle redesign beyond the current slot model
- large-scale performance benchmarking or throughput tuning

## Acceptance Criteria

This design is complete when both stages are delivered in order and each stage has dedicated regression coverage:

1. `AccelerateStrategy + SFT`
2. `native_fsdp + SFT`

At the end of the rollout:

- `MultiLoraTransformersModel` supports FSDP2 under both accelerate and native strategies
- SFT runs through the supported transformers multi-LoRA paths under both strategies
- save, load, and remove adapter semantics remain correct under FSDP2
- unsupported areas remain explicitly unchanged
