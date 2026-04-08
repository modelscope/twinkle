# Multi-LoRA FSDP2 SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MultiLoraTransformersModel` support SFT under FSDP2 for both `AccelerateStrategy` and `NativeFSDPStrategy`, including adapter save/load/remove semantics.

**Architecture:** Reuse one shared FSDP2 compatibility layer instead of building separate accelerate-only and native-only implementations. Keep the existing multi-slot adapter model, but move model wrapping back into the shared transformers lifecycle and teach `MultiLora` how to read and write LoRA slot weights safely when parameters are sharded.

**Tech Stack:** Python 3.11+, PyTorch distributed FSDP2, Accelerate, PEFT LoRA, pytest, Twinkle transformers model stack

---

## References

- Spec: `docs/superpowers/specs/2026-04-08-multi-lora-fsdp2-design.md`
- Use `@test-driven-development` for every behavior change.
- Use `@verification-before-completion` before claiming any phase is done.

## File Map

- Modify: `src/twinkle/model/transformers/multi_lora_transformers.py`
  - Rejoin the shared strategy-selection and lazy-wrap lifecycle.
- Modify: `src/twinkle/model/multi_lora.py`
  - Add sharding-safe helpers for LoRA slot reads, writes, save/load, and reset.
- Create: `tests/model/transformers/test_multi_lora_fsdp2_sft.py`
  - End-to-end regression coverage for accelerate and native FSDP2 SFT.
- Optional Create: `tests/model/test_multi_lora_state.py`
  - Fast, non-distributed coverage for slot save/load/remove semantics if the distributed tests are too expensive to debug first.

## Assumptions

- GPU-backed tests are acceptable for the FSDP2 integration path.
- Test model path should resolve from `TEST_MODEL_ID` with an offline-cache fallback before attempting any network access.
- GRPO is out of scope for this plan.

### Task 1: Create the Accelerate FSDP2 SFT regression test scaffold

**Files:**
- Create: `tests/model/transformers/test_multi_lora_fsdp2_sft.py`
- Reference: `cookbook/transformers/fsdp2.py`
- Reference: `cookbook/transformers/sp_fsdp_dense.py`

- [ ] **Step 1: Write the failing accelerate SFT test**

```python
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires 2+ GPUs')
def test_multi_lora_accelerate_fsdp2_sft_round_trip(tmp_path):
    model = build_multi_lora_model(strategy='accelerate')
    model.add_adapter_to_model('default', build_lora_config(), gradient_accumulation_steps=1)
    model.set_loss('CrossEntropyLoss', adapter_name='default')
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')

    batch = build_sft_batch()
    model.forward_backward(inputs=batch, adapter_name='default')
    model.clip_grad_and_step(adapter_name='default')

    state_before = model.get_state_dict(adapter_name='default')
    model.save('ckpt', output_dir=str(tmp_path), adapter_name='default')
    model.remove_adapter('default')
    model.add_adapter_to_model('default', build_lora_config(), gradient_accumulation_steps=1)
    model.load('ckpt', output_dir=str(tmp_path), adapter_name='default')
    state_after = model.get_state_dict(adapter_name='default')

    assert_same_lora_state(state_before, state_after)
```

- [ ] **Step 2: Run the accelerate test to verify it fails**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -k accelerate -v`
Expected: FAIL because `MultiLoraTransformersModel` still blocks or bypasses FSDP2 wrapping.

- [ ] **Step 3: Add test helpers inside the same file**

```python
def build_multi_lora_model(strategy: str):
    model_path = get_model_path()
    mesh = DeviceMesh.from_sizes(world_size=2, fsdp_size=2)
    return MultiLoraTransformersModel(
        model_id=model_path,
        device_mesh=mesh,
        strategy=strategy,
    )


def build_sft_batch():
    return [{
        'input_ids': [1, 2, 3, 4],
        'labels': [1, 2, 3, 4],
    }]
```

- [ ] **Step 4: Re-run the accelerate test to confirm the failure is the intended one**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -k accelerate -v`
Expected: FAIL in model setup or FSDP2 execution, not from missing helper functions or syntax errors.

- [ ] **Step 5: Commit the red test scaffold**

```bash
git add tests/model/transformers/test_multi_lora_fsdp2_sft.py
git commit -m "test: add accelerate multi-lora fsdp2 sft regression"
```

### Task 2: Make `MultiLoraTransformersModel` participate in the shared wrap lifecycle

**Files:**
- Modify: `src/twinkle/model/transformers/multi_lora_transformers.py`
- Reference: `src/twinkle/model/transformers/transformers.py`
- Test: `tests/model/transformers/test_multi_lora_fsdp2_sft.py`

- [ ] **Step 1: Write the smallest failing assertion for strategy selection and lazy wrap**

```python
def test_multi_lora_accelerate_fsdp2_uses_device_mesh():
    model = build_multi_lora_model(strategy='accelerate')
    assert model.strategy.device_mesh is not None
    assert model._model_wrapped is False
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -k uses_device_mesh -v`
Expected: FAIL because the class still hardcodes `AccelerateStrategy(device_mesh=None)` and eagerly wraps in `__init__`.

- [ ] **Step 3: Implement the minimal lifecycle fix**

```python
class MultiLoraTransformersModel(TransformersModel, PreTrainedModel):
    def __init__(..., strategy: Literal['accelerate', 'native_fsdp'] = 'accelerate', fsdp_config=None, ...):
        self._fsdp_config = dict(fsdp_config or {})
        self._decide_strategy(strategy)
        ...
        self.model = self.multi_adapter.patch(self.model)
        self._model_wrapped = False

    def _lazy_wrap_model(self):
        return super()._lazy_wrap_model()
```

- [ ] **Step 4: Run the accelerate tests again**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -k "uses_device_mesh or accelerate" -v`
Expected: first lifecycle assertion passes; full accelerate round-trip test still fails in slot save/load or sharded tensor handling.

- [ ] **Step 5: Commit the lifecycle change**

```bash
git add src/twinkle/model/transformers/multi_lora_transformers.py tests/model/transformers/test_multi_lora_fsdp2_sft.py
git commit -m "feat: reuse shared fsdp lifecycle for multi-lora transformers"
```

### Task 3: Add sharding-safe LoRA slot tensor helpers in `MultiLora`

**Files:**
- Modify: `src/twinkle/model/multi_lora.py`
- Test: `tests/model/transformers/test_multi_lora_fsdp2_sft.py`
- Optional Test: `tests/model/test_multi_lora_state.py`

- [ ] **Step 1: Write a failing state round-trip test that isolates slot semantics**

```python
def test_multi_lora_state_dict_round_trip_preserves_rank_slices(tmp_path):
    model = build_multi_lora_model(strategy='accelerate')
    model.add_adapter_to_model('default', build_lora_config(r=4), gradient_accumulation_steps=1)
    state = model.get_state_dict(adapter_name='default')
    assert state
    assert all('.default.' not in key for key in state)
```

- [ ] **Step 2: Run the state round-trip test to verify it fails for the expected reason**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -k round_trip_preserves_rank_slices -v`
Expected: FAIL because current multi-LoRA save/load helpers assume local tensors and direct `parameter.data` writes.

- [ ] **Step 3: Add minimal helper methods for slot IO**

```python
def _read_param_tensor(self, parameter):
    return torch_util.to_local_tensor(parameter)


def _write_param_tensor(self, parameter, value):
    if hasattr(parameter, 'device_mesh') and hasattr(parameter, 'placements'):
        value = distribute_tensor(value.to(parameter.device), parameter.device_mesh, parameter.placements)
    parameter.data.copy_(value)
```

- [ ] **Step 4: Refactor `save_initial_weights`, `_load_initial_weights`, `set_state_dict`, `get_state_dict`, and `save_lora_converter` to use the new helpers**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -k accelerate -v`
Expected: the accelerate SFT round-trip test passes.

- [ ] **Step 5: Commit the shared slot-IO layer**

```bash
git add src/twinkle/model/multi_lora.py tests/model/transformers/test_multi_lora_fsdp2_sft.py
git commit -m "feat: support sharded multi-lora slot state io"
```

### Task 4: Add native FSDP2 SFT regression coverage

**Files:**
- Modify: `tests/model/transformers/test_multi_lora_fsdp2_sft.py`
- Reference: `cookbook/transformers/sp_fsdp_dense.py`

- [ ] **Step 1: Write the failing native FSDP SFT test**

```python
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires 2+ GPUs')
def test_multi_lora_native_fsdp_sft_round_trip(tmp_path):
    model = build_multi_lora_model(strategy='native_fsdp')
    model.add_adapter_to_model('default', build_lora_config(), gradient_accumulation_steps=1)
    model.set_loss('CrossEntropyLoss', adapter_name='default')
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')
    model.forward_backward(inputs=build_sft_batch(), adapter_name='default')
    model.clip_grad_and_step(adapter_name='default')
    model.save('native-ckpt', output_dir=str(tmp_path), adapter_name='default')
```

- [ ] **Step 2: Run the native test to verify it fails**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -k native -v`
Expected: FAIL because native FSDP has not yet been exercised through multi-LoRA wrapping, optimizer rebinding, or slot restore.

- [ ] **Step 3: Add native-specific assertions before implementation**

```python
assert model._model_wrapped is False
assert model.device_mesh.fsdp_world_size == 2
```

- [ ] **Step 4: Re-run the native test to verify the failure still points at native FSDP support**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -k native -v`
Expected: FAIL in native FSDP wrapping or native slot state handling, not in test setup.

- [ ] **Step 5: Commit the native regression scaffold**

```bash
git add tests/model/transformers/test_multi_lora_fsdp2_sft.py
git commit -m "test: add native multi-lora fsdp2 sft regression"
```

### Task 5: Make the native FSDP2 SFT path pass

**Files:**
- Modify: `src/twinkle/model/transformers/multi_lora_transformers.py`
- Modify: `src/twinkle/model/multi_lora.py`
- Test: `tests/model/transformers/test_multi_lora_fsdp2_sft.py`

- [ ] **Step 1: Implement the smallest native FSDP2 compatibility change**

```python
model = MultiLoraTransformersModel(
    ...,
    strategy='native_fsdp',
)
```

Required behavior:
- strategy is selected through `_decide_strategy`
- wrapping still happens only through `_lazy_wrap_model`
- optimizer binding remains valid after wrap

- [ ] **Step 2: Run the native regression to verify the new failure, if any**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -k native -v`
Expected: either PASS or a narrower failure around native slot IO / sharded forward behavior.

- [ ] **Step 3: Fix the minimal remaining native-specific issue**

```python
if strategy == 'native_fsdp':
    # Keep multi-LoRA slot tensors readable/writable after fully_shard.
    ...
```

- [ ] **Step 4: Run the full SFT regression file**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -v`
Expected: PASS for both accelerate and native SFT tests.

- [ ] **Step 5: Commit the native support**

```bash
git add src/twinkle/model/transformers/multi_lora_transformers.py src/twinkle/model/multi_lora.py tests/model/transformers/test_multi_lora_fsdp2_sft.py
git commit -m "feat: support native fsdp2 sft for multi-lora transformers"
```

### Task 6: Final verification and cleanup

**Files:**
- Modify: `src/twinkle/model/transformers/multi_lora_transformers.py`
- Modify: `src/twinkle/model/multi_lora.py`
- Modify: `tests/model/transformers/test_multi_lora_fsdp2_sft.py`

- [ ] **Step 1: Run the shared targeted verification suite**

Run: `python -m pytest tests/model/transformers/test_multi_lora_fsdp2_sft.py -v`
Expected: PASS

- [ ] **Step 2: Run one adjacent regression if save/load behavior was touched deeply**

Run: `python -m pytest tests/sampler/test_weight_sync.py -k lora -v`
Expected: PASS or SKIP, with no new failures caused by LoRA state handling changes.

- [ ] **Step 3: Inspect the final diff for scope discipline**

Run: `git diff --stat HEAD~1..HEAD`
Expected: only multi-LoRA transformers model code and the new SFT regression tests are touched.

- [ ] **Step 4: Document verification evidence in the final handoff**

Required notes:
- exact test commands run
- whether each command passed, failed, or skipped
- any remaining risks, especially GPU-only coverage limitations

- [ ] **Step 5: Commit cleanup if needed**

```bash
git add src/twinkle/model/transformers/multi_lora_transformers.py src/twinkle/model/multi_lora.py tests/model/transformers/test_multi_lora_fsdp2_sft.py
git commit -m "test: verify multi-lora fsdp2 sft coverage"
```
