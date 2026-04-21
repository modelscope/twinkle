# Unified `resume_from_checkpoint` API — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `load_training_state` / `read_training_progress` with a single `resume_from_checkpoint` method on both model backends and dataloader, so callers orchestrate with two lines instead of five.

**Architecture:** Add `resume_from_checkpoint` as an abstract method on `TwinkleModel`. Each backend (Transformers, Megatron) implements it to restore its own state internally and return a common `{cur_step, consumed_train_samples, gradient_accumulation_steps}` dict. DataLoader gets a matching `resume_from_checkpoint` that wraps `skip_consumed_samples`. Server/client/cookbook/docs updated to match.

**Tech Stack:** Python, PyTorch, FastAPI, Pydantic, PEFT, Megatron-Core

**Spec:** `docs/superpowers/specs/2026-04-21-unified-resume-api-design.md`

---

## Chunk 1: Core Model API

### Task 1: Add `resume_from_checkpoint` to TwinkleModel base class

**Files:**
- Modify: `src/twinkle/model/base.py:86-88`

- [ ] **Step 1: Add abstract method after `get_state_dict`**

In `src/twinkle/model/base.py`, insert after line 88 (`get_state_dict`):

```python
@abstractmethod
def resume_from_checkpoint(self, checkpoint_dir: str, *, resume_only_model: bool = False, **kwargs) -> Dict[str, Any]:
    ...
```

- [ ] **Step 2: Verify no import changes needed**

`Dict` and `Any` are already imported on line 4. No changes needed.

- [ ] **Step 3: Commit**

```bash
git add src/twinkle/model/base.py
git commit -m "feat: add resume_from_checkpoint abstract method to TwinkleModel base"
```

---

### Task 2: Implement `resume_from_checkpoint` in TransformersModel

**Files:**
- Modify: `src/twinkle/model/transformers/transformers.py:1063-1100`

- [ ] **Step 1: Delete `read_training_progress` method (lines 1063-1075)**

Remove the entire `read_training_progress` method.

- [ ] **Step 2: Delete `load_training_state` method (lines 1078-1100)**

Remove the entire `load_training_state` method.

- [ ] **Step 3: Add `resume_from_checkpoint` method**

Insert at the same location where the deleted methods were:

```python
@remote_function()
def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
    adapter_name = kwargs.get('adapter_name', '')

    has_adapter = (
        os.path.exists(os.path.join(checkpoint_dir, 'adapter_model.safetensors'))
        or os.path.exists(os.path.join(checkpoint_dir, 'adapter_model.bin'))
    )
    if has_adapter:
        self.load(checkpoint_dir, adapter_name=adapter_name)

    trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)

    if not resume_only_model:
        adapter_name = adapter_name or self._get_default_group()
        optimizer_config = self.optimizer_group[adapter_name]
        self._load_optimizer(checkpoint_dir, adapter_name=adapter_name)
        self._load_scaler_state(checkpoint_dir)
        self._load_rng_state(checkpoint_dir)
        optimizer_config.cur_step = trainer_state['cur_step']
        optimizer_config.gradient_accumulation_steps = trainer_state['gradient_accumulation_steps']

    return {
        'cur_step': trainer_state['cur_step'],
        'consumed_train_samples': trainer_state['consumed_train_samples'],
        'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
    }
```

- [ ] **Step 4: Verify `json` and `os` imports exist**

`json` is imported at line 4, `os` at line 6. No changes needed.

- [ ] **Step 5: Commit**

```bash
git add src/twinkle/model/transformers/transformers.py
git commit -m "feat(transformers): replace load_training_state/read_training_progress with resume_from_checkpoint"
```

---

### Task 3: Implement `resume_from_checkpoint` in MegatronModel + update `save`

**Files:**
- Modify: `src/twinkle/model/megatron/megatron.py:762-821` (save), add new method after `load`

- [ ] **Step 1: Update `save()` to write `trainer_state.json`**

In `src/twinkle/model/megatron/megatron.py`, find the `if save_optimizer:` block (around line 810). After the `_save_mcore_optimizer` call and before the barrier, add:

```python
        trainer_state = {
            'checkpoint_version': 1,
            'cur_step': optimizer_config.cur_step,
            'consumed_train_samples': kwargs.get('consumed_train_samples', 0),
            'gradient_accumulation_steps': optimizer_config.gradient_accumulation_steps,
        }
        state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            with open(state_path, 'w') as f:
                json.dump(trainer_state, f, indent=2)
```

- [ ] **Step 2: Add `resume_from_checkpoint` method**

Insert after the `load` method (after line 867):

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

- [ ] **Step 3: Verify `json` import exists**

`json` is imported at line 3. No changes needed.

- [ ] **Step 4: Commit**

```bash
git add src/twinkle/model/megatron/megatron.py
git commit -m "feat(megatron): add resume_from_checkpoint and save trainer_state.json"
```

---

### Task 4: Add `resume_from_checkpoint` to DataLoader

**Files:**
- Modify: `src/twinkle/dataloader/dataloader.py` (after `skip_consumed_samples`, around line 152)

- [ ] **Step 1: Add method after `skip_consumed_samples`**

```python
@remote_function()
def resume_from_checkpoint(self, consumed_train_samples, **kwargs):
    self.skip_consumed_samples(consumed_train_samples)
```

- [ ] **Step 2: Commit**

```bash
git add src/twinkle/dataloader/dataloader.py
git commit -m "feat(dataloader): add resume_from_checkpoint wrapping skip_consumed_samples"
```

---

## Chunk 2: Server, Client, Types

### Task 5: Update Pydantic types

**Files:**
- Modify: `src/twinkle_client/types/model.py:92-105` (request types), `231-233` (response type)

- [ ] **Step 1: Delete `LoadTrainingStateRequest` (lines 92-97) and `ReadTrainingProgressRequest` (lines 100-105)**

Remove both request classes.

- [ ] **Step 2: Add `ResumeFromCheckpointRequest`**

Insert at the same location:

```python
class ResumeFromCheckpointRequest(BaseModel):
    """Request for /resume_from_checkpoint endpoint."""
    name: str
    adapter_name: str = ''
    resume_only_model: bool = False
```

- [ ] **Step 3: Rename `TrainingProgressResponse` docstring (line 232)**

Update the docstring from `"Response for /read_training_progress endpoint"` to `"Response for /resume_from_checkpoint endpoint"`. Keep the class name and `result` field unchanged.

- [ ] **Step 4: Commit**

```bash
git add src/twinkle_client/types/model.py
git commit -m "feat(types): replace training state request types with ResumeFromCheckpointRequest"
```

---

### Task 6: Update server endpoints

**Files:**
- Modify: `src/twinkle/server/model/twinkle_handlers.py:352-402`

- [ ] **Step 1: Delete `load_training_state` endpoint (lines 352-376)**

Remove the entire endpoint function.

- [ ] **Step 2: Delete `read_training_progress` endpoint (lines 378-402)**

Remove the entire endpoint function.

- [ ] **Step 3: Add `resume_from_checkpoint` endpoint**

Insert at the same location, following the existing endpoint pattern:

```python
@app.post('/twinkle/resume_from_checkpoint', response_model=types.TrainingProgressResponse)
async def resume_from_checkpoint(
    request: Request,
    body: types.ResumeFromCheckpointRequest,
    self: ModelManagement = Depends(self_fn),
):
    token = await self._on_request_start(request)

    async def _task():
        checkpoint_dir = self._resolve_checkpoint_dir(body.name)
        result = self.model.resume_from_checkpoint(
            checkpoint_dir,
            resume_only_model=body.resume_only_model,
            adapter_name=body.adapter_name or token,
        )
        return types.TrainingProgressResponse(result=result)

    return await run_task(self.schedule_task_and_wait(_task, task_type='resume'))
```

Note: Check how `load_training_state` resolves `checkpoint_dir` from `body.name` — replicate the same pattern. If there's a `_resolve_checkpoint_dir` helper, use it. Otherwise inline the resolution logic (typically `os.path.join(output_dir, name)` or direct path).

- [ ] **Step 4: Commit**

```bash
git add src/twinkle/server/model/twinkle_handlers.py
git commit -m "feat(server): replace training state endpoints with /resume_from_checkpoint"
```

---

### Task 7: Update client SDK

**Files:**
- Modify: `src/twinkle_client/model/multi_lora_transformers.py:192-208`
- Modify: `client_tools/client_generator.py:621-637`

- [ ] **Step 1: Update `src/twinkle_client/model/multi_lora_transformers.py`**

Delete `load_training_state` (lines 192-199) and `read_training_progress` (lines 201-208). Replace with:

```python
def resume_from_checkpoint(self, name: str, *, resume_only_model: bool = False, **kwargs) -> Dict[str, Any]:
    response = http_post(
        url=f'{self.server_url}/resume_from_checkpoint',
        json_data={'name': name, 'adapter_name': self.adapter_name,
                   'resume_only_model': resume_only_model, **kwargs}
    )
    response.raise_for_status()
    return TrainingProgressResponse(**response.json()).result
```

- [ ] **Step 2: Update `client_tools/client_generator.py`**

Delete `load_training_state` (lines 621-628) and `read_training_progress` (lines 630-637). Replace with the same `resume_from_checkpoint` method as above.

- [ ] **Step 3: Commit**

```bash
git add src/twinkle_client/model/multi_lora_transformers.py client_tools/client_generator.py
git commit -m "feat(client): replace training state methods with resume_from_checkpoint"
```

---

## Chunk 3: Cookbook and Documentation

### Task 8: Update cookbook examples

**Files:**
- Modify: `cookbook/transformers/resume_utils.py:16-55`
- Modify: `cookbook/client/twinkle/self_host/self_congnition.py:102-110`

- [ ] **Step 1: Rewrite `resume_from_checkpoint` in `cookbook/transformers/resume_utils.py`**

The old helper function manually orchestrated model + dataloader state. Replace the function body (lines 16-55) with a simplified version that delegates to the new model API:

```python
def resume_from_checkpoint(model, dataloader, checkpoint_path, *, resume_only_model=False,
                           ignore_data_skip=False, adapter_name=None) -> int:
    kwargs = {}
    if adapter_name:
        kwargs['adapter_name'] = adapter_name

    progress = model.resume_from_checkpoint(
        checkpoint_path, resume_only_model=resume_only_model, **kwargs)

    consumed_train_samples = int(progress.get('consumed_train_samples', 0))
    if not ignore_data_skip and consumed_train_samples > 0:
        dataloader.resume_from_checkpoint(consumed_train_samples)

    return consumed_train_samples
```

This keeps the helper for backward compatibility with existing cookbook scripts that call it, but the implementation now delegates to the model's own method.

- [ ] **Step 2: Update `cookbook/client/twinkle/self_host/self_congnition.py`**

Replace the resume block (around lines 102-110):

```python
# Before:
consumed_train_samples = 0
global_step = 0
if resume_path:
    logger.info(f'Resuming model weights from {resume_path}')
    model.load(resume_path)
    trainer_state = model.load_training_state(resume_path)
    dataloader.skip_consumed_samples(trainer_state['consumed_train_samples'])
    consumed_train_samples = int(trainer_state['consumed_train_samples'])
    global_step = int(trainer_state['cur_step'])
```

With:

```python
consumed_train_samples = 0
global_step = 0
if resume_path:
    logger.info(f'Resuming from checkpoint {resume_path}')
    progress = model.resume_from_checkpoint(resume_path)
    dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
    consumed_train_samples = int(progress['consumed_train_samples'])
    global_step = int(progress['cur_step'])
```

- [ ] **Step 3: Commit**

```bash
git add cookbook/transformers/resume_utils.py cookbook/client/twinkle/self_host/self_congnition.py
git commit -m "refactor(cookbook): use model.resume_from_checkpoint API"
```

---

### Task 9: Update documentation

**Files:**
- Modify: `docs/source_en/Components/Model/TransformersModel.md:54-65`
- Modify: `docs/source_zh/组件/模型/TransformersModel.md:54-65`
- Modify: `docs/source_en/Usage Guide/Quick-Start.md:289-296`
- Modify: `docs/source_zh/使用指引/快速开始.md:290-297`
- Modify: `docs/source_en/Usage Guide/Server and Client/Twinkle-Client.md:141,191`
- Modify: `docs/source_zh/使用指引/服务端和客户端/Twinkle客户端.md:141,181`

- [ ] **Step 1: Update English TransformersModel.md (lines 54-65)**

Replace the checkpoint section with:

```markdown
### Checkpoint and Resume

- `model.save(name, save_optimizer=True, consumed_train_samples=...)` saves weights together with optimizer, scheduler, scaler, RNG, and `trainer_state.json`.
- `model.resume_from_checkpoint(checkpoint_dir)` restores full training state (weights, optimizer, scheduler, scaler, RNG) and returns `{'cur_step', 'consumed_train_samples', 'gradient_accumulation_steps'}`.
- `model.resume_from_checkpoint(checkpoint_dir, resume_only_model=True)` loads weights only and returns progress metadata without restoring optimizer state.
- `dataloader.resume_from_checkpoint(consumed_train_samples)` skips already-consumed samples.
```

- [ ] **Step 2: Update Chinese TransformersModel.md (lines 54-65)**

Mirror the English changes in Chinese:

```markdown
### 检查点保存与续训

- `model.save(name, save_optimizer=True, consumed_train_samples=...)` 保存权重、优化器、调度器、scaler、RNG 状态和 `trainer_state.json`。
- `model.resume_from_checkpoint(checkpoint_dir)` 恢复完整训练状态（权重、优化器、调度器、scaler、RNG），返回 `{'cur_step', 'consumed_train_samples', 'gradient_accumulation_steps'}`。
- `model.resume_from_checkpoint(checkpoint_dir, resume_only_model=True)` 仅加载权重并返回进度元数据，不恢复优化器状态。
- `dataloader.resume_from_checkpoint(consumed_train_samples)` 跳过已消费的样本。
```

- [ ] **Step 3: Update Quick-Start docs (EN and ZH)**

In both `docs/source_en/Usage Guide/Quick-Start.md` and `docs/source_zh/使用指引/快速开始.md`, replace `model.load_training_state(resume_path)` references with:

```python
progress = model.resume_from_checkpoint(resume_path)
dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
```

Update the explanatory text accordingly.

- [ ] **Step 4: Update Twinkle-Client docs (EN and ZH)**

In both `docs/source_en/Usage Guide/Server and Client/Twinkle-Client.md` and `docs/source_zh/使用指引/服务端和客户端/Twinkle客户端.md`, replace `model.load_training_state(resume_path)` references with `model.resume_from_checkpoint(resume_path)`.

- [ ] **Step 5: Commit**

```bash
git add docs/
git commit -m "docs: update checkpoint/resume documentation for unified API"
```

---

### Task 10: Final grep verification

- [ ] **Step 1: Verify no stale references remain**

```bash
grep -rn "load_training_state\|read_training_progress" src/ cookbook/ client_tools/ docs/ --include="*.py" --include="*.md"
```

Expected: Only hits in `docs/superpowers/` (our spec/plan files). No hits in source code, cookbook, or user-facing docs.

- [ ] **Step 2: Run pre-commit hooks**

```bash
pre-commit run --all-files
```

Fix any formatting issues (isort, yapf, flake8).

- [ ] **Step 3: Final commit if needed**

```bash
git add -A
git commit -m "chore: fix formatting after resume API refactor"
```
