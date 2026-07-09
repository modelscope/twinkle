# Twinkle Training Script Skill

You are an expert at writing training scripts for the Twinkle framework.

## CRITICAL RULES

1. **Model/Dataset names MUST use full org/name format**: `Qwen/Qwen3.5-4B`, NOT `Qwen3.5-4B`
2. **Name resolution workflow** (MUST follow when user gives a model or dataset name):
   - If user says "Qwen3.5-4B" or any short/ambiguous name → call `search_models(query='Qwen3.5-4B')` to get full ID
   - If user says "gsm8k" → call `search_datasets(query='gsm8k')` to get full ID like `modelscope/gsm8k`
   - If server is running → also call `list_supported_models()` to verify the model is deployed
   - **Never guess model/dataset full names** — always search to confirm
3. **Scripts MUST use Server Mode** (`twinkle_client` for model + `twinkle` for data)
4. **DO NOT modify the Twinkle SDK** (`src/twinkle/` or `src/twinkle_client/`)
5. **Every script MUST register graceful shutdown** via `rt.register_graceful_shutdown(model, dataloader)`
6. **All imports MUST be explicit** — never use a class/function without importing it first
7. **`batch_size` constraints** — **Always set `drop_last=True`** in DataLoader.
   - **All modes**: `batch_size >= model_dp` (number of model data-parallel GPUs). Call `list_supported_models()` to get GPU count.
   - **GRPO**: `batch_size >= sampler_dp` (sampler's data-parallel worker count = `sampler_gpus / tp`). The sampler dispatches input batch across `dp` workers, each worker must get at least 1 item. This is often the tighter constraint (e.g., 6 sampler GPUs with tp=1 → need batch_size >= 6).
8. **`rt.start()` MUST be called BEFORE `model.add_adapter_to_model()`** — `add_adapter_to_model` triggers NCCL init (60-120s) and no logs appear until `rt.start()` runs
9. **`metric.result` values are auto-converted** inside `rt.log_metrics()` — no manual `float()` needed
10. **NEVER use float format specifiers** (like `:.4f`, `:.2e`) on metric values in `print()` — they may be strings. Just use `{loss}`
11. **NEVER access internal fields** of model/optimizer/scheduler objects (e.g. `model.optimizer.param_groups[0]['lr']`). Training runs on a remote Ray cluster — only public API methods are available. Use `model.calculate_metric()` to get metrics like loss/lr
12. **Log ALL available metrics** via `rt.log_metrics(step=step, total_steps=MAX_STEPS, **metric.result)`. NEVER cherry-pick only `loss` — always pass the full `metric.result` dict. Different training types produce different metrics:
    - **SFT**: loss, grad_norm, lr
    - **GRPO**: loss, reward, reward_std, kl, entropy, grad_norm, lr
    - **DPO**: loss, chosen_reward, rejected_reward, reward_margin, grad_norm, lr
    - Use `**metric.result` to capture all of them automatically
13. **Every script MUST include resume logic** after DataLoader creation. This enables seamless continuation when the script is auto-fixed and restarted:
    ```python
    resume = rt.get_resume_info()
    global_step = resume['last_step']
    if global_step > 0:
        dataloader.skip_consumed_samples(global_step * BATCH_SIZE)
        print(f'[twinkle] Resuming from step {global_step}')
    ```

## Pre-Training Planning

> **Cloud shortcut:** If using `base_url='http://www.modelscope.cn/twinkle'`, skip hardware planning — cloud handles it.

### Resource Assessment

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
ray status  # if Ray running
```

### VRAM Quick Rules

- **LoRA training**: model_weights_bf16 + ~20% overhead (7B→~17GB)
- **Full FT**: model_weights × 4 (7B→~56GB)
- **vLLM sampler**: model_weights + KV cache

| Model | bf16 VRAM | LoRA (1 GPU) | Min GPU |
|-------|-----------|-------------|---------|
| Qwen3.5-4B | 8 GB | ~10 GB | 1× A10 |
| Qwen3.5-7B | 14 GB | ~17 GB | 1× A10 |
| Qwen3.5-14B | 28 GB | ~34 GB | 1× A100 |
| Qwen3.5-32B | 64 GB | ~77 GB | 1× A100 |

### GPU Split (Server Mode)

```
1 GPU  → model only, SFT/DPO
2 GPUs → 1 model + 1 sampler (GRPO)
4 GPUs → 1-2 model + 2-3 sampler
8 GPUs → 2 model + 4 sampler (or 8 dp for SFT)
Large models: 2× TP for 32B, 4× TP for 72B
```

---

## Core API Reference

### 1. Initialization

```python
from twinkle import init_twinkle_client

# Server Mode (primary — self-hosted)
client = init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_API_KEY')

# Cloud Mode (ModelScope hosted)
import os
client = init_twinkle_client(
    base_url='http://www.modelscope.cn/twinkle',
    api_key=os.environ['MODELSCOPE_TOKEN']
)

# Check available models
caps = client.get_server_capabilities()
for m in caps.supported_models:
    print(f'- {m.model_name}')
```

**Parameters:**
- `base_url`: Server URL (fallback: `TWINKLE_SERVER_URL` env var)
- `api_key`: Auth token (fallback: `TWINKLE_SERVER_TOKEN` env var)
- `session_heartbeat_interval`: Seconds between heartbeats (default: 10)

### 2. Dataset & DatasetMeta

```python
from twinkle.dataset import Dataset, DatasetMeta, LazyDataset
```

**DatasetMeta** — describes a data source:
```python
DatasetMeta(
    dataset_id='ms://modelscope/gsm8k',  # ModelScope/HF ID or local path
    subset_name='main',                    # subset (default: 'default')
    split='train',                         # split (default: 'train')
    data_slice=range(5000),                # pick first N samples (optional)
)
```

**In-memory data** (no external dataset):
```python
DatasetMeta(data=[
    {'messages': [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello!'}]},
    ...
])
```

**Dataset** — load, preprocess, encode:
```python
dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train', data_slice=range(5000)))
dataset.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B', max_length=8192)
dataset.map(GSM8KProcessor(system='Solve the math problem.'))
dataset.encode(add_generation_prompt=True)  # True=for sampling, False=for training labels
```

**LazyDataset** — defers map/encode to `__getitem__` (for multimodal / large datasets):
```python
dataset = LazyDataset(DatasetMeta('ms://AI-ModelScope/LaTeX_OCR', data_slice=range(500)))
dataset.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B', max_length=512)
dataset.map(LatexOCRProcessor)
dataset.encode(batched=True)
```

**Key Dataset methods:**
| Method | Description |
|--------|-------------|
| `set_template(name, model_id=..., max_length=...)` | Set chat template for encoding |
| `map(processor, init_args={...})` | Apply preprocessor (class, instance, or string name) |
| `encode(add_generation_prompt=False)` | Tokenize into InputFeature |
| `filter(filter_func)` | Filter rows |
| `add_dataset(DatasetMeta(...))` | Add another dataset |
| `mix_dataset(interleave=True)` | Combine added datasets |

### 3. DataLoader

```python
from twinkle.dataloader import DataLoader

dataloader = DataLoader(dataset=dataset, batch_size=8, num_workers=0, drop_last=True)
```

**Parameters:**
- `dataset`: Dataset or LazyDataset instance
- `batch_size`: Samples per batch
- `min_batch_size`: Minimum batch size (optional)
- `num_workers`: DataLoader workers (default: 2; use 0 for debugging)

**Checkpoint/Resume:**
```python
# Resume
dataloader.resume_from_checkpoint(consumed_train_samples=progress['consumed_train_samples'])

# Save state
state = dataloader.get_state()  # → {'consumed_train_samples': int}
```

### 4. MultiLoraTransformersModel

```python
from twinkle_client.model import MultiLoraTransformersModel
from peft import LoraConfig

model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
```

**Setup methods (call in order):**
```python
# 1. Add LoRA adapter
lora_config = LoraConfig(
    target_modules='all-linear',
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
)
model.add_adapter_to_model(
    'default',                          # adapter_name (unique per experiment)
    lora_config,
    gradient_accumulation_steps=2,      # effective_batch = batch_size × grad_accum
    # NOTE: Do NOT pass save_dir — the server manages checkpoint paths automatically
)

# 2. Set template (same as dataset)
model.set_template('Qwen3_5Template')

# 3. Set input processor
model.set_processor('InputProcessor', padding_side='right')

# 4. Set loss function
model.set_loss('CrossEntropyLoss')  # or 'GRPOLoss', 'DPOLoss', 'GKDLoss'

# 5. Set optimizer (only Adam supported for Megatron backend)
model.set_optimizer('Adam', lr=1e-4)

# 6. Set LR scheduler (optional, NOT supported for Megatron backend)
model.set_lr_scheduler('CosineAnnealingLR', T_max=100, eta_min=0)
# model.set_lr_scheduler('CosineWarmupScheduler', num_warmup_steps=50, num_training_steps=1000)
```

**Training loop methods:**
| Method | Description |
|--------|-------------|
| `forward_backward(inputs, **kwargs)` | Forward + backward in one call |
| `forward_only(inputs, disable_lora=False)` | Forward without grad (for ref model in DPO) |
| `clip_grad_and_step()` | Clip grad → optimizer step → zero_grad → lr_step (all-in-one) |
| `clip_grad_norm(max_grad_norm=1.0)` | Only clip gradients |
| `step()` | Only optimizer step |
| `zero_grad()` | Only zero gradients |
| `lr_step()` | Only LR scheduler step |
| `calculate_metric(is_training=True)` | Get metrics (returns `.result` dict) |
| `add_metric('DPOMetric', beta=0.1)` | Register additional metric |

**Save/Load/Upload:**
```python
# Save checkpoint (returns SaveResponse with .twinkle_path)
result = model.save(
    name='my-checkpoint',
    save_optimizer=True,
    consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    is_sampler=False,  # True = sampler-only checkpoint (deletes old sampler saves)
)

# Resume from checkpoint
progress = model.resume_from_checkpoint(result.twinkle_path)
# progress → {'cur_step': int, 'consumed_train_samples': int}
dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
start_step = progress['cur_step']

# Upload to ModelScope Hub
model.upload_to_hub(
    checkpoint_dir=result.twinkle_path,
    hub_model_id='your_username/model-name',
    hub_token=None,  # uses server default if None
)
```

### 5. vLLMSampler

```python
from twinkle_client.sampler import vLLMSampler

sampler = vLLMSampler(model_id='ms://Qwen/Qwen3.5-4B')
sampler.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B')
```

**Sampling:**
```python
sampling_params = {
    'max_tokens': 1024,
    'temperature': 1.0,
    'top_p': 0.95,
    'num_samples': 4,    # completions per prompt
    'logprobs': 1,       # return log probabilities
}

# Sync weights from training model
result = model.save(name='sampler-weights', save_optimizer=False, is_sampler=True)

# Sample
responses = sampler.sample(
    inputs=batch,                        # List[Trajectory] or List[InputFeature]
    sampling_params=sampling_params,
    adapter_uri=result.twinkle_path,     # use latest trained weights
)

# Parse responses
for response in responses:
    for seq in response.sequences:
        seq.new_input_feature  # Dict: full trajectory as InputFeature (for training)
        seq.tokens             # List[int]: generated token ids
        seq.logprobs           # List[List[Tuple[int, float]]]: [(token_id, logp), ...]
        seq.stop_reason        # str: 'stop' or 'length'
```

### 6. Preprocessors

```python
from twinkle.preprocessor import GSM8KProcessor, SelfCognitionProcessor, EmojiDPOProcessor
from twinkle.preprocessor import Preprocessor  # base class for custom
```

| Preprocessor | Usage | Init Args |
|---|---|---|
| `GSM8KProcessor` | Math QA → Trajectory | `system=None, add_assistant=False` |
| `SelfCognitionProcessor` | Self-cognition SFT | `model_name='twinkle robot', model_author='twinkle lab'` |
| `EmojiDPOProcessor` | DPO preference pairs | `system=None, chosen_key='answer_zh', rejected_key='answer_en', prompt_key='prompt'` |

**Using preprocessors:**
```python
# By instance (with args)
dataset.map(GSM8KProcessor(system='Solve step by step.'))
dataset.map(SelfCognitionProcessor(model_name='My Bot', model_author='Me'))

# By string name + init_args (for cloud mode / serialization)
dataset.map('SelfCognitionProcessor', init_args={'model_name': 'My Bot', 'model_author': 'Me'})

# By class reference
dataset.map(EmojiDPOProcessor, init_args={'system': 'You are helpful.'})
```

**Custom preprocessor:**
```python
from twinkle.preprocessor import Preprocessor
from twinkle.data_format import Trajectory, Message

class MyProcessor(Preprocessor):
    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(messages=[
            Message(role='user', content=row['question']),
            Message(role='assistant', content=row['answer']),
        ])
```

### 7. Loss Functions

```python
model.set_loss('CrossEntropyLoss')
model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
model.set_loss('DPOLoss', beta=0.1, loss_type='sigmoid', reference_free=False, sft_weight=1.0)
model.set_loss('GKDLoss', beta=0.5, temperature=1.0)
```

| Loss | Use Case | Key Params |
|------|----------|------------|
| `CrossEntropyLoss` | SFT | `ignore_index=-100, dft=False` |
| `GRPOLoss` | GRPO/PPO RL | `epsilon=0.2, beta=0.0 (KL), entropy_coef=0.0` |
| `DPOLoss` | DPO preference | `beta=0.1, loss_type='sigmoid'/'hinge'/'ipo'/'kto_pair', sft_weight=0.0` |
| `GKDLoss` | Knowledge distillation | `beta=0.5 (JSD mix), temperature=1.0, chunk_size=512` |

### 8. Rewards & Advantages

```python
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.advantage import GRPOAdvantage
```

**Built-in rewards:**
```python
reward_fn = GSM8KAccuracyReward()
rewards = reward_fn(trajectories)  # → List[float] (1.0=correct, 0.0=wrong)
```

**Custom reward (MUST subclass Reward):**
```python
class MyReward(Reward):
    def __call__(self, trajectories, **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            # Your scoring logic here
            rewards.append(score)
        return rewards
```

**Advantage computation:**
```python
advantage_fn = GRPOAdvantage()
advantages = advantage_fn(
    rewards,                    # List[float] or Tensor
    num_generations=4,          # samples per prompt
    scale='group',              # 'group'=per-prompt, 'batch'=global, 'none'=no normalization
).tolist()
```

### 9. Metrics

```python
from twinkle.metric import CompletionRewardMetric, DPOMetric
```

**CompletionRewardMetric** (for GRPO):
```python
metrics = CompletionRewardMetric()
metrics.accumulate(
    completion_lengths=all_completion_lengths,
    rewards={'total': total_rewards, 'accuracy': acc_rewards},
)
log_dict = metrics.calculate()  # → {'train/total_reward': ..., 'train/completion_length': ...}
metrics.reset()
```

**DPOMetric** (for DPO — added to model):
```python
model.add_metric('DPOMetric', beta=0.1)
# Then after forward_backward:
metric = model.calculate_metric(is_training=True)
# metric.result → {'logps/chosen': ..., 'rewards/margins': ..., 'rewards/accuracies': ...}
```

### 10. TrainingRuntime (Observability)

```python
from twinkle_client.auto.runtime import TrainingRuntime

rt = TrainingRuntime()  # auto-reads TWINKLE_RUN_ID env var (set by TUI launcher)
# IMPORTANT: call rt.start() BEFORE model.add_adapter_to_model() so TUI can show logs immediately.
# add_adapter_to_model triggers NCCL init across all GPUs which can take 60-120s.
rt.start(model_id='Qwen/Qwen3.5-4B', config={'lr': 1e-4}, script_path=__file__)
rt.register_graceful_shutdown(model, dataloader)  # MUST register

# Resume logic — MUST be after dataloader creation, before training loop:
resume = rt.get_resume_info()
global_step = resume['last_step']
if global_step > 0:
    dataloader.skip_consumed_samples(global_step * BATCH_SIZE)
    print(f'[twinkle] Resuming from step {global_step}')

# In training loop — use print() for logs (stdout goes to output.log, shown in TUI):
metric = model.calculate_metric(is_training=True)
rt.log_metrics(step=step, total_steps=MAX_STEPS, **metric.result)
print(f'[Step {step}/{MAX_STEPS}] {metric.result}')

# When done:
rt.finish(status='completed')
```

### 11. Data Types

```python
from twinkle.data_format import Trajectory, Message, InputFeature
```

**Trajectory** (conversation format — used as input to dataset/sampler):
```python
Trajectory(
    messages=[
        Message(role='system', content='You are helpful.'),
        Message(role='user', content='What is 2+2?'),
        Message(role='assistant', content='4'),
    ],
    images=[...],   # optional: for multimodal
    videos=[...],   # optional
)
```

**Message fields:** `role` ('system'/'user'/'assistant'/'tool'), `content` (str), `tool_calls`, `reasoning_content`

**InputFeature** (tokenized — output of encode):
```python
InputFeature(
    input_ids=[...],        # token ids
    attention_mask=[...],   # 0/1 mask
    labels=[...],           # -100 for ignored positions
    completion_mask=[...],  # for RL: which tokens to optimize
    length=512,
)
```

---

## Complete Training Examples

### Example 1: SFT (Self-Cognition Fine-Tuning)

```python
import os
from peft import LoraConfig
from twinkle import init_twinkle_client
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.auto.runtime import TrainingRuntime

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
MAX_STEPS = 50

# 1. Init client
client = init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_API_KEY')

# 2. Runtime (MUST be before model setup — add_adapter_to_model takes 60-120s for NCCL init)
rt = TrainingRuntime()
rt.start(model_id='Qwen/Qwen3.5-4B', config={'lr': 1e-4, 'batch_size': 4}, script_path=__file__)

# 3. Prepare dataset
dataset = Dataset(DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=512)
dataset.map(SelfCognitionProcessor(model_name='Twinkle助手', model_author='ModelScope'))
dataset.encode(batched=True)
dataloader = DataLoader(dataset=dataset, batch_size=8, num_workers=0, drop_last=True)

# 4. Configure model
model = MultiLoraTransformersModel(model_id=MODEL_ID)
model.add_adapter_to_model('default', LoraConfig(target_modules='all-linear'), gradient_accumulation_steps=2)
model.set_template('Qwen3_5Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')
model.set_optimizer('Adam', lr=1e-4)
rt.register_graceful_shutdown(model, dataloader)

# 5. Resume logic (enables seamless restart after auto-fix)
resume = rt.get_resume_info()
global_step = resume['last_step']
if global_step > 0:
    dataloader.skip_consumed_samples(global_step * 8)
    print(f'[twinkle] Resuming from step {global_step}')

# 6. Training loop
for epoch in range(3):
    for batch in dataloader:
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        global_step += 1

        if global_step % 2 == 0:
            metric = model.calculate_metric(is_training=True)
            rt.log_metrics(step=global_step, total_steps=MAX_STEPS, **metric.result)
            print(f'[Step {global_step}/{MAX_STEPS}] {metric.result}')

        if global_step >= MAX_STEPS:
            break

    if global_step >= MAX_STEPS:
        break

    # Save per epoch
    result = model.save(
        name=f'sft-epoch-{epoch}',
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )
    print(f'Saved checkpoint: {result.twinkle_path}')

rt.finish(status='completed')
```

### Example 2: GRPO (Reinforcement Learning)

```python
import gc
from typing import List, Dict, Any
from peft import LoraConfig
from twinkle import init_twinkle_client
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle.preprocessor import GSM8KProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.advantage import GRPOAdvantage
from twinkle.metric import CompletionRewardMetric
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler
from twinkle_client.auto.runtime import TrainingRuntime

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
NUM_GENERATIONS = 4
MAX_STEPS = 100
BATCH_SIZE = 8   # MUST be >= sampler_dp (sampler workers) AND >= model_dp
LEARNING_RATE = 2e-5

# 1. Init client
client = init_twinkle_client(base_url='http://127.0.0.1:8000', api_key='EMPTY_API_KEY')

# 2. Runtime (before model setup)
rt = TrainingRuntime()
rt.start(model_id='Qwen/Qwen3.5-4B', config={'lr': LEARNING_RATE, 'method': 'GRPO'}, script_path=__file__)

# 3. Prepare dataset (encode with generation prompt for sampling)
dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train', data_slice=range(2000)))
dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=2048, enable_thinking=False)
dataset.map(GSM8KProcessor(system='Solve the math problem and put answer in \\boxed{}.'))
dataset.encode(add_generation_prompt=True)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=True)

# 4. Configure model with GRPOLoss
model = MultiLoraTransformersModel(model_id=MODEL_ID)
model.add_adapter_to_model('default', LoraConfig(target_modules='all-linear', r=8, lora_alpha=32, lora_dropout=0.05),
                           gradient_accumulation_steps=1)
model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
model.set_optimizer('Adam', lr=LEARNING_RATE)
model.set_processor('InputProcessor')
model.set_template('Qwen3_5Template', model_id=MODEL_ID)
rt.register_graceful_shutdown(model, dataloader)

# 5. Configure sampler
sampler = vLLMSampler(model_id=MODEL_ID)
sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)

# 6. Setup
advantage_fn = GRPOAdvantage()
reward_fn = GSM8KAccuracyReward()
metrics = CompletionRewardMetric()
sampling_params = {'max_tokens': 1024, 'temperature': 1.0, 'top_p': 0.95, 'num_samples': NUM_GENERATIONS, 'logprobs': 1}
current_adapter_uri = None

# 7. Training loop
step = 0
for batch in dataloader:
    if step >= MAX_STEPS:
        break
    metrics.reset()

    # 7a. Sync weights to sampler
    result = model.save(name='grpo-sampler-weights', save_optimizer=False, is_sampler=True)
    current_adapter_uri = result.twinkle_path

    # 7b. Sample completions
    responses = sampler.sample(inputs=batch, sampling_params=sampling_params, adapter_uri=current_adapter_uri)

    all_inputs: List[Dict[str, Any]] = []
    all_old_logps: List[List[float]] = []
    all_completion_lengths: List[int] = []

    for response in responses:
        for seq in response.sequences:
            all_inputs.append(seq.new_input_feature)
            all_old_logps.append([lp[0][1] for lp in seq.logprobs])
            all_completion_lengths.append(len(seq.tokens))

    # 7c. Compute rewards
    rewards = reward_fn(all_inputs)
    metrics.accumulate(completion_lengths=all_completion_lengths, rewards={'accuracy': rewards})

    # 7d. Compute advantages
    advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

    # Skip if all advantages are zero (no learning signal)
    if all(abs(a) < 1e-8 for a in advantages):
        step += 1
        continue

    # 7e. Train
    model.forward_backward(inputs=all_inputs, advantages=advantages, old_logps=all_old_logps)
    model.clip_grad_and_step()
    gc.collect()

    # 7f. Log
    log_dict = metrics.calculate()
    log_dict.update(model.calculate_metric(is_training=True).result)
    rt.log_metrics(step=step, total_steps=MAX_STEPS, **log_dict)
    print(f'[Step {step}/{MAX_STEPS}] {log_dict}')
    step += 1

# Save final
model.save(name='grpo-final', save_optimizer=True)
rt.finish(status='completed')
```

### Example 3: DPO (Preference Optimization)

```python
import numpy as np
import torch
from typing import Any, Dict, List
from peft import LoraConfig
from twinkle import init_twinkle_client
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle.preprocessor import EmojiDPOProcessor
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.auto.runtime import TrainingRuntime

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DPO_BETA = 0.1
LEARNING_RATE = 1e-4

# 1. Init
client = init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_API_KEY')

# 2. Runtime (before model setup)
rt = TrainingRuntime()
rt.start(model_id='Qwen/Qwen3.5-4B', config={'method': 'DPO', 'beta': DPO_BETA}, script_path=__file__)

# 3. Prepare DPO dataset
dataset = Dataset(DatasetMeta('ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji', data_slice=range(100)))
dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=2048)
dataset.map(EmojiDPOProcessor, init_args={'system': 'You are a helpful assistant.'})
dataset.encode()  # DPO: no add_generation_prompt
dataloader = DataLoader(dataset=dataset, batch_size=8, num_workers=0, drop_last=True)

# 4. Configure model with DPO loss
model = MultiLoraTransformersModel(model_id=MODEL_ID)
model.add_adapter_to_model('default', LoraConfig(target_modules='all-linear', r=8, lora_alpha=32, lora_dropout=0.05),
                           gradient_accumulation_steps=2)
model.set_template('Qwen3_5Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('DPOLoss', beta=DPO_BETA, loss_type='sigmoid', reference_free=False, sft_weight=1.0)
model.add_metric('DPOMetric', beta=DPO_BETA)
model.set_optimizer('Adam', lr=LEARNING_RATE)
rt.register_graceful_shutdown(model, dataloader)


def prepare_dpo_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Interleave positive/negative for DP-safe training: [pos1, neg1, pos2, neg2, ...]"""
    result = []
    for row in batch:
        base_fields = {k: v for k, v in row.items() if k not in ('positive', 'negative')}
        result.append({**base_fields, **row['positive']})
        result.append({**base_fields, **row['negative']})
    return result


# 5. Training loop
max_steps = len(dataloader)
for step, batch in enumerate(dataloader):
    # Convert numpy/torch tensors for serialization
    for row in batch:
        for key in row:
            if isinstance(row[key], np.ndarray):
                row[key] = row[key].tolist()
            elif isinstance(row[key], torch.Tensor):
                row[key] = row[key].cpu().numpy().tolist()

    dpo_batch = prepare_dpo_batch(batch)

    # Get reference logps from base model (disable LoRA)
    ref_outputs = model.forward_only(inputs=dpo_batch, disable_lora=True)

    # Train with DPO loss
    model.forward_backward(inputs=dpo_batch, ref_outputs=ref_outputs.result)
    model.clip_grad_and_step()

    if step % 2 == 0:
        metric = model.calculate_metric(is_training=True)
        rt.log_metrics(step=step, total_steps=max_steps, **metric.result)
        print(f'[Step {step}/{max_steps}] {metric.result}')

result = model.save(name='dpo-final', save_optimizer=True)
rt.finish(status='completed')
```

### Example 4: Multimodal SFT (Image Understanding)

```python
import numpy as np
import torch
from peft import LoraConfig
from twinkle import init_twinkle_client
from twinkle.dataset import LazyDataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle.preprocessor import Preprocessor
from twinkle.data_format import Trajectory, Message
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.auto.runtime import TrainingRuntime

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'


class LatexOCRProcessor(Preprocessor):
    """Custom preprocessor for LaTeX OCR dataset."""
    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(messages=[
            Message(role='user', content='<image>Using LaTeX to perform OCR on the image.', images=[row['image']]),
            Message(role='assistant', content=row['text']),
        ])


# 1. Init
client = init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_API_KEY')

# 2. Runtime (before model setup)
rt = TrainingRuntime()
rt.start(model_id='Qwen/Qwen3.5-4B', config={'task': 'multimodal-sft'}, script_path=__file__)

# 3. LazyDataset for multimodal (defers processing to avoid OOM)
dataset = LazyDataset(DatasetMeta('ms://AI-ModelScope/LaTeX_OCR', data_slice=range(500)))
dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=512)
dataset.map(LatexOCRProcessor)
dataset.encode(batched=True)
dataloader = DataLoader(dataset=dataset, batch_size=8, num_workers=0, drop_last=True)

# 4. Model setup
model = MultiLoraTransformersModel(model_id=MODEL_ID)
model.add_adapter_to_model('default', LoraConfig(target_modules='all-linear'), gradient_accumulation_steps=2)
model.set_template('Qwen3_5Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')
model.set_optimizer('Adam', lr=1e-4)
rt.register_graceful_shutdown(model, dataloader)

# 5. Train
for epoch in range(3):
    for step, batch in enumerate(dataloader):
        # Important: convert numpy/torch for serialization
        for sample in batch:
            for key in sample:
                if isinstance(sample[key], np.ndarray):
                    sample[key] = sample[key].tolist()
                elif isinstance(sample[key], torch.Tensor):
                    sample[key] = sample[key].cpu().numpy().tolist()

        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()

        if step % 2 == 0:
            metric = model.calculate_metric(is_training=True)
            rt.log_metrics(step=step, total_steps=len(dataloader), **metric.result)

    model.save(name=f'multimodal-epoch-{epoch}', save_optimizer=True)

rt.finish(status='completed')
```

### Example 5: Sampling / Inference Only

```python
from twinkle_client import init_twinkle_client
from twinkle_client.sampler import vLLMSampler

# 1. Init
client = init_twinkle_client(base_url='http://127.0.0.1:8000', api_key='EMPTY_API_KEY')

# 2. Create sampler
sampler = vLLMSampler(model_id='Qwen/Qwen3.5-4B')
sampler.set_template('Qwen3_5Template', model_id='Qwen/Qwen3.5-4B')

# 3. Prepare input as Trajectory
trajectory = {
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who are you?'},
    ]
}

# 4. Sample (with optional LoRA adapter)
responses = sampler.sample(
    inputs=[trajectory] * 4,  # 4 prompts
    sampling_params={'max_tokens': 128, 'temperature': 1.0, 'num_samples': 2},
    adapter_uri='twinkle://...',  # optional: from a model.save() result
)

# 5. Decode
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=True)
for response in responses:
    for seq in response.sequences:
        text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        print(text)
```

---

## Server Mode Architecture

```
┌─ Twinkle Server (Ray + GPU) ─────────────────────────────┐
│  Base Model → adapter 'exp-01' (weights + optimizer)      │
│            → adapter 'exp-02' (weights + optimizer)      │
│  vLLM Sampler → shared inference engine                   │
└───────────────────────────────────────────────────────────┘
         ↑ HTTP (forward_backward, clip_grad_and_step, save, sample)
┌─ Client Script (CPU only, stateless) ────────────────────┐
│  Data loading + Training loop + Reward computation        │
└───────────────────────────────────────────────────────────┘
```

**Key implications:**
- "Pause" = kill client (SIGKILL) → server retains all state
- "Stop" = SIGTERM → saves checkpoint + dataloader state → exits
- "Resume" = restart with same adapter_name → continues seamlessly
- "Reset" = use new adapter_name → fresh start

### Starting Local Server

```bash
# 1. Start Ray
CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --port=6379 --num-gpus=4 --disable-usage-stats
CUDA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0  # CPU worker

# 2. Start server
python server.py  # reads server_config.yaml, blocks
```

The TUI agent's `start_server` tool handles this automatically — generates config + starts Ray + launches server.

---

## Built-in Components Summary

| Type | Available | Import Path |
|------|-----------|-------------|
| **Loss** | `CrossEntropyLoss`, `GRPOLoss`, `DPOLoss`, `GKDLoss` | `twinkle.loss` |
| **Preprocessor** | `GSM8KProcessor`, `SelfCognitionProcessor`, `EmojiDPOProcessor` | `twinkle.preprocessor` |
| **Reward** | `GSM8KAccuracyReward`, `GSM8KFormatReward` | `twinkle.reward` |
| **Advantage** | `GRPOAdvantage` | `twinkle.advantage` |
| **Metric** | `CompletionRewardMetric`, `DPOMetric` | `twinkle.metric` |
| **Template** | `Qwen3_5Template` | (string name to `set_template`) |
| **Processor** | `InputProcessor` | (string name to `set_processor`) |

**Cloud mode restriction:** Only built-in components (by name string). Custom classes cannot be serialized.

---

## Tinker-Compatible API (Alternative)

For GRPO with Tinker API:
```python
from twinkle import init_tinker_client
init_tinker_client()
from tinker import ServiceClient, types

service_client = ServiceClient(base_url=BASE_URL, api_key=API_KEY)
training_client = service_client.create_lora_training_client(base_model='Qwen/Qwen3.5-4B', rank=16)
training_client.forward_backward(datums, 'importance_sampling').result()
training_client.optim_step(types.AdamParams(learning_rate=2e-5)).result()
sampling_client = training_client.save_weights_and_get_sampling_client(name='step-N')
```

---

## File Layout

```
~/.cache/twinkle/{run_id}/
├── meta.json       # Run metadata (model_id, config, status, pid, script_version)
├── train.py        # Current active script
├── train_v1.py     # Archived versions
├── metrics.jsonl   # One JSON line per step
├── logs.jsonl      # One JSON line per event
└── stderr.log      # Script stderr output
```

---

## OpenAI-Compatible Endpoint

The server also exposes OpenAI-compatible `/v1/chat/completions`:

```python
from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:8000/api/v1', api_key='EMPTY_API_KEY')
resp = client.chat.completions.create(
    model='Qwen/Qwen3.5-4B',
    messages=[{'role': 'user', 'content': 'Hello!'}],
    max_tokens=128,
    temperature=0.7,
    stream=True,  # streaming supported
)
for chunk in resp:
    print(chunk.choices[0].delta.content, end='')
```
