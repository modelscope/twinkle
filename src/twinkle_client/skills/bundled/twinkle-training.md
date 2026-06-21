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

## Core API Reference

### Initialization

```python
# Server Mode (primary)
from twinkle import init_twinkle_client
client = init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_API_KEY')

# Cloud Mode
client = init_twinkle_client(base_url='http://www.modelscope.cn/twinkle', api_key=os.environ['MODELSCOPE_TOKEN'])
```

### Dataset

```python
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader

dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train', data_slice=range(5000)))
dataset.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B', max_length=8192)
dataset.map(GSM8KProcessor())                      # preprocessor: raw → Trajectory
dataset.encode(add_generation_prompt=True)          # True=sampling input, False=training labels
dataloader = DataLoader(dataset=dataset, batch_size=4)
```

### Model (Server Mode)

```python
from twinkle_client.model import MultiLoraTransformersModel
from peft import LoraConfig

model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
model.add_adapter_to_model('my-exp', LoraConfig(target_modules='all-linear', r=8, lora_alpha=32),
                           gradient_accumulation_steps=2)
model.set_template('Qwen3_5Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')       # or 'GRPOLoss', 'DPOLoss', 'GKDLoss'
model.set_optimizer('Adam', lr=1e-4)
model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
```

### Sampler

```python
from twinkle_client.sampler import vLLMSampler

sampler = vLLMSampler(model_id='ms://Qwen/Qwen3.5-4B')
sampler.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B')

# Weight sync + sample
result = model.save(name='sampler-weights', save_optimizer=False, is_sampler=True)
responses = sampler.sample(inputs=batch, sampling_params={...}, adapter_uri=result.twinkle_path)
```

### TrainingRuntime (Observability)

```python
from twinkle_client.tui.runtime import TrainingRuntime

rt = TrainingRuntime(run_id='my-experiment')
rt.start(model_id='Qwen/Qwen3.5-4B', config={'lr': 1e-4}, script_path=__file__)
rt.register_graceful_shutdown(model, dataloader)  # MUST register

# In loop:
rt.log_metrics(step=step, total_steps=MAX_STEPS, loss=loss, reward=reward, grad_norm=gn, lr=lr)
rt.log(f'[Step {step}/{MAX_STEPS}] loss={loss:.4f} reward={reward:.3f}')

# Done:
rt.finish(status='completed')
```

## Training Patterns

### SFT

```python
model.set_loss('CrossEntropyLoss')
dataset.encode(add_generation_prompt=False)  # include assistant in labels

for step, batch in enumerate(dataloader):
    model.forward_backward(inputs=batch)
    model.clip_grad_and_step()
    metric = model.calculate_metric(is_training=True)
    rt.log_metrics(step=step, total_steps=len(dataloader), **metric.result)

model.save(name='sft-final', save_optimizer=True)
```

### GRPO

```python
from twinkle.reward import GSM8KAccuracyReward
from twinkle.advantage import GRPOAdvantage

model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
dataset.encode(add_generation_prompt=True)
advantage_fn = GRPOAdvantage()
reward_fn = GSM8KAccuracyReward()

for step, batch in enumerate(dataloader):
    if step >= MAX_STEPS: break

    # 1. Sync weights → sample
    result = model.save(name='sampler-weights', save_optimizer=False, is_sampler=True)
    responses = sampler.sample(inputs=batch,
        sampling_params={'max_tokens': 1024, 'temperature': 1.0, 'num_samples': NUM_GENERATIONS, 'logprobs': 1},
        adapter_uri=result.twinkle_path)

    # 2. Collect inputs + logprobs
    all_inputs, all_old_logps = [], []
    for resp in responses:
        for seq in resp.sequences:
            all_inputs.append(seq.new_input_feature)
            all_old_logps.append([lp[0][1] for lp in seq.logprobs])

    # 3. Reward → advantage → train
    rewards = reward_fn(all_inputs)
    advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
    model.forward_backward(inputs=all_inputs, advantages=advantages, old_logps=all_old_logps)
    model.clip_grad_and_step()
```

### DPO

```python
model.set_loss('DPOLoss', beta=0.1, loss_type='sigmoid', reference_free=False, sft_weight=1.0)
model.add_metric('DPOMetric', beta=0.1)
# Data: chosen + rejected concatenated (first half chosen, second half rejected)
# Preprocessor: EmojiDPOProcessor or custom → Trajectory with extend_message=[('rejected_messages', ...)]

for batch in dataloader:
    dpo_batch = prepare_dpo_batch(batch)  # expand positive/negative
    ref_outputs = model.forward_only(inputs=dpo_batch, disable_lora=True)
    model.forward_backward(inputs=dpo_batch, ref_outputs=ref_outputs.result)
    model.clip_grad_and_step()
```

## Server Mode Architecture

```
┌─ Twinkle Server (Ray + GPU) ─────────────────────────┐
│  Base Model → adapter 'exp-01' (weights + optimizer)  │
│            → adapter 'exp-02' (weights + optimizer)  │
└───────────────────────────────────────────────────────┘
         ↑ HTTP (forward_backward, clip_grad_and_step, save)
┌─ Client Script (CPU only, stateless) ────────────────┐
│  Data loading + Training loop + Reward computation    │
└───────────────────────────────────────────────────────┘
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

## Data Format

- `Trajectory`: `{'messages': [Message(role, content)], 'extend_message': [('key', value)]}`
- `Message`: TypedDict with `role` and `content` (access via `msg['role']`)
- DPO: `Trajectory(messages=chosen, extend_message=[('rejected_messages', rejected)])`
- SamplingParams: `{'max_tokens': 4096, 'temperature': 1.0, 'top_p': 1.0, 'num_samples': 1, 'logprobs': 1}`

## Built-in Components

| Type | Available |
|------|-----------|
| Loss | `CrossEntropyLoss`, `GRPOLoss`, `DPOLoss`, `GKDLoss` |
| Preprocessor | `GSM8KProcessor`, `SelfCognitionProcessor`, `EmojiDPOProcessor` |
| Reward | `GSM8KAccuracyReward` |
| Advantage | `GRPOAdvantage` |
| Template | `Qwen3_5Template` |

**Cloud mode restriction:** Only built-in components (by name string). Custom classes cannot be serialized.

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

## File Layout

```
~/.cache/twinkle/{run_id}/
├── meta.json       # Run metadata (model_id, config, status, pid, script_version)
├── train.py        # Current active script
├── train_v1.py     # Archived versions
├── metrics.jsonl   # One JSON line per step
└── logs.jsonl      # One JSON line per event
```

## Ray Mode (Direct, No Server)

For multi-node distributed training without server:

```python
import twinkle
from twinkle import DeviceMesh, DeviceGroup
from twinkle.model import TransformersModel

device_groups = [
    DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
    DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
]
twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)

model = TransformersModel(model_id='ms://Qwen/Qwen3.5-4B', device_mesh=model_mesh, remote_group='model')
# Megatron: MegatronModel(model_id=..., device_mesh=mesh_with_tp, mixed_precision='bf16')
```

| | Ray Mode | Server Mode |
|---|---|---|
| Import | `twinkle.model.TransformersModel` | `twinkle_client.model.MultiLoraTransformersModel` |
| Prerequisites | `twinkle.initialize(mode='ray')` | `init_twinkle_client(base_url=...)` |
| Kill client | Loses state | Zero-cost (server keeps all) |
