# Twinkle Training Script Skill

You are an expert at writing training scripts for the Twinkle framework. Follow these conventions when generating training code.

## Pre-Training Planning (MUST DO FIRST)

Before writing any training script or server config, you MUST evaluate the following. This is the first step of any training task.

> **Cloud Service shortcut:** If using ModelScope cloud (`base_url='http://www.modelscope.cn/twinkle'`), skip sections 1-4 (Cluster Assessment, VRAM Estimation, DeviceMesh Design, Training Time) — the cloud handles all hardware. Jump directly to section 5 (Dataset) and 6 (Model Selection). See [Cloud Service Mode](#cloud-service-mode-modelscope-hosted).

### 1. Cluster Resource Assessment

Determine available hardware:

```bash
# Check GPU count and model
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# If Ray is already running
ray status  # shows total GPUs, CPUs, memory
```

Key info to gather:
- Number of GPUs available
- GPU model (A100 80GB, A10 24GB, H100 80GB, etc.)
- Total VRAM per GPU
- CPU count and RAM (for data preprocessing)

### 2. Model Size & VRAM Estimation

Estimate GPU memory requirements BEFORE choosing device mesh:

| Component | Formula | Example (7B model, bf16) |
|-----------|---------|-------------------------|
| Model weights | params × 2 bytes (bf16) | 7B × 2 = 14 GB |
| LoRA adapter | rank × hidden_dim × 2 × num_layers × 2 bytes | ~0.1-0.5 GB |
| Optimizer (Adam, LoRA only) | lora_params × 12 bytes (fp32 param + momentum + variance) | ~0.3-1.5 GB |
| Activations (per batch) | ~batch_size × seq_len × hidden_dim × num_layers × 2 bytes | varies greatly |
| Gradient | same as trainable params × 2 bytes | ~0.1-0.5 GB (LoRA) |
| KV Cache (sampler/vLLM) | 2 × num_layers × num_heads × head_dim × max_seq_len × 2 bytes × batch | large |

**Quick rules of thumb:**
- **LoRA training**: model weights + ~20% overhead → 14GB model needs ~17GB
- **Full fine-tuning**: model weights × 4 (weights + gradients + optimizer states) → 14GB model needs ~56GB
- **vLLM sampler**: model weights + KV cache (gpu_memory_utilization controls this)
- **Safe margin**: always leave 10-15% free VRAM for fragmentation

**Common model sizes:**

| Model | Params | bf16 VRAM (weights only) | LoRA training (1 GPU) | Min GPU |
|-------|--------|-------------------------|----------------------|--------|
| Qwen3.5-1.5B | 1.5B | 3 GB | ~4 GB | 1× A10 24GB |
| Qwen3.5-4B | 4B | 8 GB | ~10 GB | 1× A10 24GB |
| Qwen3.5-7B | 7B | 14 GB | ~17 GB | 1× A10 24GB |
| Qwen3.5-14B | 14B | 28 GB | ~34 GB | 1× A100 80GB |
| Qwen3.5-32B | 32B | 64 GB | ~77 GB | 1× A100 80GB (tight) |
| Qwen3.5-72B | 72B | 144 GB | needs TP/PP | 2-4× A100 80GB |

### 3. DeviceMesh & Server Config Design

Based on available GPUs and VRAM requirements, design the allocation:

**Server Mode (twinkle_client) — decision tree:**

```
Total GPUs available?
├─ 1 GPU: model only (no sampler), SFT/DPO only
│    server_config: ranks=1, dp_size=1
├─ 2 GPUs: 1 model + 1 sampler (for GRPO/RL)
│    or: 2 model (dp_size=2, no sampler)
├─ 4 GPUs: 1-2 model + 2-3 sampler (GRPO sweet spot)
│    or: 4 model dp (high throughput SFT)
├─ 8 GPUs: flexible split
│    GRPO: 2 model + 4 sampler (2× TP for large models)
│    SFT: 8 model dp
└─ Large models (>40B): need TP
     2× TP for 32B, 4× TP for 72B
```

**server_config.yaml key fields to adjust:**
```yaml
applications:
  - name: models-XXX
    args:
      nproc_per_node: 1          # GPUs for model (= tp_size × pp_size × dp_size)
      device_group:
        ranks: 1                 # Total GPU count for this service
      device_mesh:
        dp_size: 1               # Data parallel degree
        tp_size: 1               # Tensor parallel (for large models)
        pp_size: 1               # Pipeline parallel (rarely needed)

  - name: sampler-XXX
    args:
      nproc_per_node: 2          # GPUs for sampler
      device_group:
        ranks: 2
      device_mesh:
        dp_size: 1
        tp_size: 2               # TP for faster inference
```

### 4. Training Time Estimation

Calculate before starting:

```
total_steps = (dataset_size / batch_size / gradient_accumulation_steps / dp_size) × num_epochs

# For GRPO (no epochs, step-based):
total_steps = MAX_STEPS (user-defined, typically 100-1000)

# Time per step (rough estimates for LoRA):
#   SFT: ~0.5-2s per step (depends on seq_len and model size)
#   GRPO: ~5-30s per step (sampling dominates)
#   DPO: ~1-4s per step (2× forward per batch)

estimated_time = total_steps × time_per_step
```

**Factors that increase step time:**
- Longer sequences (quadratic attention)
- Larger batch_size (more VRAM pressure, may need gradient accumulation)
- Sampling with vLLM (GRPO bottleneck)
- num_generations in GRPO (linear with sample count)

### 5. Dataset Search & Evaluation

Use ModelScope Hub to find datasets:

```python
from modelscope.hub.api import HubApi
api = HubApi()

# Search by keyword
results = api.list_datasets(query='math reasoning', limit=10)
for ds in results:
    print(f'{ds.id}: {ds.name}')

# Check dataset size and format
from datasets import load_dataset
ds = load_dataset('modelscope/gsm8k', split='train')
print(f'Samples: {len(ds)}, Columns: {ds.column_names}')
print(ds[0])  # inspect format
```

**Dataset selection criteria:**
- Task alignment (math, code, chat, safety, etc.)
- Size vs. quality tradeoff (smaller high-quality > larger noisy)
- Format compatibility (does a built-in Preprocessor exist?)
- License considerations

**Common datasets by task:**

| Task | Datasets | Typical size |
|------|----------|-------------|
| Math reasoning | `modelscope/gsm8k`, `competition_math` | 1K-8K |
| Code | `humaneval`, `mbpp`, code-contests | 0.5K-10K |
| General chat | `sharegpt`, `ultrachat` | 50K-500K |
| Self-cognition | `swift/self-cognition` | ~500 |
| DPO preference | `shareAI-Llama3-DPO-zh-en-emoji` | 10K-100K |
| Chinese | `alpaca-zh`, `belle` | 50K-500K |

### 6. Model Selection

Choose model based on task + resources:

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Quick experiment / limited GPU | Qwen3.5-1.5B or 4B | Fast iteration, fits single GPU |
| Production quality SFT | Qwen3.5-7B or 14B | Good quality-cost balance |
| GRPO/RL research | Qwen3.5-4B or 7B | Need GPU budget for sampler |
| Maximum capability | Qwen3.5-32B+ | Needs multi-GPU TP |
| Code tasks | Qwen3.5-Coder variants | Specialized tokenizer |

**Model ID format:** `ms://Qwen/Qwen3.5-4B` (ModelScope prefix)

### 7. Planning Checklist

Before writing any code, answer these questions:

- [ ] How many GPUs are available? What type?
- [ ] Which model? Does it fit in VRAM with LoRA?
- [ ] Training method? (SFT / GRPO / DPO)
- [ ] If GRPO: how to split GPUs between model and sampler?
- [ ] Dataset identified? Format compatible?
- [ ] Estimated total_steps and wall-clock time?
- [ ] server_config.yaml: ranks, dp_size, tp_size decided?
- [ ] Hyperparameters: lr, batch_size, gradient_accumulation, max_length?

Only proceed to write the training script after all items are resolved.

## Ray Cluster Configuration

### DeviceGroup & DeviceMesh

Training typically splits GPUs into groups (model vs sampler). Define them explicitly:

```python
from twinkle import DeviceMesh, DeviceGroup

device_groups = [
    DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
    DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
]
model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
```

DeviceMesh supports: `dp_size`, `tp_size`, `pp_size`, `fsdp_size`, `cp_size`, `ep_size`, `sp_size`.

### Initialize

```python
import twinkle
twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)
```

- `mode='local'`: single node with torchrun. `mode='ray'`: multi-node Ray cluster.
- `lazy_collect=True`: results collected lazily (better throughput). `False`: synchronous collect.
- `groups` is required for Ray mode.

### Result Collection

Remote functions support `collect='first'` (return from rank-0 only) or `collect='all'` (gather from all ranks).

## Model Backend Initialization

### Transformers Backend

```python
from twinkle.model import TransformersModel
model = TransformersModel(
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=model_mesh,
    remote_group='model',
)
```

- Supports FSDP2 natively via DeviceMesh with `fsdp_size`.
- `model_cls` optional: pass HuggingFace model class for custom architectures.

### Megatron Backend

```python
from twinkle.model import MegatronModel
model = MegatronModel(
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=model_mesh,
    remote_group='model',
    mixed_precision='bf16',
)
```

- Requires DeviceMesh with `tp_size` / `pp_size` configured.
- Optimizer: `model.set_optimizer('default', lr=1e-5)` uses Megatron's built-in optimizer.
- LR scheduler: `model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=1e-5)`.

### Common Model Setup

```python
from peft import LoraConfig

lora_config = LoraConfig(
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    r=32, lora_alpha=64, lora_dropout=0.05,
)
model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
model.set_optimizer('AdamW', lr=1e-5)
model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
model.set_loss('GRPOLoss', epsilon=0.2)
model.set_processor(InputProcessor)
model.set_template('Qwen3_5Template', model_id=MODEL_ID)
```

## Dataset

### Loading

```python
from twinkle.dataset import Dataset, DatasetMeta

dataset = Dataset(DatasetMeta(
    dataset_id='ms://modelscope/gsm8k',  # ms:// prefix for ModelScope hub
    subset_name='main',
    split='train',
    data_slice=range(5000),  # optional: limit data
))
```

- `dataset_id`: local path, HuggingFace ID, or `ms://` prefixed ModelScope ID.
- `data` parameter: pass in-memory `List[Dict]` or generator callable.
- `streaming=True` in kwargs for streaming mode.

### Template & Encode

```python
dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=8192)
dataset.map(GSM8KProcessor())      # preprocessor transforms raw data -> Trajectory
dataset.encode(add_generation_prompt=True)  # encode to input_ids/labels
```

- `encode()` internally calls `template.batch_encode()` which runs the full pre_pipeline.
- `add_generation_prompt=True`: appends generation prompt (for sampling inputs).
- `add_generation_prompt=False`: includes assistant content in labels (for training).

## Preprocessor

### Built-in Preprocessors

- `GSM8KProcessor`: Extracts question/answer from GSM8K format
- `SelfCognitionProcessor(model_name, model_author)`: Generates self-cognition QA
- `EmojiDPOProcessor`: DPO pairs for emoji preference
- `CompetitionMathProcessor`, `CountdownProcessor`, etc.

### Custom Preprocessor

```python
from twinkle.preprocessor import Preprocessor
from twinkle.data_format import Trajectory, Message

class MyProcessor(Preprocessor):
    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        results = []
        for row in rows:
            messages = [
                Message(role='user', content=row['question']),
                Message(role='assistant', content=row['answer']),
            ]
            results.append(Trajectory(messages=messages))
        return self.map_row_to_col(results)
```

### Data Format

- `Trajectory`: `{'messages': [Message(role, content)], 'extend_message': [('key', value)]}`
- `Message`: `TypedDict` with `role` and `content` keys (access via bracket notation: `msg['role']`)
- DPO format: `Trajectory(messages=chosen, extend_message=[('rejected_messages', rejected)])`

## MultiTurnRollout

```python
from twinkle_agentic.envs import OpenEnv, EnvTool
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager

# Create environment and tools
env = OpenEnv(base_url='http://localhost:8000', tool_schema=TOOL_SCHEMA)
env_tools = EnvTool.from_env(env)
tool_manager = ToolManager(env_tools)

# Run multi-turn rollout
rollout = MultiTurnRollout(
    sampler=sampler,
    template=template,
    tool_managers=[tool_manager] * batch_size,
    max_turns=6,
)
trajectories = rollout.run(initial_trajectories, sampling_params)
```

- Each trajectory starts with system + initial observation.
- MultiTurnRollout drives: model generates tool calls -> env.step() -> observations fed back.
- Episode reward extracted from env after rollout completes.

## Sampler

### vLLMSampler

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams

sampler = vLLMSampler(
    model_id=MODEL_ID,
    engine_args={
        'gpu_memory_utilization': 0.8,
        'max_model_len': 4096,
        'enable_lora': True,
        'max_lora_rank': 32,
    },
    device_mesh=sampler_mesh,
    remote_group='sampler',
)
sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)
```

### SamplingParams

```python
sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    num_samples=1,
    logprobs=1,    # needed for GRPO old_logps
)
```

### Input/Output

- Input: `List[Trajectory]` or `List[InputFeature]`
- Output: `List[SampleResponse]`, each containing `.sequences[].tokens`, `.sequences[].logprobs`, `.sequences[].new_input_feature`

### Weight Sync

```python
from twinkle.checkpoint_engine import CheckpointEngineManager

ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
# In training loop:
ckpt_manager.sync_weights(merge_and_sync=True)   # merge LoRA -> full model -> sync to vLLM
ckpt_manager.sync_weights(merge_and_sync=False)  # sync LoRA weights only (requires enable_lora=True)
```

## Server Mode (twinkle_client)

Server Mode is the **primary mode for TUI-managed training**. The client is stateless; all model/optimizer state lives on the server.

### Client-side API

```python
from twinkle import init_twinkle_client
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler

client = init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_API_KEY')
model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
sampler = vLLMSampler(model_id='ms://Qwen/Qwen3.5-4B')

# Weight sync via adapter_uri (for sampler to use latest LoRA weights)
result = model.save('sampler-weights', save_optimizer=False, is_sampler=True)
responses = sampler.sample(inputs, adapter_uri=result.twinkle_path)
```

### Key differences from Ray mode

| | Ray Mode | Server Mode (twinkle_client) |
|---|---|---|
| Import | `from twinkle.model import TransformersModel` | `from twinkle_client.model import MultiLoraTransformersModel` |
| Model init | `TransformersModel(model_id=..., device_mesh=..., remote_group=...)` | `MultiLoraTransformersModel(model_id='ms://...')` |
| Prerequisites | `twinkle.initialize(mode='ray', ...)` | `init_twinkle_client(base_url=...)` + running server |
| State location | In Ray workers (distributed) | In server GPU memory (centralized) |
| Kill client | Loses state (need checkpoint resume) | Zero-cost (server keeps everything) |

## Cloud Service Mode (ModelScope Hosted)

When using ModelScope's online training service, you do NOT need to:
- Start a local Ray cluster
- Evaluate local hardware (GPUs, VRAM)
- Write server_config.yaml
- Run `server.py`

All infrastructure is managed by ModelScope. You only need to decide:
1. **Training method** (SFT / DPO / GRPO)
2. **Base model** (from ModelScope Hub)
3. **Dataset** (from ModelScope Hub or custom)

### Prerequisites

- `MODELSCOPE_TOKEN` environment variable set (get from https://modelscope.cn/my/myaccesstoken)
- `pip install twinkle-kit` (includes `twinkle` and `twinkle_client`)
- Optional: `pip install tinker` (for Tinker-compatible API)

### Discovering Available Models

Before choosing a model, query the server for its supported model list:

```python
import os
from twinkle_client import init_twinkle_client

# For cloud service
client = init_twinkle_client(
    base_url='http://www.modelscope.cn/twinkle',
    api_key=os.environ.get('MODELSCOPE_TOKEN')
)

# For local server
# client = init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_API_KEY')

caps = client.get_server_capabilities()
print('Available models:')
for model in caps.supported_models:
    print(f'  - {model.model_name}')
```

The TUI agent has a built-in `list_supported_models` tool that does this automatically:
- Default (local): `list_supported_models()` → queries `http://localhost:8000`
- Cloud: `list_supported_models(base_url='http://www.modelscope.cn/twinkle')` → uses `MODELSCOPE_TOKEN`

**Always call `list_supported_models` first** to verify which models the server actually supports before writing the training script.

### Two Client APIs

| | Twinkle Native API | Tinker-Compatible API |
|---|---|---|
| Init | `init_twinkle_client(base_url, api_key)` | `init_tinker_client()` → `from tinker import ServiceClient` |
| Model | `MultiLoraTransformersModel(model_id=...)` | `ServiceClient(...).create_lora_training_client(...)` |
| Forward/backward | `model.forward_backward(inputs=batch)` | `training_client.forward_backward(datums, loss_fn)` |
| Optimizer step | `model.clip_grad_and_step()` | `training_client.optim_step(AdamParams(...))` |
| Save checkpoint | `model.save(name=..., save_optimizer=True)` | `training_client.save_state(name)` |
| Sampling/Inference | `vLLMSampler` via `model.save(..., is_sampler=True)` | `service_client.create_sampling_client(model_path=..., base_model=...)` |
| Best for | Full control, TUI integration, GRPO | Quick prototyping, Tinker migration |

### Cloud SFT — Twinkle Native Client

```python
import os
from peft import LoraConfig
from twinkle import get_logger, init_twinkle_client
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle_client.model import MultiLoraTransformersModel

logger = get_logger()

# Configuration
base_model = 'Qwen/Qwen3.6-27B'
base_url = 'http://www.modelscope.cn/twinkle'

# Step 1: Connect to ModelScope cloud server
client = init_twinkle_client(base_url=base_url, api_key=os.environ.get('MODELSCOPE_TOKEN'))

# Step 2: Check previous training runs (for resume)
runs = client.list_training_runs()
for run in runs:
    checkpoints = client.list_checkpoints(run.training_run_id)
    for cp in checkpoints:
        logger.info(f'Found checkpoint: {cp.twinkle_path}')

# Step 3: Prepare dataset (runs locally on CPU)
dataset = Dataset(DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=512)
dataset.map('SelfCognitionProcessor', init_args={'model_name': 'MyModel', 'model_author': 'MyTeam'})
dataset.encode(batched=True)
dataloader = DataLoader(dataset=dataset, batch_size=4)

# Step 4: Configure model (state lives on ModelScope cloud GPU)
model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')
lora_config = LoraConfig(target_modules='all-linear', r=8, lora_alpha=32)
model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
model.set_template('Qwen3_5Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')
model.set_optimizer('Adam', lr=1e-4)

# Step 5: Training loop
for epoch in range(3):
    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        if step % 2 == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'[Epoch {epoch} Step {step}] {metric.result}')
    # Save checkpoint (stored on ModelScope cloud)
    twinkle_path = model.save(name=f'epoch-{epoch}', save_optimizer=True)
    logger.info(f'Saved: {twinkle_path}')
```

### Cloud DPO — Twinkle Native Client

```python
import os
import numpy as np
import torch
from peft import LoraConfig
from twinkle import get_logger, init_twinkle_client
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle_client.model import MultiLoraTransformersModel
from twinkle.preprocessor import EmojiDPOProcessor

logger = get_logger()

base_model = 'Qwen/Qwen3.6-27B'
base_url = 'http://www.modelscope.cn/twinkle'

client = init_twinkle_client(base_url=base_url, api_key=os.environ.get('MODELSCOPE_TOKEN'))

# Dataset
dataset = Dataset(DatasetMeta('ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji', data_slice=range(5000)))
dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=2048)
dataset.map(EmojiDPOProcessor, init_args={'system': 'You are a helpful assistant.'})
dataset.encode()
dataloader = DataLoader(dataset=dataset, batch_size=4)

# Model
model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')
model.add_adapter_to_model('default', LoraConfig(target_modules='all-linear', r=8, lora_alpha=32),
                           gradient_accumulation_steps=2)
model.set_template('Qwen3_5Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('DPOLoss', beta=0.1, loss_type='sigmoid', reference_free=False, sft_weight=1.0)
model.add_metric('DPOMetric', beta=0.1)
model.set_optimizer('Adam', lr=1e-4)

# Training
def prepare_dpo_batch(batch):
    result = []
    for row in batch:
        base_fields = {k: v for k, v in row.items() if k not in ('positive', 'negative')}
        result.append({**base_fields, **row['positive']})
        result.append({**base_fields, **row['negative']})
    return result

for batch in dataloader:
    for row in batch:
        for key in row:
            if isinstance(row[key], np.ndarray):
                row[key] = row[key].tolist()
            elif isinstance(row[key], torch.Tensor):
                row[key] = row[key].cpu().numpy().tolist()
    dpo_batch = prepare_dpo_batch(batch)
    ref_outputs = model.forward_only(inputs=dpo_batch, disable_lora=True)
    model.forward_backward(inputs=dpo_batch, ref_outputs=ref_outputs.result)
    model.clip_grad_and_step()

model.save(name='dpo-final', save_optimizer=True)
```

### Cloud GRPO — Tinker-Compatible Client

For GRPO, the Tinker API provides `save_weights_and_get_sampling_client()` for weight sync:

```python
import os
import numpy as np
from tinker import types
from twinkle import init_tinker_client, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.template import Qwen3_5Template

logger = get_logger()

BASE_MODEL = 'Qwen/Qwen3.6-27B'
NUM_GENERATIONS = 4
MAX_STEPS = 100
LEARNING_RATE = 2e-5

# Step 1: Initialize Tinker client
init_tinker_client()
from tinker import ServiceClient

service_client = ServiceClient(
    base_url='http://www.modelscope.cn/twinkle',
    api_key=os.environ.get('MODELSCOPE_TOKEN')
)

# Step 2: Create training client
training_client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=16)

# Step 3: Dataset
dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train', data_slice=range(2000)))
dataset.set_template('Qwen3_5Template', model_id=f'ms://{BASE_MODEL}', max_length=4096, enable_thinking=True)
dataset.map(GSM8KProcessor(system='Solve step by step. Put answer in \\boxed{}.'))
dataset.encode(add_generation_prompt=True)
dataloader = DataLoader(dataset=dataset, batch_size=2)
template = Qwen3_5Template(model_id=f'ms://{BASE_MODEL}')

advantage_fn = GRPOAdvantage()
reward_fn = GSM8KAccuracyReward()
sampling_client = None

for step, batch in enumerate(dataloader):
    if step >= MAX_STEPS:
        break

    # Weight sync: save weights and get a sampling client
    sampling_client = training_client.save_weights_and_get_sampling_client(name=f'step-{step}')

    # Sample completions
    prompts = batch if isinstance(batch, list) else [batch]
    all_sequences, all_user_data = [], []
    for prompt_feature in prompts:
        input_ids = prompt_feature['input_ids']
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        result = sampling_client.sample(
            prompt=types.ModelInput.from_ints(input_ids),
            sampling_params=types.SamplingParams(max_tokens=4096, temperature=1.0, top_p=0.95),
            num_samples=NUM_GENERATIONS,
        ).result()
        for _ in range(NUM_GENERATIONS):
            all_user_data.append(prompt_feature.get('user_data', []))
        all_sequences.extend(result.sequences)

    # Compute rewards & advantages
    trajectories = []
    for idx, seq in enumerate(all_sequences):
        decoded = template.decode(seq.tokens, skip_special_tokens=True)
        trajectories.append({
            'messages': [{'role': 'assistant', 'content': decoded}],
            'user_data': all_user_data[idx]
        })
    rewards = reward_fn(trajectories)
    advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

    # Build training data
    training_data = []
    for i, seq in enumerate(all_sequences):
        prompt_ids = prompts[i // NUM_GENERATIONS]['input_ids']
        if hasattr(prompt_ids, 'tolist'):
            prompt_ids = prompt_ids.tolist()
        sampled_tokens = list(seq.tokens)
        logprobs = seq.logprobs if seq.logprobs else [0.0] * len(sampled_tokens)
        ob_len = len(prompt_ids) - 1
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(prompt_ids + sampled_tokens[:-1]),
            loss_fn_inputs={
                'target_tokens': [0] * ob_len + sampled_tokens,
                'weights': [0] * ob_len + [1] * len(sampled_tokens),
                'logprobs': types.TensorData.from_numpy(np.array([0.0] * ob_len + logprobs, dtype=np.float32)),
                'advantages': types.TensorData.from_numpy(np.array([0.0] * ob_len + [float(advantages[i])] * len(sampled_tokens), dtype=np.float32)),
            },
        )
        training_data.append(datum)

    # Train
    training_client.forward_backward(training_data, 'importance_sampling').result()
    optim_result = training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()
    logger.info(f'[Step {step}] metrics={optim_result.metrics}')

training_client.save_state('grpo-final').result()
```

### Cloud Inference (Sampling from Checkpoint)

After training, load a saved checkpoint for inference:

```python
import os
from tinker import types
from twinkle import init_tinker_client
from twinkle.data_format import Message, Trajectory
from twinkle.template import Template

init_tinker_client()
from tinker import ServiceClient

base_model = 'Qwen/Qwen3.6-27B'
service_client = ServiceClient(
    base_url='http://www.modelscope.cn/twinkle',
    api_key=os.environ.get('MODELSCOPE_TOKEN')
)

# Load checkpoint (twinkle:// path from training)
sampling_client = service_client.create_sampling_client(
    model_path='twinkle://xxx-checkpoint-path/weights/epoch-2',
    base_model=base_model
)

# Prepare prompt
template = Template(model_id=f'ms://{base_model}')
trajectory = Trajectory(messages=[
    Message(role='system', content='You are a helpful assistant'),
    Message(role='user', content='Who are you?'),
])
input_feature = template.encode(trajectory, add_generation_prompt=True)
prompt = types.ModelInput.from_ints(input_feature['input_ids'].tolist())

# Sample
result = sampling_client.sample(
    prompt=prompt,
    sampling_params=types.SamplingParams(max_tokens=128, temperature=0.7),
    num_samples=1,
).result()

for i, seq in enumerate(result.sequences):
    print(f'{i}: {template.decode(seq.tokens)}')
```

### Cloud vs Local Decision

| Scenario | Use Cloud (ModelScope) | Use Local (Self-Hosted) |
|----------|------------------------|------------------------|
| No local GPUs | Yes | No |
| Quick experiments | Yes (zero setup) | No (need Ray + server) |
| Production RL with custom env | No (need low latency) | Yes |
| Large-scale multi-node | No | Yes |
| Privacy-sensitive data | No | Yes |
| TUI-managed training | Possible (still stateless client) | Yes (full control) |
| Cost sensitive (own hardware) | No | Yes |

**Key difference:** Cloud mode uses `base_url='http://www.modelscope.cn/twinkle'` + `MODELSCOPE_TOKEN`. Everything else (dataset loading, training loop, client API) is identical.

## Plugin Architecture & Component System

Twinkle is a **plugin-based framework**. All core components (loss functions, preprocessors, metrics, samplers, etc.) are registered as plugins and can be extended by writing new implementations.

### Local Self-Hosted Mode (Full Plugin Support)

When running against a self-hosted Twinkle Server, you have **full extensibility**:

```python
# Register a custom loss function (local mode only)
from twinkle.loss import BaseLoss, register_loss

@register_loss('MyCustomLoss')
class MyCustomLoss(BaseLoss):
    def forward(self, logits, labels, **kwargs):
        # Custom loss logic
        ...

# Then use in training script:
model.set_loss('MyCustomLoss', my_param=0.5)
```

You can write custom:
- **Loss functions** — inherit from `BaseLoss`, register with `@register_loss`
- **Preprocessors** — inherit from `BasePreprocessor`, register with `@register_preprocessor`
- **Metrics** — inherit from `BaseMetric`
- **Reward functions** — any callable `(responses) -> scores`
- **Samplers** — custom sampling strategies

### Cloud Service Mode (Built-in Components Only)

**IMPORTANT RESTRICTION**: The ModelScope online environment (`http://www.modelscope.cn/twinkle`) **does NOT support**:
- Passing custom classes or functions
- Pickle-based serialization of user code
- Any dynamically defined components

This is due to **security sandboxing** on the cloud platform.

In Cloud Service Mode, you can **only use built-in (pre-registered) components**:

| Component | Available Built-in Options |
|-----------|---------------------------|
| Loss | `CrossEntropyLoss`, `GRPOLoss`, `DPOLoss`, `GKDLoss` |
| Preprocessor | `SFTPreprocessor`, `RLPreprocessor`, `DPOPreprocessor` |
| Sampler | `vLLMSampler` |

```python
# ✅ Cloud-safe: use built-in by name
model.set_loss('GRPOLoss', epsilon=0.2)

# ❌ NOT possible in cloud: custom class
# model.set_loss(MyCustomLoss())  # Will fail — cannot serialize class
```

### Decision Rule

| Scenario | Mode | Custom Components? |
|----------|------|--------------------|
| Need custom loss/preprocessor | Local self-hosted | ✅ Full support |
| Standard training (SFT/GRPO/DPO) | Either | Use built-in by name |
| No local GPU, quick experiment | Cloud service | ❌ Built-in only |

**When writing scripts for the TUI Agent**: Always check whether the target is local or cloud. If cloud, restrict to built-in component names passed as strings.

---

## Training Methods

### SFT (Supervised Fine-Tuning)

```python
model.set_loss('CrossEntropyLoss')
dataset.encode(add_generation_prompt=False)  # include assistant in labels
for batch in dataloader:
    model.forward_backward(inputs=batch)
    model.clip_grad_and_step()
```

### GRPO (Group Relative Policy Optimization)

```python
model.set_loss('GRPOLoss', epsilon=0.2)
# Loop: sample -> reward -> advantage -> train
ckpt_manager.sync_weights(merge_and_sync=False)
responses = sampler.sample(prompts, sampling_params)
rewards = reward_fn(responses)
advantages = GRPOAdvantage()(rewards, num_generations=NUM_GENERATIONS, scale='group')
model.forward_backward(inputs=inputs, old_logps=old_logps, advantages=advantages, micro_batch_size=2)
model.clip_grad_and_step()
```

### DPO (Direct Preference Optimization)

```python
model.set_loss('DPOLoss', beta=0.1)
# Data: chosen + rejected concatenated as a single batch (first half chosen, second half rejected)
# Preprocessor outputs Trajectory with extend_message=[('rejected_messages', rejected)]
model.forward_backward(inputs=dpo_batch)
model.clip_grad_and_step()
```

### GKD (Generalized Knowledge Distillation)

```python
model.set_loss('GKDLoss', beta=0.5, temperature=1.0)
# On-policy: sample from student, score with teacher
# Off-policy: use pre-computed teacher logits
```

### PT (Pre-Training)

```python
# No LoRA, full parameter training
model = TransformersModel(model_id=MODEL_ID, device_mesh=mesh)
model.set_optimizer('AdamW', lr=1e-4)
# Use streaming dataset for large corpora
dataset = Dataset(DatasetMeta(dataset_id='...'), streaming=True)
```

## TUI Integration & Training Observability

### CRITICAL CONSTRAINTS

1. **DO NOT modify the Twinkle SDK itself (`src/twinkle/` or `src/twinkle_client/`).** All training logic MUST be implemented as external scripts.
2. **Scripts MUST use Server Mode** (`twinkle_client` for model operations + `twinkle` for data/dataset). This enables stateless client architecture where kill = pause, restart = resume.
3. **A local Twinkle Server MUST be running** before executing any training script.

### Server Mode Architecture

In Server Mode, all model state (LoRA weights, optimizer momentum, LR scheduler, cur_step) lives in the Server's GPU memory. The client script is **completely stateless** — killing the client loses nothing. Restarting with the same `adapter_name` seamlessly continues training.

```
┌────────────────────────────────────────────────────────────────┐
│  Twinkle Server (Ray Cluster)                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ ModelManagement (GPU)                                        │  │
│  │   ├─ Base Model (shared weights)                              │  │
│  │   ├─ adapter 'exp-01': OptimizerGroup(optimizer, lr_sched...)  │  │
│  │   └─ adapter 'exp-02': OptimizerGroup(optimizer, lr_sched...)  │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
         ↑ HTTP (forward_backward, clip_grad_and_step, save, ...)
┌────────────────────────────────────────────────────────────────┐
│  Client Script (CPU only, stateless)                           │
│   ├─ Data loading (twinkle.dataset)                             │
│   ├─ Training loop logic                                        │
│   ├─ Reward computation (client-side)                            │
│   └─ Metrics logging (TrainingRuntime)                           │
└────────────────────────────────────────────────────────────────┘
```

**Key implications:**
- "Pause" = kill client process (server retains all state in GPU memory)
- "Resume" = start a new client with the same adapter_name
- "Modify config" = kill client → edit script → restart (zero-cost iteration)
- No need for checkpoint resume from disk (optimizer state is live in server)
- `adapter_timeout` in server config controls when idle adapters are cleaned up

### Starting the Local Server

Before running any training script, the server must be started:

**1. Server config (`server_config.yaml`):**

```yaml
proxy_location: EveryNode
http_options:
  host: 0.0.0.0
  port: 8000

persistence:
  mode: file
  file_path: /tmp/twinkle_state.json

applications:
  - name: server
    route_prefix: /api/v1
    import_path: server
    args:
      server_config:
        per_token_model_limit: 3
      supported_models:
        - Qwen/Qwen3.5-4B
    deployments:
      - name: TinkerCompatServer
        max_ongoing_requests: 50
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 128
        ray_actor_options:
          num_cpus: 0.1

  - name: models-Qwen3.5-4B
    route_prefix: /api/v1/model/Qwen/Qwen3.5-4B
    import_path: model
    args:
      backend: transformers
      model_id: "ms://Qwen/Qwen3.5-4B"
      max_length: 10240
      nproc_per_node: 1
      device_group:
        name: model
        ranks: 1
        device_type: cuda
      device_mesh:
        device_type: cuda
        dp_size: 1
      adapter_config:
        adapter_timeout: 600  # seconds before idle adapter cleanup
    deployments:
      - name: ModelManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 16
        ray_actor_options:
          num_cpus: 0.1

  - name: processor
    route_prefix: /api/v1/processor
    import_path: processor
    args:
      ncpu_proc_per_node: 2
      device_group:
        name: model
        ranks: 2
        device_type: CPU
      device_mesh:
        device_type: CPU
        dp_size: 2
    deployments:
      - name: ProcessorManagement
        autoscaling_config:
          min_replicas: 1
          max_replicas: 1
          target_ongoing_requests: 128
        ray_actor_options:
          num_cpus: 0.1
```

**2. Server startup script (`server.py`):**

```python
import os
os.environ['TWINKLE_TRUST_REMOTE_CODE'] = '0'

from twinkle.server import launch_server

file_dir = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(file_dir, 'server_config.yaml')
launch_server(config_path=config_path)
```

**3. Launch sequence (`run.sh`):**

```bash
export RAY_ROTATION_MAX_BYTES=1024
export RAY_ROTATION_BACKUP_COUNT=1
# Start Ray head (adjust GPU list for your machine)
CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --port=6379 --num-gpus=4 --disable-usage-stats --include-dashboard=false
# Optional: additional GPU workers
# CUDA_VISIBLE_DEVICES=4,5,6,7 ray start --address=127.0.0.1:6379 --num-gpus=4
# CPU worker for processor
CUDA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0
# Start Twinkle server (blocks)
python server.py
```

### Writing Training Scripts (Server Mode)

Training scripts use `twinkle_client` for model operations and `twinkle` for data:

```python
# Imports: twinkle for data, twinkle_client for model
from twinkle import init_twinkle_client
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_tui.runtime import TrainingRuntime
from peft import LoraConfig

# Step 1: Connect to server
client = init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_API_KEY')

# Step 2: Prepare data (runs on client CPU)
dataset = Dataset(DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
dataset.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B', max_length=512)
dataset.encode(batched=True)
dataloader = DataLoader(dataset=dataset, batch_size=4)

# Step 3: Configure model (all state lives on server)
model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
lora_config = LoraConfig(target_modules='all-linear', r=8, lora_alpha=32)
model.add_adapter_to_model('my-experiment', lora_config, gradient_accumulation_steps=2)
model.set_template('Qwen3_5Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')
model.set_optimizer('Adam', lr=1e-4)

# Step 4: Training loop with observability
rt = TrainingRuntime(run_id='my-experiment')
rt.start(model_id='Qwen/Qwen3.5-4B', config={'lr': 1e-4, 'batch_size': 4}, script_path=__file__)
rt.register_graceful_shutdown(model, dataloader)  # SIGTERM saves checkpoint

for step, batch in enumerate(dataloader):
    model.forward_backward(inputs=batch)
    model.clip_grad_and_step()

    if step % 2 == 0:
        metric = model.calculate_metric(is_training=True)
        rt.log_metrics(step=step, total_steps=len(dataloader), **metric.result)
        rt.log(f'[Step {step}/{len(dataloader)}] metrics={metric.result}')

model.save(name='final', save_optimizer=True)
rt.finish(status='completed')
```

### Training Control (Kill & Restart Pattern)

Since the server retains all state, training control is simple:

| Action | Signal | Training Script Behavior | Resume Method |
|--------|--------|-------------------------|---------------|
| Pause | SIGKILL (Ctrl+C) | Immediate death | Restart same script (server keeps adapter state) |
| Stop | SIGTERM | Saves checkpoint + dataloader state → exits | `model.resume_from_checkpoint()` + `dataloader.resume_from_checkpoint()` |
| Modify & Restart | SIGKILL → edit → run | New config applied, optimizer state preserved | Restart same adapter_name |
| Reset training | Use new adapter_name | Fresh start | Old adapter expires per `adapter_timeout` |

**Important:**
- `adapter_timeout` in `server_config.yaml` controls how long an idle adapter stays in GPU memory. Set it high (e.g., 600s) for iterative development.
- SIGTERM allows the script to save a clean checkpoint (model + optimizer + dataloader position) before exiting.
- SIGKILL is fine for quick pause since server retains everything — but you lose the dataloader position.

### Graceful Shutdown (MUST register in every script)

Every training script MUST register a SIGTERM handler for graceful stop:

```python
from twinkle_tui.runtime import TrainingRuntime

rt = TrainingRuntime(run_id='my-experiment')
rt.start(model_id=MODEL_ID, config={...}, script_path=__file__)

# Register SIGTERM handler — saves checkpoint + dataloader state on stop
rt.register_graceful_shutdown(model, dataloader)

# ... training loop ...
```

When TUI sends stop command → SIGTERM → script automatically:
1. Saves model checkpoint (`model.save(name='interrupted', save_optimizer=True, consumed_train_samples=...)`) 
2. Records dataloader position for exact resume
3. Logs the saved path
4. Calls `rt.finish(status='stopped')` and exits

To resume from this checkpoint later:
```python
progress = model.resume_from_checkpoint('twinkle://...path...')
dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
```

### TrainingRuntime (Observability Only)

`TrainingRuntime` provides structured logging for the TUI monitor. It does NOT control training flow — it only writes data:

```python
from twinkle_tui.runtime import TrainingRuntime

rt = TrainingRuntime(run_id='my-grpo-run')
rt.start(model_id=MODEL_ID, config={'lr': LR, 'batch_size': BATCH_SIZE}, script_path=__file__)

# In training loop:
rt.log_metrics(step=step, total_steps=MAX_STEPS, loss=loss, reward=reward, grad_norm=gn, lr=lr)
rt.log(f'[Step {step}] loss={loss:.4f}')

# When done:
rt.finish(status='completed')
```

### Required Metrics Logging

Log ALL available metrics every step for LLM analysis. The more data, the better the AI monitor can diagnose issues:

| Metric | Key | Why |
|--------|-----|-----|
| Step progress | `step`, `total_steps` | Track completion |
| Loss | `loss` | Core training signal |
| Reward (total) | `reward_total` | RL objective |
| Reward (per-component) | `reward_accuracy`, `reward_format`, etc. | Diagnose reward hacking |
| Gradient norm | `grad_norm` | Detect explosion/vanishing |
| Learning rate | `lr` | Verify scheduler |
| KL divergence | `kl_divergence` | Policy drift detection |
| Entropy | `entropy` | Detect mode collapse |
| Throughput | `throughput_samples_per_sec` | Performance regression |
| Completion length | `completion_avg_length` | Reward gaming detection |
| Epoch | `epoch` | Multi-epoch tracking |

Additional recommended metrics for specific scenarios:
- **GRPO**: `advantage_mean`, `advantage_std`, `clip_fraction`
- **DPO**: `chosen_reward`, `rejected_reward`, `reward_margin`
- **Multi-turn**: `avg_turns`, `success_rate`, `tool_call_count`

### Required Log Messages

Use `rt.log()` for human-readable events that help the LLM understand context:

```python
# Good: rich, contextual logs
rt.log(f'[Step {step}/{MAX_STEPS}] loss={loss:.4f} reward={reward:.3f} grad_norm={gn:.2f}')
rt.log(f'Sampling {NUM_GENERATIONS} completions per prompt, batch_size={BATCH_SIZE}')
rt.log(f'Weight sync to sampler completed: {adapter_uri}')
rt.log(f'Checkpoint saved: {checkpoint_path}')
rt.log(f'Dataset loaded: {len(dataset)} samples, template={template_name}')
rt.log(f'Reward breakdown: accuracy={acc:.3f} brevity={brev:.3f} format={fmt:.3f}')

# Bad: too terse, no context
rt.log(f'step {step}')  # ← useless, already in metrics
rt.log('training')       # ← meaningless
```

### GRPO Training Example (Server Mode)

Full GRPO example using `twinkle_client`:

```python
from twinkle import init_twinkle_client
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle.reward import GSM8KAccuracyReward
from twinkle.advantage import GRPOAdvantage
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler
from twinkle_tui.runtime import TrainingRuntime
from peft import LoraConfig

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
NUM_GENERATIONS = 4
MAX_STEPS = 100

client = init_twinkle_client(base_url='http://localhost:8000', api_key='EMPTY_API_KEY')

# Data
dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train', data_slice=range(2000)))
dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=2048, enable_thinking=False)
dataset.map(GSM8KProcessor(system='Solve with minimal reasoning. Put answer in \\boxed{}.'))
dataset.encode(add_generation_prompt=True)
dataloader = DataLoader(dataset=dataset, batch_size=2)

# Model
model = MultiLoraTransformersModel(model_id=MODEL_ID)
model.add_adapter_to_model('grpo-exp', LoraConfig(target_modules='all-linear', r=8, lora_alpha=32),
                           gradient_accumulation_steps=1)
model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
model.set_optimizer('Adam', lr=2e-5)
model.set_processor('InputProcessor')
model.set_template('Qwen3_5Template', model_id=MODEL_ID)

# Sampler
sampler = vLLMSampler(model_id=MODEL_ID)
sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)

# Metrics
rt = TrainingRuntime(run_id='grpo-gsm8k')
rt.start(model_id=MODEL_ID, config={'lr': 2e-5, 'num_gen': NUM_GENERATIONS}, script_path=__file__)
rt.register_graceful_shutdown(model, dataloader)  # SIGTERM saves checkpoint
advantage_fn = GRPOAdvantage()
reward_fn = GSM8KAccuracyReward()

current_adapter_uri = None
for step, batch in enumerate(dataloader):
    if step >= MAX_STEPS:
        break

    # Sync weights to sampler
    result = model.save(name='sampler-weights', save_optimizer=False, is_sampler=True)
    current_adapter_uri = result.twinkle_path

    # Sample
    responses = sampler.sample(
        inputs=batch,
        sampling_params={'max_tokens': 1024, 'temperature': 1.0, 'num_samples': NUM_GENERATIONS, 'logprobs': 1},
        adapter_uri=current_adapter_uri,
    )

    # Process responses, compute rewards and advantages
    all_inputs, all_old_logps = [], []
    for resp in responses:
        for seq in resp.sequences:
            all_inputs.append(seq.new_input_feature)
            all_old_logps.append([lp[0][1] for lp in seq.logprobs])

    rewards = reward_fn(all_inputs)
    advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

    # Train
    model.forward_backward(inputs=all_inputs, advantages=advantages, old_logps=all_old_logps)
    model.clip_grad_and_step()

    # Log
    metric = model.calculate_metric(is_training=True)
    rt.log_metrics(step=step, total_steps=MAX_STEPS, reward_mean=sum(rewards)/len(rewards), **metric.result)
    rt.log(f'[Step {step}] reward={sum(rewards)/len(rewards):.3f} loss={metric.result.get("loss", 0):.4f}')

model.save(name='grpo-final', save_optimizer=True)
rt.finish(status='completed')
```

### File Layout

```
~/.cache/twinkle/{run_id}/
├── meta.json          # Run metadata (model_id, config, status, pid, script_path, script_version)
├── train.py           # Current active script (always the latest version)
├── train_v1.py        # Archived: version 1 (original, possibly failed)
├── train_v2.py        # Archived: version 2 (first fix attempt)
├── metrics.jsonl      # One JSON line per step (all metrics)
└── logs.jsonl         # One JSON line per event (ts + msg)
```

**Script naming & versioning:**
- `train.py` is ALWAYS the current/active version. `resume_training` executes only this file.
- When script is updated (via `update_script` tool), old `train.py` is archived as `train_v{N}.py`
- `meta.json` tracks `script_version` (integer), `script_path`, and `pid`
- `run_id` is user-defined (e.g. `'grpo-gsm8k'`, `'sft-self-cognition'`)

**Initial script storage:**
```python
rt = TrainingRuntime(run_id='my-experiment')
rt.start(
    model_id='Qwen/Qwen3.5-4B',
    config={'lr': 1e-4, 'batch_size': 4},
    script_path=__file__,  # <-- copies current script to run_dir/train.py
)
```

**Script update flow (when fixing errors):**
1. Script fails → Agent reads logs/metrics to diagnose
2. Agent calls `update_script(run_id, new_code)` → archives old `train.py` as `train_v{N}.py`, writes new code
3. Agent calls `resume_training(run_id)` → re-executes the updated `train.py`
4. Same `run_id`, same server adapter state — only client logic changes

## Experiment Management

Each experiment MUST be organized in a dedicated folder:

```
experiments/{exp_name}/
├── plan.md              # Experiment design and hypothesis
├── config.yaml          # Training configuration (reproducible)
├── train.py             # Training script
├── train.sh             # Launch command with environment vars
├── logs/                # Training logs (auto-generated)
├── checkpoints/         # Model checkpoints (auto-generated)
└── results.md           # Results, metrics, conclusions
```

Always record in `plan.md`:
- Objective and hypothesis
- Baseline comparison
- Key hyperparameters and why
- Expected outcome and success criteria
