# Embedding Training

Twinkle supports contrastive embedding model training with InfoNCE loss, in-batch negatives, and cross-rank gathering. This guide demonstrates how to train embedding models using Twinkle, including an advanced architecture with online compression via a frozen vLLM condenser.

---

## Overview

Embedding training in Twinkle uses the following core components:

| Component | Role |
|:----------|:-----|
| `InfonceLoss` | Contrastive loss with in-batch negatives |
| `EmbeddingMetric` | Tracks pos/neg similarity and loss |
| `TransformersModel` | Trainable embedding model (with LoRA or full) |
| `vLLMSampler` | Optional: online compression condenser |
| `InputProcessor` | Processes anchor/positive pairs into features |

### Data Format

Each training sample consists of **(anchor, positive)** pairs. In the embedding feature tensor:

```
embeddings: [anchor_0, positive_0, anchor_1, positive_1, ...]
labels:     [       1,         0,        1,          0, ...]
```

- `labels=1` marks the start of a new group (anchor)
- `labels=0` marks positives/negatives within the group

---

## Basic Embedding Training

A minimal embedding training script with DDP:

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.dataloader import DataLoader
from twinkle.loss import InfonceLoss
from twinkle.metric import EmbeddingMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.template import Qwen3_5Template

logger = get_logger()

# --- Configuration ---
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
MODEL_GPUS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
TEMPERATURE = 0.07
EMB_MAX_LENGTH = 8192

# --- Initialize ---
device_groups = [
    DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
]
model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS, groups=device_groups)

# --- Model ---
model = TransformersModel(
    model_id=MODEL_ID,
    device_mesh=model_mesh,
    remote_group='model',
    ddp_config={'find_unused_parameters': True},
)
model.set_processor(InputProcessor)
model.set_loss(InfonceLoss, temperature=TEMPERATURE, use_batch=True)
model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
model.set_lr_scheduler(
    scheduler_cls='CosineWarmupScheduler',
    num_warmup_steps=200,
    num_training_steps=total_steps,
)
model.add_metric(EmbeddingMetric, is_training=True)

# --- Template ---
template = Qwen3_5Template(
    model_id=MODEL_ID,
    max_length=EMB_MAX_LENGTH,
    enable_thinking=False,
)

# --- Training Loop ---
for step, batch in enumerate(dataloader):
    # batch: list of features with anchor/positive pairs
    model.forward_backward(inputs=batch, task='embedding')
    model.clip_grad_and_step(gradient_accumulation_steps=1)

    if step % 10 == 0:
        metric = model.calculate_metric(is_training=True)
        logger.info(f'Step {step}: {metric}')
```

### Key Parameters

| Parameter | Recommended | Description |
|:----------|:------------|:------------|
| `temperature` | 0.05–0.1 | Lower = sharper contrast. 0.07 keeps gradients flowing until cosine > 0.75 |
| `use_batch` | True | Enables cross-sample in-batch negatives for better efficiency |
| `hard_negatives` | None or 7 | Fix negative count per sample; None uses all in-batch |
| `find_unused_parameters` | True | Required for embedding models (only last hidden state contributes gradients) |

---

## Advanced: Online Compression Architecture

For training retrieval-augmented embeddings, Twinkle supports a sophisticated architecture where a frozen vLLM condenser compresses text online during training.

### Architecture (8 GPUs)

```
┌─────────────────────────────────────────────────────────┐
│ GPU 0-3: Trainable Embedding Model (LoRA)               │
│   TransformersModel + InfonceLoss + EmbeddingMetric     │
├─────────────────────────────────────────────────────────┤
│ GPU 4-7: Frozen vLLM Condenser                          │
│   vLLMSampler (online text compression)                 │
└─────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   Embedding features ◄── Compressed text from condenser
```

### Pipeline

1. **Prefetch**: Load a mega-batch (batch_size × prefetch_multiplier)
2. **Compress**: Feed (query, CoT) pairs through the vLLM condenser to produce dense summaries
3. **Validate**: Check compression quality; fall back to external API if truncated
4. **Encode**: Convert compressed texts to embedding features via template
5. **Train**: Forward/backward with InfoNCE loss on mini-batches

### Device Groups Setup

```python
device_groups = [
    DeviceGroup(name='model',
                ranks=list(range(MODEL_GPUS)),
                device_type='GPU'),
    DeviceGroup(name='condenser_sampler',
                ranks=list(range(MODEL_GPUS, MODEL_GPUS + CONDENSER_GPUS)),
                device_type='GPU'),
]
model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
condenser_mesh = DeviceMesh.from_sizes(world_size=CONDENSER_GPUS, dp_size=CONDENSER_GPUS)

twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS + CONDENSER_GPUS, groups=device_groups)
```

### Condenser Sampler Configuration

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams

condenser_sampler = vLLMSampler(
    model_id='ms://twinkle-kit/Qwen3.5-4B-CM-v2',
    engine_args={
        'gpu_memory_utilization': 0.8,
        'max_model_len': 32768,
    },
    device_mesh=condenser_mesh,
    remote_group='condenser_sampler',
)
condenser_sampler.set_template(
    'Qwen3_5Template',
    model_id='ms://twinkle-kit/Qwen3.5-4B-CM-v2',
    enable_thinking=False,
    truncation_strategy='delete',
    max_length=32768,
)

compress_params = SamplingParams(
    max_tokens=8192,
    temperature=0.2,
    top_p=0.5,
    num_samples=1,
)
```

### Compression Quality Validation

The script validates condenser output structure before using it for training:

```python
def _is_truncated_compression(text: str, schema: str = 'new') -> bool:
    """Reject incomplete or schema-regressed condenser output."""
    if not text or '## More' not in text or '## Summary' not in text:
        return True
    # Check ## More section has content
    after_more = text.split('## More', 1)[1].strip()
    if not after_more:
        return True
    # For 'new' schema: verify Problem/Skill/Knowledge markers
    if schema == 'new':
        summary_body = text.split('## Summary', 1)[1].split('## More', 1)[0]
        if not all(m in summary_body for m in ('Problem:', 'Skill:', 'Knowledge:')):
            return True
    return False
```

When validation fails, the system falls back to an external OpenAI-compatible API:

```python
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient

api_client = OpenAIClient(
    model='qwen3.7-max',
    api_key=os.environ['COMPRESS_API_KEY'],
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
)
```

### Prefetch with ThreadPoolExecutor

To overlap compression and training, the script uses a background prefetch thread:

```python
from concurrent.futures import ThreadPoolExecutor

prefetch_executor = ThreadPoolExecutor(max_workers=1)

batch_iter = iter(dataloader)
first = next(batch_iter)
future = prefetch_executor.submit(_sample_batch, first)

for raw_mega_batch in batch_iter:
    # Wait for previous compression to complete
    minibatches = future.result()
    # Start compressing next batch in background
    future = prefetch_executor.submit(_sample_batch, raw_mega_batch)

    # Train on current minibatches
    for mb in minibatches:
        model.forward_backward(inputs=mb, task='embedding')
        model.clip_grad_and_step(gradient_accumulation_steps=1)
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `MODEL_ID` | `ms://Qwen/Qwen3.5-4B` | Base embedding model |
| `CONDENSE_MODEL_ID` | `ms://twinkle-kit/Qwen3.5-4B-CM-v2` | Condenser model |
| `MODEL_GPUS` | 4 | GPUs for the trainable model |
| `CONDENSER_SAMPLER_GPUS` | 4 | GPUs for the frozen condenser |
| `BATCH_SIZE` | 32 | Per-step effective batch size |
| `PREFETCH_BATCH_MULTIPLIER` | 8 | Mega-batch = BATCH_SIZE × this |
| `RESUME_CHECKPOINT` | `` | Path to resume from |
| `RESUME_STEP` | 0 | Step to resume from |
| `COMPRESS_API_KEY` | `` | API key for fallback compression |
| `COMPRESS_BASE_URL` | DashScope | Base URL for fallback API |
| `COMPRESS_MODEL` | `qwen3.7-max` | Model for API fallback |
| `API_CONCURRENCY` | 8 | Max concurrent API calls |
| `SAMPLER_TIMEOUT` | 300 | vLLM timeout before API fallback (seconds) |

### Hyperparameters

| Parameter | Value | Notes |
|:----------|:------|:------|
| Learning rate | 1e-5 | With CosineWarmup (200 warmup steps) |
| Temperature | 0.07 | Keeps gradient on diagonal pairs until cosine > 0.75 |
| Max embedding length | 8192 | Token limit for anchor/positive |
| Compression max tokens | 8192 | Max generation length for condenser |
| Gradient accumulation | 1 | Adjust for memory constraints |

---

## Fault Tolerance

The advanced script includes several resilience mechanisms:

- **Sampler timeout & rebuild**: If the vLLM sampler hangs beyond `SAMPLER_TIMEOUT`, actors are killed and the sampler is recreated from scratch
- **API fallback**: Truncated/invalid compressions automatically trigger an external API call
- **Failure logging**: Failed compressions are logged to `failures.jsonl` for offline SFT data regeneration
- **Response logging**: All compression results (both vLLM and API) are logged to `responses.jsonl` for debugging

---

## Monitoring

The `EmbeddingMetric` reports key training signals:

| Metric | What it means |
|:-------|:--------------|
| `pos_sim` | Average anchor-positive cosine similarity (target: > 0.8) |
| `neg_sim` | Average anchor-negative similarity (target: < 0.3) |
| `loss` | InfoNCE loss value |
| `grad_norm` | Gradient magnitude |

Healthy training shows `pos_sim` rising and `neg_sim` stable or falling. If `pos_sim` saturates near 1.0, lower the temperature.
