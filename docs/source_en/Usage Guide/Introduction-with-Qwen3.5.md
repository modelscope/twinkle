# Qwen3.5 Training Best Practices

Using Qwen3.5-4B as an example, this guide demonstrates the core capability of the Twinkle framework: **one component-based code, used from single GPU training to Client-Server mode**.

---

## 1. What is Twinkle

Twinkle is a production-oriented large model training framework. Its core design is straightforward: **training logic is expressed in Python code, and the runtime mode is switched via initialization parameters**.

This means:
- A training script written in the lab can be used to ray and server training by changing a single line
- Open to customize your training algorithm
- No need to maintain separate codebases to support different modes like torchrun, Ray, or HTTP
- Algorithm engineers focus on training logic; the framework handles distributed communication automatically

Twinkle supports both **Transformers** and **Megatron** backends, as well as **multi-tenant LoRA training** — multiple users share a single base model while each trains their own adapter.

---

## 2. Local Multi-GPU Training

### Overview

Training on 1–8 local GPUs or NPUs. Twinkle is built on PyTorch native interfaces and supports parallel strategies such as FSDP2 and DDP.

### Full Code

```python
from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

# Build device_mesh: fsdp=4, dp=2, using 8 GPUs in total
device_mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=2)
# Use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


def eval(model):
    # Validation set: 100 samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(100)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3.5-4B')
    dataset.map(SelfCognitionProcessor('twinkle LLM', 'ModelScope Community'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics


def train():
    # Training set: 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    # Set template to prepare encoding
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3.5-4B')
    # Preprocess: replace placeholders in self-cognition data
    dataset.map(SelfCognitionProcessor('twinkle LLM', 'ModelScope Community'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 8; each of the 8 GPUs processes 1 sample
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    # Load model
    model = TransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Add LoRA adapter named 'default'
    # Comment this out to switch to full-parameter training
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    # Configure optimizer for LoRA
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    # Configure learning rate scheduler
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    # LoRA training: ~18G * 4 GPU memory
    # Full-parameter training: ~50G * 4 GPU memory
    for step, batch in enumerate(dataloader):
        # Forward + backward pass
        model.forward_backward(inputs=batch)
        # Gradient clipping + optimizer step
        model.clip_grad_and_step()
        if step % 20 == 0:
            # Print training metrics
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        if step > 0 and step % 40 == 0:
            # Periodic evaluation
            metrics = eval(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = step
            # Save best checkpoint
            if loss_metric > float(metrics['loss']):
                model.save(f'checkpoint-{step}')
                loss_metric = float(metrics['loss'])
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()
```

### Launch Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 fsdp2.py
```

### Key Design Notes

**DeviceMesh Parallelism Strategy**

```python
device_mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=2)
```

A hybrid parallel strategy with 4-way FSDP sharding + 2-way data parallelism. Qwen3.5-4B weights occupy ~8GB in bf16 precision. In LoRA mode, single-GPU memory usage is around 18GB — 8× A100/H100 handles it comfortably.

**Gradient Accumulation**

```python
model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
```

`gradient_accumulation_steps=2` updates parameters every 2 micro-batches, effectively doubling the batch size. Useful when GPU memory is constrained but a larger effective batch is desired.

**Algorithm Transparency**

All key training steps — forward pass, backward pass, gradient clipping, checkpoint saving — are written directly in the main loop. Developers retain full control over the training process. The underlying distributed communication is handled by Twinkle's infra layer; switching between Ray and torchrun has no impact on the main loop.

For complex algorithms, this transparency is especially important.

### RL Training: Reinforcement Learning with Ray

Twinkle supports multiple RL algorithms, including GRPO, RLOO, GSPO, and more. Here we use GRPO (Group Relative Policy Optimization) as an example — the core RL algorithm used in DeepSeek-R1 — to show how RL training works in Ray mode.

Unlike PPO, GRPO does not require training a separate value model. Instead, it estimates the advantage function using relative rewards within a sampled group, simplifying the training pipeline and reducing memory overhead. Twinkle's Ray mode is particularly well-suited for RL algorithms that require **model and sampler to run on separate devices**. In the example below, 4 GPUs run model training while another 4 run vLLM sampling, coordinated through a Ray cluster:

```python
from typing import List, Dict, Any
from peft import LoraConfig
import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle.metric import CompletionRewardMetric
from twinkle.preprocessor.llm import GSM8KProcessor

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
MODEL_GPUS = 4      # 4 GPUs for model training
SAMPLER_GPUS = 4    # 4 GPUs for vLLM sampling
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = 8     # 8 samples per group
MAX_NEW_TOKENS = 4096
LEARNING_RATE = 1e-5
MAX_STEPS = 200
BATCH_SIZE = 16
MINI_BATCH_SIZE = 16
MICRO_BATCH_SIZE = 2
ADAPTER_NAME = 'default'

def create_gsm8k_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=2048)
    dataset.map(GSM8KProcessor())
    dataset.encode(add_generation_prompt=True)
    return dataset

def compute_rewards(trajectories: List[Dict[str, Any]]):
    accuracy_reward_fn = GSM8KAccuracyReward()
    format_reward_fn = GSM8KFormatReward()
    accuracy_rewards = accuracy_reward_fn(trajectories)
    format_rewards = format_reward_fn(trajectories)
    total_rewards = [a + f for a, f in zip(accuracy_rewards, format_rewards)]
    return total_rewards, format_rewards, accuracy_rewards

def main():
    # Assign model and sampler to separate GPU groups
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    # Initialize in Ray mode
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(target_modules='all-linear', r=32, lora_alpha=64, lora_dropout=0.05)

    # Model deployed in the 'model' group
    model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Template', model_id=MODEL_ID)

    # Sampler deployed in the 'sampler' group
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 4096,
            'max_lora_rank': 32,
            'enable_lora': False,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS)

    optim_step = 0
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        global_prompts = batch if isinstance(batch, list) else [batch]

        # Sync weights to sampler
        ckpt_manager.sync_weights(merge_and_sync=True)
        sampler.reset_prefix_cache()

        # Group sampling: sample NUM_GENERATIONS completions per prompt
        sample_response = sampler.sample(
            global_prompts * NUM_GENERATIONS,
            sampling_params,
            num_samples=1,
        )

        all_input_data = []
        all_old_logps = []
        all_completion_lengths = []

        for sequence in sample_response.sequences:
            all_input_data.append(sequence.new_input_feature)
            all_old_logps.append(sequence.logprobs)
            all_completion_lengths.append(len(sequence.tokens))

        # Compute rewards
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(all_input_data)
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'accuracy': accuracy_rewards,
            },
        )

        # GRPO advantage estimation: group-level normalization
        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        # Mini-batch training
        total_completions = len(all_input_data)
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            mb_old_logps = all_old_logps[mb_start:mb_end]
            mb_advantages = advantages[mb_start:mb_end]

            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1

            if optim_step >= MAX_STEPS:
                break
            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            metrics.reset()
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('grpo-gsm8k-checkpoint')

if __name__ == '__main__':
    main()
```

Since this runs on a Ray cluster, launching is simply:

```shell
python train.py
```

**Key Design Points for GRPO Training:**

1. **Model-sampler separation**: `DeviceGroup` splits 8 GPUs into two groups. Training and sampling run independently, allowing the sampling pipeline to fully leverage vLLM's high throughput.

2. **Group sampling strategy**: `global_prompts * NUM_GENERATIONS` produces multiple completions per prompt, enabling advantage estimation via intra-group relative rewards — no separate value model needed.

3. **Weight synchronization**: `ckpt_manager.sync_weights()` syncs the training model weights to vLLM before each sampling step, ensuring the sampler always uses the latest policy.

4. **Algorithm components exposed**: `GRPOAdvantage` and `GRPOLoss` are registered directly on the model and can be swapped for other RL algorithm components without modifying any other code.

The core value of this pattern: the entire RL training loop — sampling, reward computation, advantage estimation, gradient update — is laid out in a visible Python main loop with no hidden magic. Differences between RL algorithms typically amount to swapping a few components.

---

## 3. Remote Training: Client-Server Architecture

When compute resources and service consumers are separated — enterprise training platforms, cloud Serverless training services — training capabilities need to be exposed as an API.

Twinkle supports two client integration modes:
- **Twinkle Client**: API identical to local training, suitable for scenarios requiring fine-grained control
- **Tinker Client**: Compatible with the [Tinker](https://github.com/thinking-machines-lab/tinker) ecosystem, with a simpler calling style

The server maintains a single base model; multiple clients can train their own LoRA adapters in parallel.

### 3.1 Twinkle Client: Fine-Grained Control

Twinkle Client provides an API nearly identical to local training, ideal for scenarios that require fine-grained control over the training process.

```python
import dotenv
dotenv.load_dotenv('.env')

from peft import LoraConfig

from twinkle import get_logger
from twinkle.dataset import DatasetMeta
from twinkle_client import init_twinkle_client
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel

logger = get_logger()

# Initialize the Twinkle client
client = init_twinkle_client(base_url='http://127.0.0.1:8000', api_key='EMPTY_TOKEN')

# Query existing training runs and checkpoints
runs = client.list_training_runs()
resume_path = None
for run in runs:
    logger.info(run.model_dump_json(indent=2))
    checkpoints = client.list_checkpoints(run.training_run_id)
    for checkpoint in checkpoints:
        logger.info(checkpoint.model_dump_json(indent=2))
        # Uncomment to resume from a specific checkpoint:
        # resume_path = checkpoint.twinkle_path


def train():
    # Prepare dataset
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3.5-4B', max_length=512)
    dataset.map('SelfCognitionProcessor', init_args={'model_name': 'twinkle model', 'model_author': 'ModelScope Community'})
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    # Configure model
    model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen3.5-4B')

    lora_config = LoraConfig(target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    model.set_template('Template')
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('AdamW', lr=1e-4)
    model.set_lr_scheduler('LinearLR')

    # Resume from checkpoint if available
    if resume_path:
        logger.info(f'Resuming training from {resume_path}')
        model.load(resume_path, load_optimizer=True)

    logger.info(model.get_train_configs())

    for epoch in range(3):
        logger.info(f'Starting epoch {epoch}')
        for step, batch in enumerate(dataloader):
            # Forward + backward
            output = model.forward_backward(inputs=batch)

            if step % 2 == 0:
                logger.info(f'Current is step {step // 2}, loss: {output}')

            model.clip_grad_norm(1.0)
            model.step()
            model.zero_grad()
            model.lr_step()

        # Save checkpoint
        twinkle_path = model.save(name=f'twinkle-epoch-{epoch}', save_optimizer=True)
        logger.info(f'Saved checkpoint: {twinkle_path}')


if __name__ == '__main__':
    train()
```

**Twinkle Client highlights:**

- API identical to local training — no additional learning curve
- Supports checkpoint management and resume from checkpoint
- Dynamically swap LoRA adapters, loss functions, and optimizer components

### 3.2 Tinker Client: Simple and Ready to Use

Tinker is a lightweight training API. Twinkle provides full support for the Tinker client — a few lines of code is all it takes to start training. Existing Tinker-based projects can be migrated directly to a Twinkle server.

```python
import os
from tinker import types
from tqdm import tqdm

from twinkle import init_tinker_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.server.tinker.common import input_feature_to_datum

# Initialize Tinker client (must be called before importing ServiceClient)
init_tinker_client()

from tinker import ServiceClient

# Base model
base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://www.modelscope.cn/twinkle'


def train():
    # Prepare dataset
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Template', model_id=f'ms://{base_model}', max_length=256)
    dataset.map(SelfCognitionProcessor('Twinkle Model', 'ModelScope Team'), load_from_cache_file=False)
    dataset.encode(batched=True, load_from_cache_file=False)
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    # Initialize training client
    service_client = ServiceClient(
        base_url=base_url,
        api_key=os.environ.get('MODELSCOPE_TOKEN')
    )
    training_client = service_client.create_lora_training_client(base_model=base_model, rank=16)

    # Training loop
    for epoch in range(3):
        print(f'Epoch {epoch}')
        for step, batch in tqdm(enumerate(dataloader)):
            # Convert input format
            input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

            # Remote forward + backward
            fwdbwd_future = training_client.forward_backward(input_datum, 'cross_entropy')
            # Remote optimizer step
            optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

            # Wait for results
            fwdbwd_result = fwdbwd_future.result()
            optim_result = optim_future.result()
            print(f'Training Metrics: {optim_result}')

        # Save checkpoint
        save_future = training_client.save_state(f'twinkle-lora-{epoch}')
        save_result = save_future.result()
        print(f'Saved checkpoint to {save_result.path}')


if __name__ == '__main__':
    train()
```

**Tinker Client highlights:**

- Minimal API surface, easy to get started
- Fully compatible with the Tinker ecosystem — existing code migrates seamlessly
- Supports ModelScope's official training environment (see below)

### 3.3 ModelScope Official Training Environment

Alongside the open-source release of Twinkle, ModelScope provides a hosted model training service (Training as a Service, TaaS) powered by its own compute infrastructure. Developers can access Twinkle's training capabilities for free via API, without provisioning any GPUs.

**How to use:**

1. Register a ModelScope account and apply to join the [Twinkle-Explorers](https://modelscope.cn/organization/twinkle-explorers) organization
2. Obtain your API Key on the [Token Management page](https://www.modelscope.cn/my/access/token)
3. Use the Tinker Client code above with the following endpoint:

```python
base_url = 'https://www.modelscope.cn/twinkle'
base_model = 'Qwen/Qwen3-30B-A3B-Instruct-2507'  # Model currently deployed in the official environment
```

---

## 4. Choosing the Right Training Mode

| Scenario | Recommended Approach | Key Advantage |
|----------|----------------------|---------------|
| Local experimentation | Single GPU / torchrun | Code-as-config, high debugging efficiency |
| Large-scale distributed training | torchrun + FSDP2 / Ray | Native parallel performance, production-ready |
| Enterprise training platform | Twinkle Client + self-hosted server | Multi-tenant isolation, fine-grained control |
| Rapid prototyping | Tinker Client + ModelScope TaaS | Zero resource setup, instant access |
| Existing Tinker codebase | Tinker Client | Seamless migration, ecosystem compatibility |

**Recommendations:**

- If you are an algorithm researcher who frequently iterates on the training pipeline, start with torchrun mode and consider moving to a service-based setup once experiments are validated.
- If you are a platform engineer building an internal training service, deploy Twinkle Server and offer both Twinkle Client and Tinker Client based on your users' preferences.
- If you just want to try Twinkle quickly, use the ModelScope official environment — get your first training run done in 5 minutes.

Twinkle's design philosophy is **to give you the building blocks, not make the decisions for you**. Whether you need maximum performance at scale or maximum convenience via API, there's a solution that fits.
