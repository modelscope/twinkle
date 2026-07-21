# Twinkle Client

Twinkle Client is the native client, designed with the philosophy: **Change `from twinkle import` to `from twinkle_client import`, and you can migrate local training code to remote calls without modifying the original training logic**.

## Initialization

```python
from twinkle_client import init_twinkle_client

# Initialize client, connect to Twinkle Server
client = init_twinkle_client(
    base_url='http://127.0.0.1:8000',   # Server address
    api_key='your-api-key'               # Authentication token (can be set via environment variable TWINKLE_SERVER_TOKEN)
)
```

After initialization, the `client` object (`TwinkleClient`) provides the following management functions:

```python
# Health check
client.health_check()

# List current user's training runs
runs = client.list_training_runs(limit=20)

# Get specific training run details
run = client.get_training_run(run_id='xxx')

# List checkpoints
checkpoints = client.list_checkpoints(run_id='xxx')

# Get checkpoint path (for resuming training)
path = client.get_checkpoint_path(run_id='xxx', checkpoint_id='yyy')

# Get latest checkpoint path
latest_path = client.get_latest_checkpoint_path(run_id='xxx')
```

## Migrating from Local Code to Remote

Migration is very simple, just replace the import path from `twinkle` to `twinkle_client`:

```python
# Local training code (original)
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset
from twinkle.model import MultiLoraTransformersModel

# Remote training code (after migration)
# DataLoader and Dataset can be imported from either local twinkle or remote twinkle_client
from twinkle.dataloader import DataLoader        # or: from twinkle_client.dataloader import DataLoader
from twinkle.dataset import Dataset              # or: from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
```

Training loops, data processing, and other logic do not need any modifications.

## Complete Training Example (Transformers Backend)

```python
import dotenv
dotenv.load_dotenv('.env')

from peft import LoraConfig
from twinkle import get_logger
from twinkle.dataset import DatasetMeta
from twinkle_client import init_twinkle_client

# DataLoader and Dataset can be imported from either local twinkle or remote twinkle_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel

logger = get_logger()

base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://localhost:8000'
api_key = 'EMPTY_API_KEY'

# Step 1: Initialize client
client = init_twinkle_client(base_url=base_url, api_key=api_key)

# List available models on the server
print('Available models:')
for item in client.get_server_capabilities().supported_models:
    print('- ' + item.model_name)

# Step 2: Query existing training runs (optional, for resuming training)
runs = client.list_training_runs()
resume_path = None
for run in runs:
    logger.info(run.model_dump_json(indent=2))
    checkpoints = client.list_checkpoints(run.training_run_id)
    for checkpoint in checkpoints:
        logger.info(checkpoint.model_dump_json(indent=2))
        # Uncomment to resume from checkpoint:
        # resume_path = checkpoint.twinkle_path

# Step 3: Prepare dataset
# data_slice limits the number of samples loaded
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))

# Set chat template to match model's input format
dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=512)

# Data preprocessing: Replace placeholders with custom names
dataset.map('SelfCognitionProcessor',
            init_args={'model_name': 'twinkle model', 'model_author': 'ModelScope Team'})

# Encode dataset into tokens usable by the model
dataset.encode(batched=True)
# For large datasets, use num_proc to enable multi-process parallelism:
# dataset.encode(batched=True, num_proc=8)
# When using twinkle_client.dataset, encode calls the remote server over HTTP
# with a default 600s timeout; raise it via the timeout argument if needed:
# dataset.encode(batched=True, num_proc=8, timeout=3600)

# Create DataLoader
dataloader = DataLoader(dataset=dataset, batch_size=4)

# Step 4: Configure model
model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')

# Configure LoRA: apply low-rank adapters to all linear layers
lora_config = LoraConfig(target_modules='all-linear')
# gradient_accumulation_steps=2: accumulate gradients over 2 micro-batches before each optimizer step
model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)

# Set template, processor, loss function
model.set_template('Qwen3_5Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')

# Set optimizer (only Adam is supported if the server uses Megatron backend)
model.set_optimizer('Adam', lr=1e-4)

# Set LR scheduler (not supported if the server uses Megatron backend)
# model.set_lr_scheduler('LinearLR')

# Step 5: Resume training (optional)
start_step = 0
if resume_path:
    logger.info(f'Resuming from checkpoint {resume_path}')
    progress = model.resume_from_checkpoint(resume_path)
    dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
    start_step = progress['cur_step']

# Step 6: Training loop
logger.info(model.get_train_configs().model_dump())

for epoch in range(3):
    logger.info(f'Starting epoch {epoch}')
    for cur_step, batch in enumerate(dataloader, start=start_step + 1):
        # Forward propagation + backward propagation
        model.forward_backward(inputs=batch)

        # Gradient clipping + optimizer update (equivalent to calling clip_grad_norm / step / zero_grad / lr_step in sequence)
        model.clip_grad_and_step()

        # Print metric every 2 steps (aligned with gradient_accumulation_steps)
        if cur_step % 2 == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {cur_step} of {len(dataloader)}, metric: {metric.result}')

    # Step 7: Save checkpoint
    twinkle_path = model.save(
        name=f'twinkle-epoch-{epoch}',
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )
    logger.info(f'Saved checkpoint: {twinkle_path}')

# Step 8: Upload to ModelScope Hub (optional)
# YOUR_USER_NAME = "your_username"
# hub_model_id = f'{YOUR_USER_NAME}/twinkle-self-cognition'
# model.upload_to_hub(
#     checkpoint_dir=twinkle_path,
#     hub_model_id=hub_model_id,
#     async_upload=False
# )
```

For checkpoint resumption, the recommended client-side flow is:

1. Query the server for an existing checkpoint path with `client.list_checkpoints(...)` or `client.get_latest_checkpoint_path(...)`.
2. Call `model.resume_from_checkpoint(resume_path)` to restore weights, optimizer, scheduler, RNG, and progress metadata.
3. Call `dataloader.resume_from_checkpoint(progress['consumed_train_samples'])` to skip already-consumed samples.

This matches the end-to-end example in `cookbook/client/twinkle/self_cognition.py`.

## Differences with Megatron Backend

When using the Megatron backend, the main differences in client code:

```python
# Megatron backend does not need explicit loss setting (computed internally by Megatron)
# model.set_loss('CrossEntropyLoss')  # Not needed

# Optimizer and LR scheduler use Megatron built-in defaults
model.set_optimizer('default', lr=1e-4)
model.set_lr_scheduler('default', lr_decay_steps=1000, max_lr=1e-4)
```

The rest of the data processing, training loop, checkpoint saving, and other code remains exactly the same.

## Trainable Multi-turn Rollout (ClientMultiTurnRollout)

The examples above are all single-turn training. If you want to do **multi-turn agentic RL with tool use** (e.g. GRPO) and need training-ready token-level alignment info, use `twinkle_client.rollout.ClientMultiTurnRollout`. It drives the "sample → call tool → stitch context → sample again" multi-turn loop on the client side, samples over HTTP each round (`/twinkle/sample`), and produces a trainable result with `logprobs` per trajectory that can be fed directly into GRPO and other RL training.

### Dependencies and Constraints

- **Local Template**: bridge-token stitching (rendering tool turns + the next generation prompt) requires a local `Template` instance on the client.
- **vLLMSampler**: the client sampler pointing at the server's Sampler service.
- **ToolManager** (optional): register your tools; if a trajectory produces tool_calls but no tool_manager is provided, a `ValueError` is raised at dispatch.
- **`num_samples=1`**: each trajectory is sampled once. For a GRPO group, replicate the same prompt into `NUM_GENERATIONS` independent trajectories.

### Minimal Example

```python
from peft import LoraConfig
from twinkle import init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.template import Qwen3_5Template
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.rollout import ClientMultiTurnRollout
from twinkle_client.sampler import vLLMSampler

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
NUM_GENERATIONS = 2   # GRPO group size (rollout samples num_samples=1 per trajectory)

init_twinkle_client(base_url='http://127.0.0.1:8000', api_key='EMPTY_TOKEN')

# Training model (GRPO)
model = MultiLoraTransformersModel(model_id=MODEL_ID)
model.add_adapter_to_model('default', LoraConfig(target_modules='all-linear', r=16, lora_alpha=32))
model.set_loss('GRPOLoss', epsilon=0.2)
model.set_optimizer('Adam', lr=1e-5)
model.set_processor('InputProcessor')
model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

# Client sampler (HTTP)
sampler = vLLMSampler(model_id=MODEL_ID)
sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

# Multi-turn rollout: needs a local Template (bridge stitching) and a ToolManager
rollout_template = Qwen3_5Template(model_id=MODEL_ID, max_length=8192, enable_thinking=False)
rollout_template.truncation_strategy = 'delete'
tool_manager = ToolManager([MyCalculatorTool()])   # your tools

rollout = ClientMultiTurnRollout(
    sampler=sampler,
    template=rollout_template,
    tool_manager=tool_manager,
    sampling_params=SamplingParams(max_tokens=512, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95),
    max_turns=4,
)
advantage_fn = GRPOAdvantage()

for step in range(3):
    # 1. Batched multi-turn rollout: replicate each prompt into NUM_GENERATIONS trajectories
    trajectories = build_trajectories(tool_manager.tool_infos())  # see cookbook
    rolled = rollout(trajectories, tool_manager=tool_manager)

    # 2. Read back token-level logprobs (top-1) and rewards
    all_inputs, all_old_logps = [], []
    for traj in rolled:
        all_old_logps.append([lp[0][1] for lp in (traj.get('logprobs') or [])])
        all_inputs.append(traj)
    rewards = compute_rewards(rolled)   # see cookbook

    # 3. GRPO advantages (group-relative)
    advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

    # 4. Policy update
    model.forward_backward(inputs=all_inputs, advantages=advantages, old_logps=all_old_logps)
    model.clip_grad_and_step()
```

### Output Fields

Each returned trajectory has the following top-level fields appended to the original dict:

| Field | Meaning |
|:------|:--------|
| `messages` | Full multi-turn conversation (including assistant tool_calls and tool-response turns) |
| `logprobs` | Top-1 logprob per trainable token; `None` if the round sampled no logprobs |
| `turns` | Number of turns actually taken (`<= max_turns`) |
| `stop_reason` | One of `'stop'` / `'length'` / `'max_turns'` |
| `truncated` | Whether truncated due to `max_turns` or the length cap |

### Common Errors

- A trajectory triggered a tool call but no `tool_manager` was provided → raises `ValueError`. Pass `tool_manager` at construction or per call.
- The sampler's network / timeout errors are **raised as-is** (not swallowed); handle retry/backoff outside your loop.

See `cookbook/client/twinkle/multi_turn_rollout.py` for a full runnable example.
