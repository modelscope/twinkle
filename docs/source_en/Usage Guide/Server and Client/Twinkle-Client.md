# Twinkle Client

Twinkle Client is the native client, designed with the philosophy: **Change `from twinkle import` to `from twinkle_client import`, and you can migrate local training code to remote calls without modifying the original training logic**.

## Initialization

```python
from twinkle import init_twinkle_client

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

Keep the data-processing and training-loop code, then create the remote model through `client.model(...)` after initializing the remote client:

```python
# Local training code (original)
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset
from twinkle.model import MultiLoraTransformersModel

# Remote training code (after migration)
# DataLoader and Dataset can be imported from either local twinkle or remote twinkle_client
from twinkle.dataloader import DataLoader        # or: from twinkle_client.dataloader import DataLoader
from twinkle.dataset import Dataset              # or: from twinkle_client.dataset import Dataset
from twinkle import init_twinkle_client

client = init_twinkle_client(base_url=base_url, api_key=api_key)
model = client.model(f'ms://{base_model}')
```

Training loops and data processing do not need any modifications. Prefer `client.model(...)` over constructing `MultiLoraTransformersModel(...)` directly so the model wrapper explicitly reuses the current client's transport, session, and authentication context.

## Complete Training Example (Transformers Backend)

```python
import dotenv
dotenv.load_dotenv('.env')

from peft import LoraConfig
from twinkle import get_logger
from twinkle.dataset import DatasetMeta
from twinkle import init_twinkle_client

# DataLoader and Dataset can be imported from either local twinkle or remote twinkle_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset

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

# Step 4: Create the remote model bound to the current client
model = client.model(f'ms://{base_model}')

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
# The server always uploads in the background; this call waits through the future layer
# until upload completion or failure. Do not pass async_upload or poll_interval: they
# remain only for old-call compatibility, are deprecated, and have no effect.
# YOUR_USER_NAME = "your_username"
# hub_model_id = f'{YOUR_USER_NAME}/twinkle-self-cognition'
# model.upload_to_hub(
#     checkpoint_dir=twinkle_path,
#     hub_model_id=hub_model_id,
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

(client-async-grpo)=
## Client-side Async GRPO

The client can train while sampling continues. After [installing Twinkle](../Installation.md), choose an example and follow the steps below:

| Task | Example |
|---|---|
| Single-turn math questions (GSM8K) | [client_orchestrated_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/client_orchestrated_grpo.py) |
| Multi-turn code repair with tools (text only, asyncio orchestration) | [multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/multi_turn_grpo.py) |
| Complete-episode ready streaming with an explicit synchronous training loop (StreamingDriver) | [streaming_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/streaming_multi_turn_grpo.py) |

### 1. Start the Server

The example uses Qwen3.5-4B with Transformers for training and vLLM for sampling, requiring two GPUs. Prepare a local model directory, then run these commands from the repository root:

```bash
pip install -e '.[async-rl,client,server]'
export TWINKLE_LOCAL_MODEL_PATH=/absolute/path/to/Qwen3.5-4B
export CUDA_VISIBLE_DEVICES=0,1
bash cookbook/client/async_rl/run_server.sh
```

The server listens on `http://127.0.0.1:8000` by default. Skip this step if you already have a suitable server; see [server_config.yaml](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/server_config.yaml) for its configuration.

### 2. Run the Math Example

Keep the server running and use another terminal, also at the repository root:

```bash
export TWINKLE_SERVER_URL=http://127.0.0.1:8000
export TWINKLE_SERVER_TOKEN=EMPTY_TOKEN
export TWINKLE_TEMPLATE_MODEL_ID=/absolute/path/to/Qwen3.5-4B
python cookbook/client/async_rl/client_orchestrated_grpo.py
```

The example downloads GSM8K by default. For offline use, also set `TWINKLE_DATASET_ID=/absolute/path/to/gsm8k`. When connecting to an existing server, use its address and authentication token. The template directory must match the server's model.

### 3. Run the Code Repair Example (Optional)

The multi-turn example requires an accessible AgentENV service and a working Python environment template. Keep the client settings above and run:

```bash
pip install 'e2b>=2.7'
export AGENTENV_API_URL=http://your-agentenv-gateway:8080
export AGENTENV_TEMPLATE=your-python-template
python cookbook/client/async_rl/multi_turn_grpo.py
```

The model reads files, runs tests and repairs code, then trains on the test results. Each attempt uses an independent workspace; the example handles initialization, scoring and cleanup. For environment setup, see [Agentic RL Deployment and Training](../Agentic-RL-Deployment-and-Training.md).

### 4. Run the Episode-streaming Code Example (Optional)

`streaming_multi_turn_grpo.py` reuses Blackjack from the existing multi-turn examples: the model calls `play(hit/stand)`, the environment returns observations and terminal rewards, and complete groups feed GRPO training and policy publication. OpenEnv/OpenSpiel runs inside the client process; no environment server or generated-code execution is needed.

Install the client environment dependencies, then use the Twinkle server and client settings above:

```bash
pip install 'openenv==0.11.0' 'open_spiel==2.0.2'
pip install 'openenv-openspiel-env @ git+https://github.com/huggingface/OpenEnv.git@9c6376390489f28e0e545cf74a277064e7edd52f#subdirectory=envs/openspiel_env'
python cookbook/client/async_rl/streaming_multi_turn_grpo.py
```

Group members use the same environment seed, independent environments and different model sampling seeds. Unfinished games and zero-advantage groups are skipped; a short run need not produce an update or increasing reward. `_train_loop` shows training explicitly, with one optimizer step per complete group. Top-level constants control group size, buffering and the update budget.

`rollout.streaming(driver)` closes streaming admission, signals cancellation, attempts to settle every call and then joins the driver. The instance accepts no further streaming admission after exit. Environments remain caller-owned. `rollout.is_idle` reads the SDK's existing execution/resource completion evidence; the example retains the transport and reports failure when execution is unconfirmed.

Sampling stays at `temperature=1`, `top_p=1`, `top_k=-1` and `repetition_penalty=1` to match processed logprobs to raw training logits. The shared `server_config.yaml` still deploys DataPlane and requires the server's `[async-rl]` dependencies.

### Use Your Own Task

- Math tasks: change the dataset and reward function in the script.
- Code tasks: edit `TASK` at the top of `multi_turn_grpo.py`.
- Streaming tool tasks: replace the Blackjack environment, tools and environment reward seam while keeping complete groups, advantages, training and publication.
- Start with the default training settings; edit the constants at the top of the script when needed. Training logs show rewards and metrics, and LoRA checkpoints are saved.

## Trainable Multi-turn Rollout (ClientMultiTurnRollout)

For a custom multi-turn training loop, use `twinkle_client.rollout.ClientMultiTurnRollout`. It supports text-only tool use and returns complete trajectories ready for training.

Inside an async training loop where the rollout, trajectories, tools and policy checkpoint are already initialized:

```python
rolled = await rollout.acall(
    trajectories,
    tool_manager=tool_managers,
    adapter_uri=policy_uri,
)
```

For synchronous use, call `rollout(...)` with the same arguments and result format. Give each trajectory the `ToolManager` for its independent environment; rollout does not close environments you pass in. See [multi_turn_rollout.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/multi_turn_rollout.py) for the full synchronous example.
