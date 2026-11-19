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
| Blackjack with tools (StreamingDriver, explicit synchronous training loop) | [streaming_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/streaming_multi_turn_grpo.py) |
| Blackjack in E2B sandboxes (same training loop) | [e2b_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/e2b_multi_turn_grpo.py) |

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

### 3. Run the Streaming Blackjack Example (Optional)

`streaming_multi_turn_grpo.py` demonstrates client-side multi-turn async GRPO with Blackjack. The model calls `play(hit/stand)`, the environment returns cards and terminal rewards, and complete groups feed training. The local environment calls the OpenSpiel game in `blackjack_game.py` directly; no environment server or OpenEnv adapter package is required.

The client runs `StreamingDriver` and `MultiTurnRollout`: the Driver schedules episodes and delivers complete groups, the Rollout manages conversations and environment sessions, and `HTTPSampler` sends each generation request to the server. Model training calls also go through the client API to the server.

Use the server and client configuration above, install the local game dependency, then run:

```bash
pip install 'open_spiel==2.0.2'
python cookbook/client/async_rl/streaming_multi_turn_grpo.py
```

Configure the matching Swift dependency providing `swift.dev.template_contract` and the same tokenizer/template as the server. Check imports first:

```bash
python -c "from twinkle.template import Qwen3_5Template"
```

Group members share an environment seed but keep independent action histories; model sampling does not use a fixed seed. The environment implements the session-based `_open/_step` hooks, and Rollout calls `reset/step/close`. It reads the game outcome before closing the session and stores accumulated reward in `rewards.env`. An unfinished or truncated member causes its group to be dropped before delivery.

Read `train()` for assembly, then `_train_loop()`: receive `rows, facts` from the Driver → map environment rewards to named scores → compute `advantages/old_logps` with `GRPOAdvantage` → `forward_backward` → `clip_grad_and_step` → `ClientCheckpointManager.sync_weights()`. Equal-reward groups skip the optimizer update; a short run need not produce updates or reward improvement.

This recipe publishes once per consumed batch, including zero-advantage batches, to advance the Driver's admission budget. Publication versions and optimizer updates are separate counters: `MAX_STEPS` limits confirmed optimizer updates, whereas the Driver's `max_steps` would limit delivered batches and is left unset here. Every run generates a unique checkpoint-name prefix.

`ClientCheckpointManager` saves the server-side LoRA and switches the default URI on `HTTPSampler`. An issued request keeps its original URI; subsequent requests read the new URI. A multi-turn episode can span publications, and Rollout is not given a fixed `adapter_path`.

On exit, the recipe closes the Driver, waits for issued step/stop Futures through `_RolloutTasks`, then lets `RolloutResources` close Rollout, environment components and the client. The helper only tracks pending calls; the shared Rollout still owns the interaction logic. Model training and save calls retain the SDK's interrupt protection; `model.completed_step_calls` counts acknowledged updates. Waiting for local calls is not remote force-cancellation.

Sampling uses `temperature=1`, `top_p=1`, `top_k=-1` and `repetition_penalty=1` to align sampling logprobs with training probabilities. The shared `server_config.yaml` still deploys DataPlane and requires the server's `[async-rl]` dependencies.

### 4. Run the Same Task in Official E2B Sandboxes (Optional)

`e2b_multi_turn_grpo.py` connects `AgentEnv` to E2B's official service, with a separate sandbox session for each active episode. Game history, card draws and rewards execute inside the sandbox. The model still uses only `play(hit/stand)`, orchestrated by the shared `MultiTurnRollout`, `HTTPSampler` and `StreamingDriver`. It reuses the previous example's `train(env_factory=...)`; there is no additional training framework, verifier or external Agent harness.

First run the [AgentEnv official E2B connectivity check](../../Components/Agentic/Envs.md). It checks creation, command execution and release without a training server, separating key/template/network issues from training setup.

Keep the Twinkle server from the previous steps running. Configure both services in `.env` at the repository root. `TWINKLE_SERVER_TOKEN` authenticates training/sampling; `E2B_API_KEY` authenticates cloud sandboxes:

```dotenv
TWINKLE_SERVER_URL=http://127.0.0.1:8000
TWINKLE_SERVER_TOKEN=EMPTY_TOKEN
TWINKLE_MODEL_ID=Qwen/Qwen3.5-4B
E2B_API_KEY=your_e2b_api_key
# Offline clients also need a local template directory matching the server model:
# TWINKLE_TEMPLATE_MODEL_ID=/absolute/path/to/Qwen3.5-4B
```

Replace the URL, token and key with your own settings and keep `.env` out of Git. For the official service, remove `E2B_API_URL`, `E2B_SANDBOX_URL` and `E2B_DOMAIN` overrides from `.env` and the shell environment. Then run:

```bash
pip install -e '.[client,e2b]'
python cookbook/client/async_rl/e2b_multi_turn_grpo.py
```

The default E2B `base` template installs `open_spiel==2.0.2` in the writable `/home/user/blackjack` directory and receives the adjacent `blackjack_game.py` file. OpenEnv/OpenSpiel is not needed on the client. Set `E2B_TEMPLATE` to a custom template with that OpenSpiel version preinstalled to reduce startup time. The environment allows at most `(MAX_STALENESS + 1) × NUM_GENERATIONS = 8` active sessions by default; match this to your account quota.

Read the three files by responsibility:

Run the entry point directly; it constructs rollout and driver. `BlackjackTask` supplies instructions and tools. Task adaptation changes the environment session hooks, prompt inputs and reward mapping.

| File | Responsibility |
| --- | --- |
| `streaming_multi_turn_grpo.py` | `train()` connects services and constructs rollout/driver; `_train_loop()` shows advantages, backward, optimizer step and policy publication. Assembles task instructions, environment sessions and tools; OpenSpiel supplies the game rules. |
| `e2b_multi_turn_grpo.py` | `E2BBlackjackEnv` implements session-based `_open/_step` hooks, with sandbox ownership handled by `AgentEnv`, then selects it with `train(env_factory=E2BBlackjackEnv)`. |
| `blackjack_game.py` | Shared Blackjack task definition (`BlackjackTask.system_prompt` / `tools()`) and the OpenSpiel program uploaded to the sandbox. It stores seed/action history in the sandbox's `game.json` and returns cards, reward and terminal state. It performs no training. |

The flow is: driver admits separate games → rollout calls the remote sampler → the model calls `play` → AgentEnv executes the game in E2B → complete groups supply advantages and training → newly published URIs serve subsequent generation requests. Group members share a seed but own separate sessions; later turns of an episode may use a new publication.

`env_factory(max_sessions=...)` creates one environment component per training run, with independent sessions for its episodes. Both backends share instructions, the `play` tool, terminal checks and rewards. For a new task, implement environment hooks and adapt the prompt inputs in `_build_driver()` and scores in `_train_loop()`. The shared `MultiTurnRollout` supports multiple tools.

Scale is controlled by `NUM_GENERATIONS`, `MAX_STALENESS`, `MAX_GROUPS` and `MAX_STEPS`: episodes per group, allowed published-version lag, finite task groups and maximum actual updates. Each batch delivers one complete group and triggers one publication. Reducing `MAX_STALENESS` shrinks the initial sampling window and session limit, with less future work available for overlap.

A completed episode closes its session. `AgentEnv` clears the workspace and retains a healthy sandbox for reuse. On exit, the Driver stops and issued calls settle before Rollout, active sessions and idle sandboxes close. The example executes fixed game tools and provides no immediate remote force-cancellation.

Game-process failures or malformed game responses abort training rather than becoming zero rewards or ordinary incomplete episodes. Invalid model tool arguments return a corrective observation instead.

| Symptom | Check first |
| --- | --- |
| E2B authentication or template error | `E2B_API_KEY`, `E2B_TEMPLATE` and endpoint overrides. Rerun the standalone connectivity check; the default official template is `base`. |
| Sandbox quota exceeded | Reduce `MAX_STALENESS` or `NUM_GENERATIONS` and check quotas; the default session limit is `2 × 4 = 8`. |
| Slow environment startup | Each new session creates or reuses a sandbox and checks/installs OpenSpiel. Use an `E2B_TEMPLATE` with dependencies preinstalled and a compatible writable directory. |
| E2B check passes but training connection fails | Check `TWINKLE_SERVER_URL/TOKEN`, server model and local template directory. E2B connectivity does not establish training-server readiness. |
| Games run without optimizer updates | Inspect `zero advantage` logs and the Driver's `dropped_sample/dropped_group` counters; legitimate skips within a finite attempt budget are separate from connection failures. |

### Use Your Own Task

- Math tasks: change the dataset and reward function in the script.
- Streaming tool tasks: replace the Blackjack environment, tools and environment reward seam while keeping complete groups, advantages, training and publication.
- Start with the default training settings; edit the constants at the top of the script when needed. Training logs show rewards and metrics, and LoRA checkpoints are saved.

## Client-side Multi-turn Rollout

Current CS training composes `twinkle_agentic.rollout.MultiTurnRollout` with `twinkle_client.rollout.HTTPSampler` and publishes weights through `ClientCheckpointManager`. The former `ClientMultiTurnRollout` and `training_groups` APIs have been removed.

For custom multi-turn async training, start from [streaming_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/streaming_multi_turn_grpo.py): retain the Driver and HTTP connections, then replace task inputs, environment and rewards. `MultiTurnRollout.step()` advances one interaction turn; the Driver delivers complete groups for training.
