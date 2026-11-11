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

`streaming_multi_turn_grpo.py` reuses Blackjack from the existing multi-turn examples: the model calls `play(hit/stand)`, the environment returns observations and terminal rewards, and complete groups feed GRPO training and policy publication. OpenEnv/OpenSpiel runs inside the client process; no environment server or generated-code execution is needed.

`ClientMultiTurnRollout` controls each model/tool conversation; `StreamingDriver` schedules concurrent conversations and complete training groups. This example uses no external Agent harness. Embedded OpenEnv, remote OpenEnvClient and AgentENV share the `Env` interface, but process/OS isolation depends on the environment backend and deployment.

Install the client environment dependencies, then use the Twinkle server and client settings above:

```bash
pip install 'openenv==0.11.0' 'open_spiel==2.0.2'
pip install 'openenv-openspiel-env @ git+https://github.com/huggingface/OpenEnv.git@9c6376390489f28e0e545cf74a277064e7edd52f#subdirectory=envs/openspiel_env'
python cookbook/client/async_rl/streaming_multi_turn_grpo.py
```

Before running, configure a Swift version compatible with this branch that provides `swift.dev.template_contract`, and use the same tokenizer/template as the server. Check template imports first; this command creates neither a model nor a sandbox:

```bash
python -c "from twinkle.template import Qwen3_5Template"
```

If the template contract is missing, configure the matching Swift environment first. The `[client,e2b]` extras supply client and E2B connectivity dependencies; they do not configure Swift.

Group members use the same environment seed, independent environments and different model sampling seeds. Unfinished games and zero-advantage groups are skipped; a short run need not produce an update or increasing reward. `_train_loop` shows training explicitly, with one optimizer step per complete group. Top-level constants control group size, buffering and the update budget.

Read `train()` for connection and setup, then `_train_loop()` for complete groups → advantages → `forward_backward` → `clip_grad_and_step` → publication request → `commit`. `commit(0)` records no update and consumes no optimizer budget; a publication request runs when the driver resumes. Inspect `_build_driver()` when adapting environments and rewards. The SDK's `RolloutResources` scope handles resource teardown; the recipe contains no cleanup exception handling.

`twinkle_client.rollout.training_groups(samples, num_generations=..., drop_truncated=True)` prepares scored streaming batches: it preserves member order, skips incomplete or mixed-policy groups, and checks membership, response token/mask/logprob alignment and finite rewards. Rollout failures are checked across the whole batch before any group is returned, so filtering or an update budget cannot hide them. `drop_truncated` defaults to `False`; Blackjack explicitly enables it to train terminal games only. Advantage calculation, zero-advantage decisions and updates remain visible in the recipe. The synchronous batch multi-turn example keeps its Trajectory-dictionary training interface and also uses `RolloutResources` for resource ownership.

Each `train()` invocation generates its own checkpoint name prefix automatically, avoiding overwritten `policy-0` snapshots across runs. No user configuration is required. This prefix only names checkpoints; it does not provide tenant isolation, control random seeds or resume a previous run.

`rollout.streaming(driver)` closes batch and streaming admission, signals cancellation, attempts to settle every call and then joins the driver. Use a dedicated rollout instance; it accepts no further work after exit. A bare rollout borrows environments without closing them. The example passes connection settings explicitly via `RolloutResources(init_twinkle_client, base_url=..., api_key=...)`, reading `TWINKLE_SERVER_URL` and `TWINKLE_SERVER_TOKEN` from `.env`. The scope transfers client, rollout and environment ownership to the SDK. Register its dedicated rollout with `own_rollout()` and environments with `own_environment()` before `reset()`, covering partial initialization too. After collecting an episode, `close_environment()` releases its environment early. On scope exit, the SDK waits for calls, drains rollout and closes remaining environments and the client, handling repeated interrupts while preserving the primary error. Unconfirmed execution retains resource handles and reports failure rather than claiming successful release.

`RolloutResources` automatically binds interrupt protection to this client's model operations, sampler creation and configuration. The recipe calls `model.forward_backward(...)`, `model.clip_grad_and_step()` and `model.save(...)` directly. If the waiting thread is interrupted, the SDK waits for the issued call to return, registers training handles and successful results, then propagates the interrupt. A failed or interrupted managed call stops subsequent managed calls; training operations are never automatically resubmitted. `model.completed_step_calls` counts successful step/clip_grad_and_step replies on that model object, not proof of device parameter changes. `model.last_checkpoint` retains the latest successful save response. Sampling remains concurrent under rollout execution and bypasses the training-operation sequence. Clients outside the scope retain their existing behavior. Timeouts, lost connections and failed task records do not prove physical completion: unconfirmed resources are retained and reported, never automatically reused. This does not provide remote force-cancellation.

Sampling stays at `temperature=1`, `top_p=1`, `top_k=-1` and `repetition_penalty=1` to match processed logprobs to raw training logits. The shared `server_config.yaml` still deploys DataPlane and requires the server's `[async-rl]` dependencies.

### 4. Run the Same Task in Official E2B Sandboxes (Optional)

`e2b_multi_turn_grpo.py` connects `AgentEnv` to E2B's official service, with a separate sandbox for each trajectory. Game history, card draws and rewards execute inside the sandbox. The model still uses only `play(hit/stand)`, orchestrated by the same `ClientMultiTurnRollout` and `StreamingDriver`. It reuses the previous example's `train(env_factory=...)`; there is no additional training framework, verifier or external Agent harness.

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

The default E2B `base` template installs `open_spiel==2.0.2` in the writable `/home/user/blackjack` directory and receives the adjacent `blackjack_game.py` file. OpenEnv/OpenSpiel is not needed on the client. Set `E2B_TEMPLATE` to a custom template with that OpenSpiel version preinstalled to reduce startup time. The default buffer holds up to 12 trajectories; match it to your account's sandbox concurrency quota.

Read the three files by responsibility:

Run the entry point directly: it constructs rollout and driver for you. Inspect the `submit/collect` callbacks in `_build_driver()` when adapting the task; initial setup needs no manual component assembly. Environments supply `system_prompt` and the standard `Env.tools()` definitions.

| File | Responsibility |
| --- | --- |
| `streaming_multi_turn_grpo.py` | `train()` connects services and constructs rollout/driver; `_train_loop()` shows advantages, backward, optimizer step and policy publication. Reads task instructions and `tools()` from the environment rather than defining Blackjack rules. |
| `e2b_multi_turn_grpo.py` | `E2BBlackjackEnv` implements the standard `reset/step/close` interface over E2B, then selects it with `train(env_factory=E2BBlackjackEnv)`. |
| `blackjack_game.py` | Shared Blackjack task definition (`BlackjackTask.system_prompt` / `tools()`) and the OpenSpiel program uploaded to the sandbox. It stores seed/action history in the sandbox's `game.json` and returns cards, reward and terminal state. It performs no training. |

The flow is: driver admits separate games → rollout calls the remote sampler → the model calls `play` → AgentEnv executes the game in E2B → complete groups supply advantages and training → newly published URIs serve subsequent games. Group members share a seed but own separate environments; each trajectory keeps one fixed policy URI.

`env_factory` switches the same Blackjack task between local and E2B execution; both backends share instructions, the `play` tool and reward rules. To introduce another task, define its environment instructions, tools and rewards, then adapt `submit/collect` in `_build_driver()` while retaining the training loop. This recipe requires exactly one `play` tool and rejects empty, additional or differently named tools instead of silently dropping them. The SDK's `ClientMultiTurnRollout` / `ToolManager` still support multiple tools.

Start by changing only connection settings. To adjust scale, edit `NUM_GENERATIONS`, `BUFFER_GROUPS`, `MAX_GROUPS` and `MAX_STEPS` in the shared training script: group size, buffered groups, finite group attempts and maximum updates. Each pull requests one complete group. Equal-reward groups are skipped, so the update count is an upper bound. Smaller buffers retain fewer sandboxes but allow less sampling/training overlap.

Each collected trajectory releases its sandbox immediately. On exit, rollout stops and drains before remaining environments close. Cancellation is cooperative and waits for issued commands to return. Unconfirmed execution retains its handle and raises an error, with sandbox expiry after 600 seconds as a fallback; immediate remote abort is not provided. This example executes trusted game tools, not model-generated code.

Game-process failures or malformed game responses abort training rather than becoming zero rewards or ordinary incomplete episodes. Invalid model tool arguments return a corrective observation instead.

| Symptom | Check first |
| --- | --- |
| E2B authentication or template error | `E2B_API_KEY`, `E2B_TEMPLATE` and endpoint overrides. Rerun the standalone connectivity check; the default official template is `base`. |
| Sandbox quota exceeded | Reduce `BUFFER_GROUPS` and check account quotas; the default budget is `3 × 4 = 12` trajectories. |
| Slow environment startup | Each trajectory creates a sandbox and checks/installs OpenSpiel. Use an `E2B_TEMPLATE` with dependencies preinstalled and a compatible writable directory. |
| E2B check passes but training connection fails | Check `TWINKLE_SERVER_URL/TOKEN`, server model and local template directory. E2B connectivity does not establish training-server readiness. |
| Games run without optimizer updates | Inspect `zero advantage` / `truncated` logs; legitimate skips within a finite attempt budget are separate from connection failures. |

### Use Your Own Task

- Math tasks: change the dataset and reward function in the script.
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
