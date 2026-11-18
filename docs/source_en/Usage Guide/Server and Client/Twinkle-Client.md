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

## Client Training Learning Path

Use a fixed set of held-out questions to compare the model before and after training. Keep training examples, scoring, and evaluation separate so you can diagnose changes when bringing your own task.

| Stage | Exercise | What to learn |
|---|---|---|
| 1. SFT | [Self-cognition fine-tuning](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/self_cognition.py) | Predict the next token from a reference response. Inspect which tokens carry labels and which parameters LoRA updates. |
| 2. DPO (optional) | [Preference fine-tuning](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/dpo.py) | Use a preferred and a rejected answer to the same prompt to teach relative preferences. |
| 3. GRPO | [Short-answer math](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/short_math_grpo.py) | Sample multiple answers, reward correctness and brevity, and turn within-group differences into advantages. |
| 4. Online distillation | [The multiplication exercise below](#client-online-distillation) | Compare vocabulary distributions, token advantages, and self-distillation on the student's own replies. |
| 5. Agentic RL | [Blackjack below](#client-async-grpo) | Follow a multi-turn trajectory through tools, observations, environment rewards, and overlapping sampling and training. |

To bring your own task, replace the questions and targets, then write a reward or worked solution you can verify independently. For tool training, inspect one complete episode before increasing concurrency. Use held-out tasks to measure accuracy, format, tool success, and output cost. In your own project, use SFT to teach the response format or tool calls first. Start reward training from that checkpoint and retain evaluation results for each stage.

(client-online-distillation)=
## Online Distillation

[One client script](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/online_distillation.py) trains Qwen3.5-0.8B on short multiplication with thinking disabled. Read `train()` for sampling, updating, publishing, and evaluation. Then read `distillation_signal()` to see how the methods score the same sampled replies.

| Method | Teacher signal | Server devices |
|---|---|---|
| OPSD | The current student sees a worked solution and scores its original reply token by token. | Two GPUs for Transformers and vLLM |
| OPD-RL | Frozen Qwen3.5-2B scores the original reply. The teacher logprob minus the rollout logprob becomes the token advantage. | One additional teacher GPU |
| GKD | The same frozen teacher returns a full vocabulary distribution. One sample per step limits the logits sent over HTTP. | Same as OPD-RL |

Run from a Twinkle source checkout with this example and GKD HTTP support. On CUDA, install the server with `python -m pip install -e '.[client,rl,server]'`. A separate client environment can use `python -m pip install -e '.[client]'`. Both environments need the Swift dev template API. Set up the device-specific PyTorch and vLLM runtime first on PPU.

```bash
python -m pip install 'ms-swift @ git+https://github.com/modelscope/ms-swift.git@e1ac89d24f47cb8fd73a7245dd79df0959aec284' 'qwen-vl-utils>=0.0.14'
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir ./models/Qwen3.5-0.8B
export TWINKLE_DISTILL_STUDENT_PATH="$PWD/models/Qwen3.5-0.8B"
CUDA_VISIBLE_DEVICES=0,1 twinkle-server launch -c cookbook/client/server/transformer/distillation.yaml
```

Wait for vLLM warm-up and the server message `All applications started`. In a second terminal, run:

```bash
export TWINKLE_DISTILL_STUDENT_PATH="$PWD/models/Qwen3.5-0.8B"
python cookbook/client/twinkle/online_distillation.py --method opsd --steps 4
```

For OPD-RL or GKD, prepare `Qwen/Qwen3.5-2B` in both environments and set `TWINKLE_DISTILL_TEACHER_PATH` to its local directory. Launch the same [configuration](https://github.com/modelscope/twinkle/blob/main/cookbook/client/server/transformer/distillation.yaml) with `CUDA_VISIBLE_DEVICES=0,1,2` and use `--method opd-rl` or `--method gkd`. For a remote server, use the server URL and token described above. If the student directories differ, set `TWINKLE_DISTILL_SERVER_STUDENT_PATH` in the client to the server-side directory.

Compare accuracy and output length on the same held-out questions in `eval=before` and `eval=after`. Answers must match the single-integer format exactly, so the score also measures format compliance. Four steps demonstrate the flow. For learning comparisons, match the held-out questions, number of training samples, and sampling settings, then repeat with different seeds.

(client-async-grpo)=
## Client-side Async GRPO

The client can train while sampling continues. After [installing Twinkle](../Installation.md), choose an example and follow the steps below:

| Task | Example |
|---|---|
| Single-turn math questions (GSM8K) | [client_orchestrated_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/client_orchestrated_grpo.py) |
| Blackjack with tools (StreamingDriver, explicit synchronous training loop) | [streaming_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/streaming_multi_turn_grpo.py) |
| Blackjack in E2B sandboxes (same training loop) | [e2b_multi_turn_grpo.py](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/e2b_multi_turn_grpo.py) |

### Connect and Run the Math Example

Ask your administrator for a Qwen3.5-4B service with GRPO and sampling support. The asynchronous math example also uses the server's DataPlane. [Server YAML](https://github.com/modelscope/twinkle/blob/main/cookbook/client/async_rl/server_config.yaml)

Set the client connection and run from the repository root:

```bash
export TWINKLE_SERVER_URL=https://your-twinkle-server
export TWINKLE_SERVER_TOKEN=your-token
python cookbook/client/async_rl/client_orchestrated_grpo.py
```

The example uses GSM8K. Offline clients can set `TWINKLE_DATASET_ID` to a local dataset and `TWINKLE_TEMPLATE_MODEL_ID` to tokenizer and template assets that match the server model.

### Run Multi-turn Blackjack GRPO

The model calls `play(hit/stand)` and the environment returns the visible cards and final reward. `MultiTurnRollout` records the sampled tokens and log probabilities for each turn, and `HTTPSampler` sends sampling requests to the Twinkle server. The synchronous recipe gathers a group before training. The streaming recipe delegates group scheduling to `StreamingDriver`.

Use the server URL and token from the section above. The default model is `Qwen/Qwen3.5-4B`. Set `TWINKLE_MODEL_ID` to select a model from the server configuration. Use the same tokenizer and template as the server. When the service uses an alias or the client is offline, set `TWINKLE_TEMPLATE_MODEL_ID` to the matching model ID or a local template directory.

Install the shared client dependencies from the repository root. The E2B recipe also needs this step.

```bash
python -m pip install -e '.[client]' python-dotenv \
  'ms-swift @ git+https://github.com/modelscope/ms-swift.git@e1ac89d24f47cb8fd73a7245dd79df0959aec284'
```

Install the local game environment for the synchronous and OpenEnv streaming recipes.

```bash
pip install 'openenv==0.11.0' 'open_spiel==2.0.2'
pip install 'openenv-openspiel-env @ git+https://github.com/huggingface/OpenEnv.git@9c6376390489f28e0e545cf74a277064e7edd52f#subdirectory=envs/openspiel_env'

# Choose one from the repository root.
python -m cookbook.client.twinkle.multi_turn_rollout
python cookbook/client/async_rl/streaming_multi_turn_grpo.py
```

Games in one group use the same deal seed and keep independent state. Training uses completed groups and skips groups with equal rewards. Read `train_batch()` to see advantage estimation, backward, the optimizer step and LoRA publication. At exit, check `completed_groups`, `optimizer_updates`, and `latest_lora`. With zero updates, the URI points to the initial policy.

### Run the Same Task in E2B Sandboxes

After installing the shared client dependencies above, set `E2B_API_KEY` in `.env`. Leave endpoint overrides unset to use the official E2B service.

```bash
pip install -e '.[client,e2b]'
python cookbook/client/async_rl/e2b_multi_turn_grpo.py
```

This entry point reuses the streaming loop and opens a sandbox for each episode. The sandbox runs the uploaded game program and computes the deal and reward through OpenSpiel. The default `base` template installs `open_spiel==2.0.2` in the sandbox. Set `E2B_TEMPLATE` to a template with that version preinstalled if you want faster startup. This entry point needs no local OpenSpiel installation.

### Use Your Own Tool Task

Use `BlackjackEnv` or `E2BBlackjackEnv` as a starting point. Define the tools, opening observation and rewards on the environment, then pass the environment into `MultiTurnRollout`. Call `rollout(trajectories)` for a synchronous batch. For streaming, pass the same rollout to `StreamingDriver` and iterate over `(rows, facts)`. Both entry points use `ClientCheckpointManager` to publish policies and `RolloutResources` to drain the rollout, close the environment and release training handles before closing the client.
