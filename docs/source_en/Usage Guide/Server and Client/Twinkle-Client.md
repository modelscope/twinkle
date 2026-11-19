# Twinkle Client

Use Twinkle Client to define training data, objectives, and evaluation. Twinkle Server runs the models, updates weights, and saves checkpoints. Connect with the endpoint, token, and model name advertised by the server. The client can run on CPU.

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

Use `client.model(...)` to bind the model to the current session and credentials. Data processing and training loops can reuse operations supported by the model service. Check its loss, optimizer, scheduler, and resume support before migrating.

## SFT Training Example

This example requires a Qwen3.5-4B service with LoRA, `CrossEntropyLoss`, and `Adam`.

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

# Set the optimizer for this exercise
model.set_optimizer('Adam', lr=1e-4)

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
# The SDK shows progress while waiting and returns the repository ID and commit revision.
# YOUR_USER_NAME = "your_username"
# hub_model_id = f'{YOUR_USER_NAME}/twinkle-self-cognition'
# upload = model.upload_to_hub(
#     checkpoint_dir=twinkle_path,
#     hub_model_id=hub_model_id,
# )
# logger.info(f'Uploaded {upload.repo_id} at {upload.revision}')
```

Pass an `on_progress` callback to display transferred bytes, total bytes, committed bytes, and commit status. ModelScope reports updates as files finish transferring or batches are committed. Deduplication can make transferred bytes lower than committed bytes. Install the server with the `server` extra to include the upload dependency.

For checkpoint resumption, the recommended client-side flow is:

1. Query the server for an existing checkpoint path with `client.list_checkpoints(...)` or `client.get_latest_checkpoint_path(...)`.
2. Call `model.resume_from_checkpoint(resume_path)` to restore weights, optimizer, scheduler, RNG, and progress metadata.
3. Call `dataloader.resume_from_checkpoint(progress['consumed_train_samples'])` to skip already-consumed samples.

This matches the end-to-end example in `cookbook/client/twinkle/self_cognition.py`.

## Model Services and Training Capabilities

`client.model(...)` uses the same HTTP interface across deployments. The model name routes the request. The service configuration controls how the model runs. Use `client.get_server_capabilities().supported_models` to discover model names. The currently advertised protocol features and limits are service-wide. Check the model service's documentation for its supported losses, optimizers, and resumption modes.

Use `forward_backward` for training and `forward_only` for read-only scoring. Some services fuse training forward and backward, so the separate training calls are unavailable. Optimizer and scheduler options also vary. Confirm the required capabilities before running an example.

## Client Training Learning Path

Start each lesson with a question about training, then inspect the data that determines the update. Record a baseline and evaluate the same held-out examples after training. Agentic-RL-Lab's OPSD chapter shows this structure with an explanation, reproducible commands, checkpoint comparisons, and limitations. [Agentic-RL-Lab OPSD](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/readme.md)

| Stage | Exercise and model | Training question | First check |
|---|---|---|---|
| 1. SFT | [Self-cognition fine-tuning](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/self_cognition.py), Qwen3.5-4B | Which tokens learn from the reference answer? | Check assistant-response labels and compare responses to the same questions. |
| 2. DPO (optional) | [Preference fine-tuning](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/dpo.py), Qwen3.5-4B | How does a preferred answer gain probability relative to a rejected answer? | Inspect a preference pair, then evaluate held-out pairs. |
| 3. GRPO | [Short-answer math](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/short_math_grpo.py), Qwen3.5-4B | How does an outcome reward guide generated answers? | Compare scores and advantages for one prompt group, then check accuracy and length. |
| 4. Online distillation | [The multiplication exercise below](#client-online-distillation), Qwen3.5-0.8B student with Qwen3.5-2B for the external teacher methods | Who is the teacher, what does it see, and how do its scores update the student? | Verify that teacher and student scores refer to the same reply tokens. |
| 5. Agentic RL | [Blackjack below](#client-async-grpo), Qwen3.5-4B | Where do rewards and labels go in a tool trajectory? | Read one episode with actions, observations, and a terminal reward, then measure completion and reward. |

To connect these exercises on your own task, use SFT to teach response format or tool use. Start reward training from that checkpoint and keep the same held-out tasks. Verify each new reward or tool outcome before changing the training algorithm. Each example can also run independently.

(client-online-distillation)=
## Online Distillation

[One client script](https://github.com/modelscope/twinkle/blob/main/cookbook/client/twinkle/online_distillation.py) teaches all three methods on short multiplication. For `17 × 23`, the student answers first. GKD asks an external teacher for every candidate token's score at each position. OPD-RL compares scores for the sampled tokens. OPSD gives the current student the worked solution and has it score its own original reply. Read `train()` for the shared loop and `distillation_signal()` for the three signals.

Ask your service administrator for its URL, token, and model services. The example uses `distill-student` (Qwen3.5-0.8B). OPD-RL and GKD also use `distill-teacher` (frozen Qwen3.5-2B). GKD requires HTTP support for `teacher_logits`. The service administrator manages model weights, devices, and the training backend. [Server YAML](https://github.com/modelscope/twinkle/blob/main/cookbook/client/server/transformer/distillation.yaml)

Install the client from the checkout that includes this example. Use the URL, token, and model name from your administrator:

```bash
python -m pip install -e '.[client]'
export TWINKLE_SERVER_URL=https://your-twinkle-server
export TWINKLE_SERVER_TOKEN=your-token
python cookbook/client/twinkle/online_distillation.py --method opsd --model distill-student --steps 4
```

OPSD sends the exact sampled reply tokens with the worked solution and requests scores from the training model. The reference YAML pins the same template for sampling and training. The CPU client evaluates the assistant text returned by the server.

OPD-RL and GKD still use local student and teacher template assets and a compatible Swift revision. In addition to the connection settings above, set:

```bash
python -m pip install 'ms-swift @ git+https://github.com/modelscope/ms-swift.git@e1ac89d24f47cb8fd73a7245dd79df0959aec284' 'qwen-vl-utils>=0.0.14'
export TWINKLE_DISTILL_STUDENT_PATH=/path/to/student-template-assets
export TWINKLE_DISTILL_TEACHER_PATH=/path/to/teacher-template-assets
export TWINKLE_DISTILL_SERVER_STUDENT_PATH=/server/path/to/student-model
python cookbook/client/twinkle/online_distillation.py --method opd-rl --model distill-student --steps 4
```

Switch to `--method gkd` to run GKD. If the model service requires a larger training batch, set `--batch-size` to the supported size. GKD transfers full vocabulary scores, so a larger batch sends more data. Compare accuracy and output length on the same held-out questions in `eval=before` and `eval=after`. Four steps demonstrate the flow. Match the training example count and evaluation settings and use multiple seeds to compare learning.

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
