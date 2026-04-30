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

This matches the end-to-end example in `cookbook/client/twinkle/self_host/self_cognition.py`.

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
