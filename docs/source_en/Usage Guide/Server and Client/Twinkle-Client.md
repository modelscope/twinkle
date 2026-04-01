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
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
```

Training loops, data processing, and other logic do not need any modifications.

## Complete Training Example (Transformers Backend)

```python
import os
import dotenv
dotenv.load_dotenv('.env')

from peft import LoraConfig
from twinkle import get_logger
from twinkle.dataset import DatasetMeta

# Import from twinkle_client instead of twinkle to enable remote calls
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client import init_twinkle_client

logger = get_logger()

# Step 1: Initialize client
client = init_twinkle_client(
    base_url='http://127.0.0.1:8000',
    api_key=os.environ.get('MODELSCOPE_TOKEN')
)

# Step 2: Query existing training runs (optional, for resuming training)
runs = client.list_training_runs()
resume_path = None
for run in runs:
    checkpoints = client.list_checkpoints(run.training_run_id)
    for checkpoint in checkpoints:
        logger.info(checkpoint.model_dump_json(indent=2))
        # Uncomment to resume from checkpoint:
        # resume_path = checkpoint.twinkle_path

# Step 3: Prepare dataset
dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition'))

# Set chat template to match model's input format
dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)

# Data preprocessing: Replace placeholders with custom names
dataset.map('SelfCognitionProcessor',
            init_args={'model_name': 'twinkle model', 'model_author': 'ModelScope Team'})

# Encode dataset into tokens usable by the model
dataset.encode(batched=True)

# Create DataLoader
dataloader = DataLoader(dataset=dataset, batch_size=8)

# Step 4: Configure model
model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

# Configure LoRA
lora_config = LoraConfig(target_modules='all-linear')
model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)

# Set template, processor, loss function
model.set_template('Template')
model.set_processor('InputProcessor', padding_side='right')
model.set_loss('CrossEntropyLoss')

# Set optimizer and learning rate scheduler
model.set_optimizer('AdamW', lr=1e-4)
model.set_lr_scheduler('LinearLR')

# Step 5: Resume training (optional)
consumed_train_samples = 0
global_step = 0
if resume_path:
    logger.info(f'Resuming model weights from {resume_path}')
    model.load(resume_path)
    trainer_state = model.load_training_state(resume_path)
    dataloader.skip_consumed_samples(trainer_state['consumed_train_samples'])
    consumed_train_samples = int(trainer_state['consumed_train_samples'])
    global_step = int(trainer_state['cur_step'])

# Step 6: Training loop
for _, batch in enumerate(dataloader):
    # Forward propagation + backward propagation
    model.forward_backward(inputs=batch)

    # Step
    model.clip_grad_and_step()
    consumed_train_samples += len(batch)
    global_step += 1

    if global_step % 2 == 0:
        metric = model.calculate_metric(is_training=True)
        logger.info(f'Current is step {global_step} of {len(dataloader)}, metric: {metric.result}')

# Step 7: Save checkpoint
twinkle_path = model.save(
    name=f'step-{global_step}',
    save_optimizer=True,
    consumed_train_samples=consumed_train_samples,
)
logger.info(f"Saved checkpoint: {twinkle_path}")

# Step 8: Upload to ModelScope Hub (optional)
model.upload_to_hub(
    checkpoint_dir=twinkle_path,
    hub_model_id='your-username/your-model-name',
    async_upload=False
)
```

For checkpoint resumption, the recommended client-side flow is:

1. Query the server for an existing checkpoint path with `client.list_checkpoints(...)` or `client.get_latest_checkpoint_path(...)`.
2. Call `model.load(resume_path)` to restore adapter weights.
3. Call `model.load_training_state(resume_path)` to restore optimizer, scheduler, RNG, and progress metadata.
4. Call `dataloader.skip_consumed_samples(...)` with `consumed_train_samples` from the returned trainer state.

This matches the end-to-end example in `cookbook/client/twinkle/self_host/self_congnition.py`.

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
