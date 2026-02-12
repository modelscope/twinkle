# Twinkle Client - Transformers LoRA Training Example
#
# This script demonstrates how to fine-tune a language model using LoRA
# (Low-Rank Adaptation) through the Twinkle client-server architecture.
# The server must be running first (see server.py and server_config.yaml).

# Step 1: Load environment variables from a .env file (e.g., API tokens)
import dotenv
dotenv.load_dotenv('.env')

import os
from peft import LoraConfig

from twinkle import get_logger
from twinkle.dataset import DatasetMeta
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client import init_twinkle_client

logger = get_logger()

# Whether to use Megatron for training
use_megatron = True
# Step 2: Initialize the Twinkle client to communicate with the remote server.
# - base_url: the address of the running Twinkle server
# - api_key: authentication token (loaded from environment variable)
client = init_twinkle_client(
    base_url='http://127.0.0.1:8000', api_key=os.environ.get('MODELSCOPE_SDK_TOKEN'))

# Step 3: Query the server for existing training runs and their checkpoints.
# This is useful for resuming a previous training session.
runs = client.list_training_runs()

resume_path = None
for run in runs:
    logger.info(run.model_dump_json(indent=2))
    # List all saved checkpoints for this training run
    checkpoints = client.list_checkpoints(run.training_run_id)

    for checkpoint in checkpoints:
        logger.info(checkpoint.model_dump_json(indent=2))
        # Uncomment the line below to resume from a specific checkpoint:
        # resume_path = checkpoint.twinkle_path


def train():
    # Step 4: Prepare the dataset

    # Load the self-cognition dataset from ModelScope
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))

    # Apply a chat template so the data matches the model's expected input format
    dataset.set_template(
        'Template', model_id='ms://Qwen/Qwen2.5-3B-Instruct', max_length=512)

    # Replace placeholder names in the dataset with custom model/author names
    dataset.map('SelfCognitionProcessor', init_args={
                'model_name': 'twinkle模型', 'model_author': 'ModelScope社区'})

    # Tokenize and encode the dataset into model-ready input features
    dataset.encode(batched=True)

    # Wrap the dataset into a DataLoader that yields batches of size 4
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    # Step 5: Configure the model

    # Create a multi-LoRA Transformers model pointing to the base model on ModelScope
    model = MultiLoraTransformersModel(
        model_id='ms://Qwen/Qwen2.5-3B-Instruct')

    # Define LoRA configuration: apply low-rank adapters to all linear layers
    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    # Attach the LoRA adapter named 'default' to the model.
    # gradient_accumulation_steps=2 means gradients are accumulated over 2 micro-batches
    # before an optimizer step, effectively doubling the batch size.
    model.add_adapter_to_model(
        'default', lora_config, gradient_accumulation_steps=2)

    # Set the same chat template used during data preprocessing
    model.set_template('Template')

    # Set the input processor (pads sequences on the right side)
    model.set_processor('InputProcessor', padding_side='right')

    # Use cross-entropy loss for language modeling
    model.set_loss('CrossEntropyLoss')

    # Use Adam optimizer with a learning rate of 1e-4 (Only support Adam optimizer if server use megatron)
    model.set_optimizer('Adam', lr=1e-4)

    # Use a linear learning rate scheduler (Do not support LR scheduler if server use megatron)
    if not use_megatron:
        model.set_lr_scheduler('LinearLR')

    # Step 6: Optionally resume from a previous checkpoint
    if resume_path:
        logger.info(f'Resuming training from {resume_path}')
        model.load(resume_path, load_optimizer=True)

    # Step 7: Run the training loop
    logger.info(model.get_train_configs())

    for epoch in range(3):
        logger.info(f'Starting epoch {epoch}')
        for step, batch in enumerate(dataloader):
            # Forward pass + backward pass (computes gradients)
            output = model.forward_backward(inputs=batch)

            # Log the loss every 2 steps (aligned with gradient accumulation)
            if step % 2 == 0:
                logger.info(f'Current is step {step // 2}, loss: {output}')

            # Clip gradients to prevent exploding gradients (max norm = 1.0)
            model.clip_grad_norm(1.0)

            # Perform one optimizer step (update model weights)
            model.step()

            # Reset gradients to zero for the next iteration
            model.zero_grad()

            # Advance the learning rate scheduler by one step
            model.lr_step()

        # Step 8: Save the trained checkpoint
        twinkle_path = model.save(name=f'twinkle-epoch-{epoch}', save_optimizer=True)
        logger.info(f"Saved checkpoint: {twinkle_path}")

    # Step 9: Upload the checkpoint to ModelScope Hub
    # YOUR_USER_NAME = "your_username"
    # hub_model_id = f'{YOUR_USER_NAME}/twinkle-self-cognition'
    # model.upload_to_hub(
    #     checkpoint_dir=twinkle_path, 
    #     hub_model_id=hub_model_id, 
    #     async_upload=False
    # )
    # logger.info(f"Uploaded checkpoint to hub: {hub_model_id}")


if __name__ == '__main__':
    train()
