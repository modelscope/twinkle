# Twinkle Client - DPO (Direct Preference Optimization) Training with LoRA
#
# This script demonstrates how to fine-tune a language model using DPO
# through the Twinkle client-server architecture.
# The server must be running first (see server.py and server_config.yaml).

# Step 1: Load environment variables from a .env file (e.g., API tokens)
import dotenv
import os
from typing import Any, Dict, List

dotenv.load_dotenv('.env')
import numpy as np
import torch
from peft import LoraConfig

from twinkle import get_logger
from twinkle.dataset import Dataset, DatasetMeta
from twinkle_client import init_twinkle_client
from twinkle.dataloader import DataLoader
from twinkle_client.model import MultiLoraTransformersModel
from twinkle.preprocessor import EmojiDPOProcessor

logger = get_logger()

# Configuration (direct values, not from env)
base_model = 'Qwen/Qwen3.6-35B-A3B'
base_url = 'http://www.modelscope.cn/twinkle'
dataset_id = 'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji'

batch_size = 4
gradient_accumulation_steps = 2
learning_rate = 1e-4
dpo_beta = 0.1
sft_weight = 1.0
loss_type = 'sigmoid'
max_length = 2048
adapter_name = 'default'
system_prompt = 'You are a helpful assistant.'

# Step 2: Initialize the Twinkle client to communicate with the remote server.
# - base_url: the address of the running Twinkle server
# - api_key: authentication token (loaded from environment variable)
client = init_twinkle_client(base_url=base_url, api_key=os.environ.get('MODELSCOPE_TOKEN'))

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


def create_dpo_dataset():
    """Create DPO dataset with positive/negative format."""
    dataset = Dataset(DatasetMeta(dataset_id, data_slice=range(100)))
    dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=max_length)
    dataset.map(
        EmojiDPOProcessor,
        init_args={
            'system': system_prompt,
        }
    )
    # DPO preprocessor returns {'positive': [...], 'negative': [...]}
    # batch_encode handles this format automatically
    dataset.encode()
    return dataset


def prepare_dpo_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare DPO batch: reorganize batch for training with DP-safe interleaving.

    Args:
        batch: List of rows, each with 'positive' and 'negative' InputFeatures
               and other fields (question, etc.)

    Returns:
        List interleaved as [pos_1, neg_1, pos_2, neg_2, ...] to ensure each DP
        worker gets complete positive/negative pairs after slicing.
        Each item contains all original fields plus the InputFeature fields.
    """
    result = []

    for row in batch:
        # Get base fields (excluding positive/negative)
        base_fields = {k: v for k, v in row.items() if k not in ('positive', 'negative')}

        # Positive sample: merge base fields with positive InputFeature
        pos_sample = {**base_fields, **row['positive']}
        # Negative sample: merge base fields with negative InputFeature
        neg_sample = {**base_fields, **row['negative']}

        # Interleave: [pos, neg] per pair for DP-safe slicing
        result.append(pos_sample)
        result.append(neg_sample)

    return result


def train():
    # Step 4: Prepare the dataset

    # Load the DPO dataset from ModelScope
    dataset = create_dpo_dataset()

    # Wrap the dataset into a DataLoader that yields batches
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    # Step 5: Configure the model

    # Create a multi-LoRA Transformers model pointing to the base model on ModelScope
    model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')

    # Define LoRA configuration: apply low-rank adapters to all linear layers
    lora_config = LoraConfig(
        target_modules='all-linear',
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # Attach the LoRA adapter named 'default' to the model.
    # gradient_accumulation_steps means gradients are accumulated over micro-batches
    # before an optimizer step, effectively increasing the batch size.
    model.add_adapter_to_model(adapter_name, lora_config, gradient_accumulation_steps=gradient_accumulation_steps)

    # Set the same chat template used during data preprocessing
    model.set_template('Qwen3_5Template')

    # Set the input processor (pads sequences on the right side)
    model.set_processor('InputProcessor', padding_side='right')

    # Use DPO loss for preference optimization
    model.set_loss('DPOLoss', beta=dpo_beta, loss_type=loss_type, reference_free=False, sft_weight=sft_weight)

    # Add DPO metric for logging
    model.add_metric('DPOMetric', beta=dpo_beta)

    # Use Adam optimizer with a learning rate of 1e-4
    model.set_optimizer('Adam', lr=learning_rate)

    # Step 6: Optionally resume from a previous checkpoint
    if resume_path:
        logger.info(f'Resuming training from {resume_path}')
        model.load(resume_path, load_optimizer=True)

    # Step 7: Run the training loop
    logger.info(model.get_train_configs().model_dump())

    optim_step = 0
    max_steps = len(dataloader)
    logger.info(f'Starting LoRA DPO training: loss_type={loss_type}, beta={dpo_beta}, lr={learning_rate}')
    logger.info(f'Using base model (disable_lora=True) as reference model')

    for batch in dataloader:
        # batch is List[Dict] with 'positive' and 'negative' keys
        # Convert numpy/torch tensors to lists for serialization
        for row in batch:
            for key in row:
                if isinstance(row[key], np.ndarray):
                    row[key] = row[key].tolist()
                elif isinstance(row[key], torch.Tensor):
                    row[key] = row[key].cpu().numpy().tolist()

        dpo_batch = prepare_dpo_batch(batch)

        # Get reference outputs using base model (without LoRA adapter)
        # disable_lora=True tells the model to skip LoRA and use base weights
        ref_outputs = model.forward_only(inputs=dpo_batch, disable_lora=True)
        model.forward_backward(inputs=dpo_batch, ref_outputs=ref_outputs.result)
        model.clip_grad_and_step()

        optim_step += 1

        # Logging
        if optim_step % gradient_accumulation_steps == 0:
            metrics = model.calculate_metric(is_training=True)
            logger.info(f'[Step {optim_step // gradient_accumulation_steps}/{max_steps}] {metrics}')

    # Step 8: Save the trained checkpoint
    twinkle_path = model.save(name='dpo-lora-final', save_optimizer=True)
    logger.info(f'Saved checkpoint: {twinkle_path}')

    # Step 9: Upload the checkpoint to ModelScope Hub
    # YOUR_USER_NAME = "your_username"
    # hub_model_id = f'{YOUR_USER_NAME}/twinkle-dpo-lora'
    # model.upload_to_hub(
    #     checkpoint_dir=twinkle_path,
    #     hub_model_id=hub_model_id,
    #     async_upload=False
    # )
    # logger.info(f"Uploaded checkpoint to hub: {hub_model_id}")


if __name__ == '__main__':
    train()
