# Tinker-Compatible Client - DPO (Direct Preference Optimization) Training with LoRA
#
# This script demonstrates how to fine-tune a language model using DPO
# through the Tinker-compatible client API.
#
# Training flow per step:
#   1. forward_backward with 'cross_entropy' + disable_lora=True
#      → base-model forward pass; LoRA weights are NOT in the computation graph
#        so backward accumulates zero LoRA gradients (safe to discard).
#   2. Attach returned per-token ref logps to each datum's loss_fn_inputs.
#   3. forward_backward with 'importance_sampling'
#      → server detects ref_logps and switches to DPOLoss + DPOMetric.
#   4. optim_step → update LoRA, DPO metrics returned automatically.
#
# The server must be running first (see server.py and server_config.yaml).

import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Any, Dict, List

import swanlab

from tinker import types
from twinkle import init_tinker_client, get_logger
from twinkle.dataset import Dataset, DatasetMeta, LazyDataset
from twinkle.dataloader import DataLoader
from twinkle.preprocessor import EmojiDPOProcessor
from twinkle.server.common import input_feature_to_datum

logger = get_logger()

# Initialize the Tinker client before importing ServiceClient
init_tinker_client()

from tinker import ServiceClient  # noqa: E402 (must follow init_tinker_client)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
base_model = 'Qwen/Qwen3.5-4B'
base_url = 'http://localhost:8000'
api_key = 'EMPTY_API_KEY'
dataset_id = 'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji'

batch_size = 4
learning_rate = 1e-4
dpo_beta = 0.1
sft_weight = 1.0
max_length = 2048
lora_rank = 8
system_prompt = 'You are a helpful assistant.'
use_swanlab = False


# ---------------------------------------------------------------------------
# Dataset helpers  (reused from twinkle/self_host/dpo.py)
# ---------------------------------------------------------------------------

def create_dpo_dataset():
    """Create DPO dataset with positive/negative format."""
    dataset = LazyDataset(DatasetMeta(dataset_id, data_slice=range(5000)))
    dataset.set_template('Qwen3_5Template', model_id=f'ms://{base_model}', max_length=max_length)
    dataset.map(
        EmojiDPOProcessor,
        init_args={'system': system_prompt},
    )
    # EmojiDPOProcessor returns {'positive': InputFeature, 'negative': InputFeature, ...}
    # encode handles this format automatically
    dataset.encode()
    return dataset


def prepare_dpo_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorganise batch into DP-safe interleaved format [pos_1, neg_1, pos_2, neg_2, ...].

    Args:
        batch: List of rows, each with 'positive' and 'negative' InputFeatures.

    Returns:
        Interleaved list so each DP worker slice contains complete pairs.
    """
    result = []
    for row in batch:
        base_fields = {k: v for k, v in row.items() if k not in ('positive', 'negative')}
        pos_sample = {**base_fields, **row['positive']}
        neg_sample = {**base_fields, **row['negative']}
        result.append(pos_sample)
        result.append(neg_sample)
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    # Step 0: Initialize SwanLab if enabled
    if use_swanlab:
        swanlab.login(api_key=os.environ['SWANLAB_API_KEY'])
        swanlab.init(
            project='twinkle-dpo',
            experiment_name='dpo-lora-training',
            config={
                'base_model': base_model,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'dpo_beta': dpo_beta,
                'sft_weight': sft_weight,
                'max_length': max_length,
                'lora_rank': lora_rank,
            },
        )
        logger.info('SwanLab initialized')

    # Step 1: Prepare dataset & dataloader
    logger.info('Loading DPO dataset...')
    dataset = create_dpo_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    logger.info(f'Dataset ready: {len(dataloader)} steps per epoch')

    # Step 2: Connect to server and create LoRA training client
    service_client = ServiceClient(base_url=base_url, api_key=api_key)
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=lora_rank,
    )
    logger.info(f'LoRA training client created (rank={lora_rank})')
    logger.info(f'Starting DPO training: beta={dpo_beta}, lr={learning_rate}')

    # Step 3: Training loop
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Normalise numpy / torch tensors to plain Python lists for serialisation
        for row in batch:
            for key in list(row.keys()):
                if isinstance(row[key], np.ndarray):
                    row[key] = row[key].tolist()
                elif isinstance(row[key], torch.Tensor):
                    row[key] = row[key].cpu().numpy().tolist()

        # Build interleaved [pos, neg, pos, neg, ...] batch
        dpo_batch = prepare_dpo_batch(batch)

        # Convert each InputFeature dict to a Tinker Datum
        input_datums = [input_feature_to_datum(row) for row in dpo_batch]

        # -----------------------------------------------------------------
        # A. Reference forward pass (base model, disable_lora=True)
        #    LoRA weights are outside the computation graph → backward
        #    produces zero LoRA gradients, so this call is safe.
        # -----------------------------------------------------------------
        ref_result = training_client.forward(
            input_datums,
            'cross_entropy',
            loss_fn_config={'disable_lora': True},
        ).result()

        # -----------------------------------------------------------------
        # B. Attach per-token ref logps to each datum's loss_fn_inputs
        # -----------------------------------------------------------------
        for datum, ref_out in zip(input_datums, ref_result.loss_fn_outputs):
            ref_logprobs_np = np.array(ref_out['logprobs'].tolist(), dtype=np.float32)
            datum.loss_fn_inputs['ref_logps'] = types.TensorData.from_numpy(ref_logprobs_np)

        # -----------------------------------------------------------------
        # C. DPO forward_backward
        #    Server detects ref_logps → sets DPOLoss + DPOMetric automatically.
        #    Optional DPO hyper-params can be forwarded via loss_fn_config.
        #    (e.g. beta, sft_weight, not support dpo_loss_type for tinker)
        # -----------------------------------------------------------------
        fwdbwd_result = training_client.forward_backward(
            input_datums,
            'importance_sampling',
            loss_fn_config={
                'dpo_beta': dpo_beta,
                'dpo_sft_weight': sft_weight,
            },
        ).result()

        # -----------------------------------------------------------------
        # D. Optimizer step — DPOMetric is calculated automatically on the
        #    server and returned inside optim_result.metrics.
        # -----------------------------------------------------------------
        optim_result = training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate)
        ).result()

        logger.info(f'[Step {step}] metrics={optim_result.metrics}')

        # Log metrics to SwanLab
        if use_swanlab and optim_result.metrics:
            swanlab.log(optim_result.metrics, step=step)

    # Step 4: Save checkpoint
    save_result = training_client.save_state('dpo-lora-final').result()
    logger.info(f'Saved checkpoint: {save_result.path}')

    # Step 5: (Optional) Upload to ModelScope Hub
    # YOUR_USER_NAME = 'your_username'
    # hub_model_id = f'{YOUR_USER_NAME}/twinkle-tinker-dpo-lora'
    # training_client.publish_checkpoint_from_tinker_path(save_result.path).result()
    # logger.info(f'Uploaded checkpoint to hub: {hub_model_id}')


if __name__ == '__main__':
    train()
