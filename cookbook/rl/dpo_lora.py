"""DPO (Direct Preference Optimization) Training with LoRA (Single GPU Group).

LoRA-based DPO training: uses the base model (without LoRA adapter) as reference
model by calling forward_only with adapter_name=''. This eliminates the need for
a separate reference model GPU group.

Pipeline:
    1. Load preference dataset with chosen/rejected pairs.
    2. Encode positive and negative separately.
    3. Compute reference model log probabilities using base model (adapter_name='').
    4. Train policy model (with LoRA adapter) using DPO loss.

Architecture (Ray - Single Group):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Driver (CPU)                                                    │
    │  dataloader ──► batched preference pairs                       │
    │  policy_model.forward_only(adapter_name='') ──► reference logps│
    │  policy_model.forward_backward() ──► DPO loss + gradient       │
    └─────────────────────────────────────────────────────────────────┘
         │
    PolicyModel (with LoRA adapter)
     - forward_only(adapter_name='') → base model inference (reference)
     - forward_backward() → LoRA adapter training (policy)

DPO data format (after preprocessing):
    - positive: List[Trajectory] - chosen responses
    - negative: List[Trajectory] - rejected responses

Environment variables (all optional):
    MODEL_ID          – (default: ms://Qwen/Qwen2.5-7B-Instruct)
    DATASET_ID        – (default: ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji)
    MODEL_GPUS        – GPUs for policy model                 (default: 4)
    BATCH_SIZE        – global batch size (preference pairs)  (default: 8)
    MICRO_BATCH_SIZE  – per-device micro batch size           (default: 2)
    MAX_STEPS         – total optimization steps              (default: 1000)
    LR                – learning rate                         (default: 1e-4)
    DPO_BETA          – DPO temperature parameter             (default: 0.1)
    LOSS_TYPE         – DPO variant (sigmoid/hinge/ipo)       (default: sigmoid)
    SAVE_STEPS        – checkpoint save interval              (default: 100)
    MAX_LENGTH        – max sequence length                   (default: 2048)

    Dataset field mapping (for custom datasets):
    PROMPT_KEY        – key for prompt field                  (default: 'prompt')
    CHOSEN_KEY        – key for chosen response               (default: 'answer_zh')
    REJECTED_KEY      – key for rejected response             (default: 'answer_en')
    SYSTEM_PROMPT     – system prompt to prepend              (default: 'You are a helpful assistant.')
"""

import os
from typing import Any, Dict, List, Optional

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import DPOLoss
from twinkle.metric import DPOMetric
from twinkle.model import MultiLoraMegatronModel
from twinkle.preprocessor import EmojiDPOProcessor
from twinkle.processor import InputProcessor

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen2.5-7B-Instruct')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 8))

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 2))  # Number of preference pairs
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 2))
LEARNING_RATE = float(os.environ.get('LR', 1e-4))  # LoRA DPO requires higher LR (1e-4 to 3e-4)
DPO_BETA = float(os.environ.get('DPO_BETA', 0.1))
SFT_WEIGHT = float(os.environ.get('SFT_WEIGHT', 1.0))  # SFT loss weight for regularization
LOSS_TYPE = os.environ.get('LOSS_TYPE', 'sigmoid')  # sigmoid, hinge, ipo
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 100))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', 2048))
ADAPTER_NAME = 'default'
SYSTEM_PROMPT = os.environ.get('SYSTEM_PROMPT', 'You are a helpful assistant.')


def create_dpo_dataset():
    """Create DPO dataset with positive/negative format."""
    dataset = Dataset(DatasetMeta(DATASET_ID, data_slice=range(30000)))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=MAX_LENGTH)
    dataset.map(
        EmojiDPOProcessor,
        init_args={
            'system': SYSTEM_PROMPT,
        }
    )
    # DPO preprocessor returns {'positive': [...], 'negative': [...]}
    # batch_encode handles this format automatically
    dataset.encode(load_from_cache_file=True)
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


# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    # Set up device groups - only one group for LoRA training
    device_groups = [
        DeviceGroup(name='policy', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
    ]

    policy_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=1, pp_size=2, cp_size=2, tp_size=2)
    twinkle.initialize(mode='ray', nproc_per_node=8, groups=device_groups)

    # ── DataLoader Setup ──────────────────────────────────────────────────────
    dataloader = DataLoader(
        dataset=create_dpo_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=policy_mesh,
    )
    length = len(dataloader)

    # ── Policy Model Setup with LoRA ──────────────────────────────────────────
    lora_config = LoraConfig(
        target_modules='all-linear',
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    policy_model = MultiLoraMegatronModel(
        model_id=MODEL_ID,
        device_mesh=policy_mesh,
        remote_group='policy',
    )
    MAX_STEPS = len(dataloader)
    policy_model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    # policy_model.set_optimizer('AdamW', lr=LEARNING_RATE, weight_decay=0.01, adapter_name=ADAPTER_NAME)
    # policy_model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, adapter_name=ADAPTER_NAME)
    policy_model.set_optimizer('default', lr=LEARNING_RATE, weight_decay=0.01, adapter_name=ADAPTER_NAME)
    policy_model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, adapter_name=ADAPTER_NAME)

    # Set up loss function and metrics
    loss_fn = DPOLoss(
        beta=DPO_BETA,
        loss_type=LOSS_TYPE,
        reference_free=False,  # We use base model as reference via disable_lora=True
        sft_weight=SFT_WEIGHT,
    )
    policy_model.set_loss(loss_fn, adapter_name=ADAPTER_NAME)
    policy_model.add_metric(DPOMetric, beta=DPO_BETA, adapter_name=ADAPTER_NAME)
    policy_model.set_processor(InputProcessor, adapter_name=ADAPTER_NAME)
    policy_model.set_template('Template', model_id=MODEL_ID, adapter_name=ADAPTER_NAME)

    optim_step = 0
    logger.info(get_device_placement())
    logger.info(f'Starting LoRA DPO training: loss_type={LOSS_TYPE}, beta={DPO_BETA}, lr={LEARNING_RATE}')
    logger.info(f'Using base model (disable_lora=True) as reference model')

    # ── Training Loop ─────────────────────────────────────────────────────────
    for batch in dataloader:
        # batch is List[Dict] with 'positive' and 'negative' keys
        dpo_batch = prepare_dpo_batch(batch)

        # Get reference outputs using base model (without LoRA adapter)
        # disable_lora=True tells the model to skip LoRA and use base weights
        ref_outputs = policy_model.forward_only(inputs=dpo_batch, micro_batch_size=2, disable_lora=True, adapter_name=ADAPTER_NAME)
        # Forward-backward pass with DPO loss (using LoRA adapter)
        # ref_outputs is passed to loss which extracts logps internally
        policy_model.forward_backward(
            inputs=dpo_batch,
            ref_outputs=ref_outputs,
            micro_batch_size=2,
            adapter_name=ADAPTER_NAME
        )

        # Gradient clipping and optimizer step
        policy_model.clip_grad_and_step(adapter_name=ADAPTER_NAME)
        optim_step += 1

        # Logging
        if optim_step % 1 == 0:
            metrics = policy_model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME)
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metrics}')

        # Checkpointing
        if optim_step % SAVE_STEPS == 0:
            policy_model.save(f'dpo-lora-checkpoint-{optim_step}', adapter_name=ADAPTER_NAME)

    # ── Save Final Checkpoint ─────────────────────────────────────────────────
    logger.info(f'Training completed. Total steps: {optim_step}')
    policy_model.save('dpo-lora-final-checkpoint', adapter_name=ADAPTER_NAME)


if __name__ == '__main__':
    main()
