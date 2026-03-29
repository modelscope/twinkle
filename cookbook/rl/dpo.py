"""DPO (Direct Preference Optimization) Training via Ray.

Off-policy preference alignment: trains the model to prefer chosen responses
over rejected responses using preference data, without explicit reward modeling.

Pipeline:
    1. Load preference dataset with chosen/rejected pairs.
    2. Encode positive and negative separately.
    3. Compute reference model log probabilities (frozen).
    4. Train policy model using DPO loss.

Architecture (Ray):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Driver (CPU)                                                    │
    │  dataloader ──► batched preference pairs                       │
    │  ref_model.forward_only() ──► reference log probs              │
    │  policy_model.forward_backward() ──► DPO loss + gradient       │
    └─────────────────────────────────────────────────────────────────┘
         │               │                    │
    DataLoader      RefModel (frozen)   PolicyModel (trainable)
                     (ref GPUs)          (policy GPUs)

DPO data format (after preprocessing):
    - positive: List[Trajectory] - chosen responses
    - negative: List[Trajectory] - rejected responses

For SimPO/ORPO variants that don't require a reference model,
set REF_MODEL_GPUS=0 to skip reference model computation.

Environment variables (all optional):
    MODEL_ID          – (default: ms://Qwen/Qwen3.5-4B)
    DATASET_ID        – (default: ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji)
    MODEL_GPUS        – GPUs for policy model                 (default: 4)
    REF_MODEL_GPUS    – GPUs for reference model              (default: 4, 0 to disable)
    BATCH_SIZE        – global batch size (preference pairs)  (default: 8)
    MICRO_BATCH_SIZE  – per-device micro batch size           (default: 2)
    MAX_STEPS         – total optimization steps              (default: 1000)
    LR                – learning rate                         (default: 5e-6)
    DPO_BETA          – DPO temperature parameter             (default: 0.1)
    LOSS_TYPE         – DPO variant (sigmoid/hinge/ipo/simpo/orpo/cpo) (default: sigmoid)
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
from twinkle.loss import CPOLoss, DPOLoss, ORPOLoss, SimPOLoss
from twinkle.metric import DPOMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor import EmojiDPOProcessor
from twinkle.processor import InputProcessor

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen2.5-7B-Instruct')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 2))
REF_MODEL_GPUS = int(os.environ.get('REF_MODEL_GPUS', 2))
NUM_GPUS = MODEL_GPUS + REF_MODEL_GPUS

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 2))  # Number of preference pairs
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 2))
LEARNING_RATE = float(os.environ.get('LR', 1e-4))  # TRL default for DPO is 5e-7 to 5e-6
DPO_BETA = float(os.environ.get('DPO_BETA', 0.1))
SFT_WEIGHT = float(os.environ.get('SFT_WEIGHT', 1.0))  # SFT loss weight for regularization
LOSS_TYPE = os.environ.get('LOSS_TYPE', 'sigmoid')  # sigmoid, hinge, ipo, simpo, orpo, cpo
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


# ── Loss Factory ──────────────────────────────────────────────────────────────

def create_loss(loss_type: str, beta: float, sft_weight: float = 0.0, reference_free: bool = False):
    """Create the appropriate loss function based on configuration."""
    if loss_type == 'simpo':
        return SimPOLoss(beta=beta, gamma=0.5)
    elif loss_type == 'orpo':
        return ORPOLoss(lambda_orpo=beta)
    elif loss_type == 'cpo':
        return CPOLoss(beta=beta, bc_coef=1.0)
    else:
        # Standard DPO variants: sigmoid, hinge, ipo
        return DPOLoss(
            beta=beta,
            loss_type=loss_type,
            reference_free=reference_free,
            sft_weight=sft_weight,
        )


# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    # Set up device groups
    device_groups = [
        DeviceGroup(name='policy', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='reference', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]

    policy_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)

    # ── DataLoader Setup ──────────────────────────────────────────────────────
    dataloader = DataLoader(
        dataset=create_dpo_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=policy_mesh,
    )
    length = len(dataloader)

    # ── Policy Model Setup ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        target_modules='all-linear',
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    policy_model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=policy_mesh,
        remote_group='policy',
    )
    MAX_STEPS = len(dataloader)
    policy_model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    policy_model.set_optimizer('AdamW', lr=LEARNING_RATE, weight_decay=0.01)
    policy_model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=LEARNING_RATE * 0.1)

    # Determine if we need reference model based on loss type
    reference_free = LOSS_TYPE in ['simpo', 'orpo', 'cpo']

    # Set up loss function and metrics
    loss_fn = create_loss(LOSS_TYPE, DPO_BETA, sft_weight=SFT_WEIGHT, reference_free=False)
    policy_model.set_loss(loss_fn)
    policy_model.add_metric(DPOMetric, beta=DPO_BETA)
    policy_model.set_processor(InputProcessor)
    policy_model.set_template('Template', model_id=MODEL_ID)

    # ── Reference Model Setup ─────────────────────────────────────────────────
    ref_model = None
    if not reference_free:
        ref_mesh = DeviceMesh.from_sizes(world_size=REF_MODEL_GPUS, dp_size=REF_MODEL_GPUS)
        ref_model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=ref_mesh,
            remote_group='reference',
        )
        ref_model.set_processor(InputProcessor)
        ref_model.set_template('Template', model_id=MODEL_ID)
        logger.info('Reference model initialized for DPO training')
    else:
        logger.info(f'Training without reference model (loss_type={LOSS_TYPE})')

    optim_step = 0
    logger.info(get_device_placement())
    logger.info(f'Starting DPO training: loss_type={LOSS_TYPE}, beta={DPO_BETA}')

    # ── Training Loop ─────────────────────────────────────────────────────────
    for batch in dataloader:
        # batch is List[Dict] with 'positive' and 'negative' keys
        dpo_batch = prepare_dpo_batch(batch)

        # Get reference outputs (lazy - not collected to driver)
        ref_outputs = None
        if ref_model is not None:
            ref_outputs = ref_model.forward_only(inputs=dpo_batch)

        # Forward-backward pass with DPO loss
        # ref_outputs is passed to loss which extracts logps internally
        policy_model.forward_backward(
            inputs=dpo_batch,
            ref_outputs=ref_outputs,
        )

        # Gradient clipping and optimizer step
        policy_model.clip_grad_and_step()
        optim_step += 1

        # Logging
        if optim_step % 1 == 0:
            metrics = policy_model.calculate_metric(is_training=True)
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metrics}')

        # Checkpointing
        if optim_step % SAVE_STEPS == 0:
            policy_model.save(f'dpo-checkpoint-{optim_step}')

    # ── Save Final Checkpoint ─────────────────────────────────────────────────
    logger.info(f'Training completed. Total steps: {optim_step}')
    policy_model.save('dpo-final-checkpoint')


if __name__ == '__main__':
    main()
