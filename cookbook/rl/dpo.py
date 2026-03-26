"""DPO (Direct Preference Optimization) Training via Ray.

Off-policy preference alignment: trains the model to prefer chosen responses
over rejected responses using preference data, without explicit reward modeling.

Pipeline:
    1. Load preference dataset with chosen/rejected pairs.
    2. Encode chosen and rejected separately.
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

DPO Trajectory format:
    - messages: List[Message] - chosen response
    - extend_message: [('rejected_messages', List[Message])] - rejected response

For SimPO/ORPO variants that don't require a reference model,
set USE_REFERENCE_MODEL=0 to skip reference model computation.

Environment variables (all optional):
    MODEL_ID          – (default: ms://Qwen/Qwen3.5-4B)
    DATASET_ID        – (default: ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji)
    MODEL_GPUS        – GPUs for policy model                 (default: 4)
    REF_MODEL_GPUS    – GPUs for reference model              (default: 4, 0 to disable)
    USE_REFERENCE_MODEL – Whether to use reference model      (default: 1)
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

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import Message, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CPOLoss, DPOLoss, ORPOLoss, SimPOLoss
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.template import Template

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
REF_MODEL_GPUS = int(os.environ.get('REF_MODEL_GPUS', 4))
USE_REFERENCE_MODEL = bool(int(os.environ.get('USE_REFERENCE_MODEL', 1)))

# Adjust total GPUs based on whether reference model is used
if USE_REFERENCE_MODEL and REF_MODEL_GPUS > 0:
    NUM_GPUS = MODEL_GPUS + REF_MODEL_GPUS
else:
    NUM_GPUS = MODEL_GPUS
    USE_REFERENCE_MODEL = False

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))  # Number of preference pairs
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 1000))
LEARNING_RATE = float(os.environ.get('LR', 5e-6))
DPO_BETA = float(os.environ.get('DPO_BETA', 0.1))
LOSS_TYPE = os.environ.get('LOSS_TYPE', 'sigmoid')  # sigmoid, hinge, ipo, simpo, orpo, cpo
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 100))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', 2048))
ADAPTER_NAME = 'default'


# ── Dataset ───────────────────────────────────────────────────────────────────

# Dataset field configuration for shareAI/DPO-zh-en-emoji
PROMPT_KEY = os.environ.get('PROMPT_KEY', 'prompt')
CHOSEN_KEY = os.environ.get('CHOSEN_KEY', 'answer_zh')
REJECTED_KEY = os.environ.get('REJECTED_KEY', 'answer_en')
SYSTEM_PROMPT = os.environ.get('SYSTEM_PROMPT', 'You are a helpful assistant.')


def create_dpo_dataset():
    """Create preference dataset for DPO training.

    Uses shareAI/DPO-zh-en-emoji dataset:
    - answer_zh: chosen response (Chinese)
    - answer_en: rejected response (English)

    Returns raw dataset without preprocessing - preprocessing is done
    in prepare_dpo_batch() to avoid PyArrow serialization issues.
    """
    dataset = Dataset(DatasetMeta(DATASET_ID, split='train'))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=MAX_LENGTH)
    # Do NOT apply preprocessor here - raw data will be processed in prepare_dpo_batch
    return dataset


def prepare_dpo_batch(
    batch: List[Dict[str, Any]],
    template: Template,
) -> List[Dict[str, Any]]:
    """Prepare DPO batch: build trajectories and encode both chosen and rejected.

    Args:
        batch: List of raw data dicts from dataset (e.g., {prompt, answer_zh, answer_en})

    Returns:
        List organized as [chosen_1, ..., chosen_n, rejected_1, ..., rejected_n]
        where each item is an encoded InputFeature dict.
    """
    chosen_trajectories = []
    rejected_trajectories = []

    for item in batch:
        # Build messages from raw data
        prompt = item.get(PROMPT_KEY, '')
        chosen_response = item.get(CHOSEN_KEY, '')
        rejected_response = item.get(REJECTED_KEY, '')

        # Build prompt messages
        prompt_messages = []
        if SYSTEM_PROMPT:
            prompt_messages.append(Message(role='system', content=SYSTEM_PROMPT))
        prompt_messages.append(Message(role='user', content=prompt))

        # Build chosen and rejected trajectories
        chosen_messages = prompt_messages + [Message(role='assistant', content=chosen_response)]
        rejected_messages = prompt_messages + [Message(role='assistant', content=rejected_response)]

        chosen_trajectories.append(Trajectory(messages=chosen_messages))
        rejected_trajectories.append(Trajectory(messages=rejected_messages))

    # Batch encode all trajectories (properly handles multimodal preprocessing)
    chosen_encoded = template.batch_encode(chosen_trajectories)
    rejected_encoded = template.batch_encode(rejected_trajectories)

    # Convert to list of dicts
    chosen_samples = [dict(enc) for enc in chosen_encoded]
    rejected_samples = [dict(enc) for enc in rejected_encoded]

    # Return [chosen..., rejected...]
    return chosen_samples + rejected_samples


# ── Loss Factory ──────────────────────────────────────────────────────────────

def create_loss(loss_type: str, beta: float, reference_free: bool = False):
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
        )


# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    # Set up device groups
    if USE_REFERENCE_MODEL:
        device_groups = [
            DeviceGroup(name='policy', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
            DeviceGroup(name='reference', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
        ]
    else:
        device_groups = [
            DeviceGroup(name='policy', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        ]

    policy_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # ── Policy Model Setup ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    policy_model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=policy_mesh,
        remote_group='policy',
    )
    policy_model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    policy_model.set_optimizer('AdamW', lr=LEARNING_RATE, weight_decay=0.01)
    policy_model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=LEARNING_RATE * 0.1)

    # Determine if we need reference model based on loss type
    reference_free = LOSS_TYPE in ['simpo', 'orpo', 'cpo']

    # Set up loss function
    loss_fn = create_loss(LOSS_TYPE, DPO_BETA, reference_free=not USE_REFERENCE_MODEL)
    policy_model.set_loss(loss_fn)
    policy_model.set_processor(InputProcessor)
    policy_model.set_template('Template', model_id=MODEL_ID)

    # Get template for encoding rejected messages
    template = Template(model_id=MODEL_ID, max_length=MAX_LENGTH)

    # ── Reference Model Setup (if needed) ─────────────────────────────────────
    ref_model = None
    if USE_REFERENCE_MODEL and not reference_free:
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

    # ── DataLoader Setup ──────────────────────────────────────────────────────
    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_dpo_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=policy_mesh,
        remote_group='policy',
    )

    optim_step = 0
    logger.info(get_device_placement())
    logger.info(f'Starting DPO training: loss_type={LOSS_TYPE}, beta={DPO_BETA}, '
                f'use_ref_model={USE_REFERENCE_MODEL}')

    # ── Training Loop ─────────────────────────────────────────────────────────
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        # Prepare DPO batch: [chosen..., rejected...]
        batch_list = batch if isinstance(batch, list) else [batch]
        dpo_batch = prepare_dpo_batch(batch_list, template)

        # Compute reference log probabilities if using reference model
        ref_logps = None
        if ref_model is not None:
            with torch.no_grad():
                ref_outputs = ref_model.forward_only(inputs=dpo_batch)
                ref_logps = ref_outputs.get('logps')

        # Forward-backward pass with DPO loss
        # micro_batch_size should be even to maintain chosen/rejected pairing
        actual_micro_batch = MICRO_BATCH_SIZE * 2
        policy_model.forward_backward(
            inputs=dpo_batch,
            ref_logps=ref_logps,
            micro_batch_size=actual_micro_batch,
        )

        # Gradient clipping and optimizer step
        policy_model.clip_grad_and_step()
        optim_step += 1

        # Logging
        if optim_step % 10 == 0:
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
