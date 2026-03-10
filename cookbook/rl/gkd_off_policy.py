"""GKD Off-Policy Distillation via Ray.

Off-policy knowledge distillation: the student learns to match the teacher's
token distribution on pre-existing reference responses from the dataset.

Pipeline:
    1. DataLoader supplies full-text batches (prompt + reference answer).
    2. Teacher TransformersModel runs forward_only() to get frozen logits.
    3. Student TransformersModel runs forward_backward() with GKDLoss.

Key difference from on-policy:
    - No vLLM sampler needed (responses already in the dataset).
    - Simpler GPU layout: all GPUs can go to the model group.
    - Faster per-step (no generation latency), but less exploration.

Architecture (Ray):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Driver (CPU)                                                    │
    │  dataloader ──► full-text batch (prompt + reference answer)    │
    │  teacher_model.forward_only() ──► frozen teacher logits        │
    │  student_model.forward_backward(teacher_logits=...) ──► GKD    │
    └─────────────────────────────────────────────────────────────────┘
                        │
               TransformersModel ×2
            student + teacher (all GPUs)

Environment variables (all optional):
    STUDENT_MODEL_ID  – (default: ms://Qwen/Qwen2.5-1.5B-Instruct)
    TEACHER_MODEL_ID  – (default: ms://Qwen/Qwen2.5-7B-Instruct)
    NUM_GPUS          – total GPUs for both models (default: 4)
    BATCH_SIZE        – global batch size           (default: 8)
    MAX_STEPS         – total optimisation steps    (default: 200)
    LR                – learning rate               (default: 1e-4)
    GKD_BETA          – JSD beta (0=fwd KL, 1=rev KL) (default: 0.5)
    GKD_TEMPERATURE   – distillation temperature      (default: 1.0)
    GKD_TOPK          – top-k vocab reduction; 0=full (default: 0)
"""

import os

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import GKDLoss
from twinkle.model import TransformersModel
from twinkle.preprocessor import GSM8KFullProcessor

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', 'ms://Qwen/Qwen2.5-1.5B-Instruct')
TEACHER_MODEL_ID = os.environ.get('TEACHER_MODEL_ID', 'ms://Qwen/Qwen2.5-7B-Instruct')

NUM_GPUS = int(os.environ.get('NUM_GPUS', 4))

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 200))
LEARNING_RATE = float(os.environ.get('LR', 1e-4))

GKD_BETA = float(os.environ.get('GKD_BETA', 0.5))
GKD_TEMPERATURE = float(os.environ.get('GKD_TEMPERATURE', 1.0))
GKD_TOPK = int(os.environ.get('GKD_TOPK', 0))

ADAPTER_NAME = 'default'


# ── Dataset ───────────────────────────────────────────────────────────────────

def create_dataset():
    """Full-text dataset with prompt + reference answer for off-policy distillation."""
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Template', model_id=STUDENT_MODEL_ID, max_length=1024)
    dataset.map(GSM8KFullProcessor())
    dataset.encode()
    return dataset


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(NUM_GPUS)), device_type='cuda'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=NUM_GPUS, dp_size=NUM_GPUS)

    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
        lazy_collect=False,
    )
    logger.info(get_device_placement())

    # ── Student model (trainable) ──────────────────────────────────────────────
    student_model = TransformersModel(
        model_id=STUDENT_MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
    )
    student_model.add_adapter_to_model(
        ADAPTER_NAME,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules='all-linear'),
        gradient_accumulation_steps=1,
    )
    student_model.set_optimizer('AdamW', lr=LEARNING_RATE)
    student_model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    student_model.set_loss(GKDLoss(beta=GKD_BETA, temperature=GKD_TEMPERATURE))
    student_model.set_template('Template', model_id=STUDENT_MODEL_ID)

    # ── Teacher model (frozen, for logits) ─────────────────────────────────────
    teacher_model = TransformersModel(
        model_id=TEACHER_MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
    )
    teacher_model.set_template('Template', model_id=TEACHER_MODEL_ID)

    # ── DataLoader (full-text: prompt + reference answer) ──────────────────────
    dataloader = DataLoader(
        dataset=create_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    topk = GKD_TOPK if GKD_TOPK > 0 else None

    logger.info(f'GKD Off-Policy | student={STUDENT_MODEL_ID}  teacher={TEACHER_MODEL_ID}')
    logger.info(f'  beta={GKD_BETA}  T={GKD_TEMPERATURE}  topk={GKD_TOPK}')

    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        input_data = batch if isinstance(batch, list) else [batch]

        # Teacher logits (frozen)
        teacher_output = teacher_model.forward_only(inputs=input_data)
        teacher_logits = teacher_output.get('logits')

        # Student forward + GKD backward
        student_model.forward_backward(inputs=input_data, teacher_logits=teacher_logits, topk=topk)
        student_model.clip_grad_and_step()
        optim_step += 1

        if optim_step % 10 == 0:
            metric = student_model.calculate_metric(is_training=True)
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metric}')

        if optim_step % 50 == 0:
            student_model.save(f'gkd-offpolicy-ckpt-{optim_step}')

    student_model.save('gkd-offpolicy-final')
    logger.info('GKD off-policy training completed.')


if __name__ == '__main__':
    main()
