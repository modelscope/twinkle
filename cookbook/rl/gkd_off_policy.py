"""GKD Off-Policy Distillation via Ray.

Off-policy knowledge distillation: the student learns to match the teacher's
token distribution on pre-existing reference responses from the dataset.

Pipeline:
    1. DataLoader supplies full-text batches (prompt + reference answer).
    2. Teacher vLLM sampler computes top-k prompt logprobs on the sequences.
    3. Student TransformersModel runs forward_backward() with GKDLoss.

Key difference from on-policy:
    - No student sampler needed (responses already in the dataset).
    - Faster per-step (no generation latency), but less exploration.

Architecture (Ray):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Driver (CPU)                                                    │
    │  dataloader ──► full-text batch (prompt + reference answer)    │
    │  teacher_sampler.sample(prompt_logprobs=k) ──► teacher lps     │
    │  student_model.forward_backward(teacher_output=...) ──► GKD    │
    └─────────────────────────────────────────────────────────────────┘
                        │
         vLLMSampler + TransformersModel
          (teacher)       (student)

Environment variables (all optional):
    STUDENT_MODEL_ID  – (default: ms://Qwen/Qwen3-0.6B)
    TEACHER_MODEL_ID  – (default: ms://Qwen/Qwen3-8B)
    MODEL_GPUS        – GPUs for student model        (default: 4)
    SAMPLER_GPUS      – GPUs for teacher vLLM sampler (default: 2)
    BATCH_SIZE        – global batch size             (default: 8)
    MAX_STEPS         – total optimisation steps      (default: 200)
    LR                – learning rate                 (default: 1e-4)
    GKD_BETA          – JSD beta (0=fwd KL, 1=rev KL) (default: 0.5)
    GKD_TEMPERATURE   – distillation temperature      (default: 1.0)
    GKD_TOPK          – top-k vocab for teacher logprobs (default: 20)
"""

import os
from typing import List, Optional

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import GKDLoss
from twinkle.model import TransformersModel
from twinkle.preprocessor import GSM8KFullProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', 'ms://Qwen/Qwen3-0.6B')
TEACHER_MODEL_ID = os.environ.get('TEACHER_MODEL_ID', 'ms://Qwen/Qwen3-8B')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 2))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 200))
LEARNING_RATE = float(os.environ.get('LR', 1e-4))

GKD_BETA = float(os.environ.get('GKD_BETA', 0.5))
GKD_TEMPERATURE = float(os.environ.get('GKD_TEMPERATURE', 1.0))
GKD_TOPK = int(os.environ.get('GKD_TOPK', 20))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 1024))

ADAPTER_NAME = 'default'


# ── Dataset ───────────────────────────────────────────────────────────────────

def create_dataset():
    """Full-text dataset with prompt + reference answer for off-policy distillation."""
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Template', model_id=STUDENT_MODEL_ID, max_length=1024)
    dataset.map(GSM8KFullProcessor())
    return dataset


# ── Utility ───────────────────────────────────────────────────────────────────

def convert_topk_prompt_logprobs(
    topk_prompt_logprobs_batch: List[List[Optional[List[tuple]]]],
) -> dict:
    """Convert vLLM topk_prompt_logprobs to GKDLoss teacher_output format."""
    batch_logprobs = []
    batch_indices = []

    for seq_topk in topk_prompt_logprobs_batch:
        seq_logprobs = []
        seq_indices = []
        for pos_topk in seq_topk:
            if pos_topk is None:
                continue
            else:
                seq_logprobs.append([lp for _, lp in pos_topk])
                seq_indices.append([tid for tid, _ in pos_topk])
        batch_logprobs.append(seq_logprobs)
        batch_indices.append(seq_indices)

    # Pad to same seq_len within batch
    max_len = max(len(seq) for seq in batch_logprobs)
    topk = len(batch_logprobs[0][0]) if batch_logprobs and batch_logprobs[0] else GKD_TOPK

    for i in range(len(batch_logprobs)):
        pad_len = max_len - len(batch_logprobs[i])
        if pad_len > 0:
            batch_logprobs[i].extend([[0.0] * topk] * pad_len)
            batch_indices[i].extend([[0] * topk] * pad_len)

    return {
        'teacher_topk_logprobs': torch.tensor(batch_logprobs, dtype=torch.float32),
        'teacher_topk_indices': torch.tensor(batch_indices, dtype=torch.long),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    device_groups = [
        DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='cuda'),
        DeviceGroup(name='teacher_sampler', ranks=SAMPLER_GPUS, device_type='cuda'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
    )
    logger.info(get_device_placement())

    # ── Student model (trainable) ──────────────────────────────────────────────
    student_model = TransformersModel(
        model_id=STUDENT_MODEL_ID,
        device_mesh=model_mesh,
        remote_group='student_model',
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

    # ── Teacher vLLM sampler (for prompt logprobs) ─────────────────────────────
    teacher_sampler = vLLMSampler(
        model_id=TEACHER_MODEL_ID,
        engine_args={'gpu_memory_utilization': 0.85, 'max_model_len': 2048, 'logprobs_mode': 'raw_logprobs'},
        device_mesh=sampler_mesh,
        remote_group='teacher_sampler',
    )
    teacher_sampler.set_template(Template, model_id=TEACHER_MODEL_ID)

    # ── DataLoader (full-text: prompt + reference answer) ──────────────────────
    dataloader = DataLoader(
        dataset=create_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='student_model',
    )

    logger.info(f'GKD Off-Policy | student={STUDENT_MODEL_ID}  teacher={TEACHER_MODEL_ID}')
    logger.info(f'  beta={GKD_BETA}  T={GKD_TEMPERATURE}  topk={GKD_TOPK}')

    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        # Teacher vLLM computes top-k prompt logprobs on the reference sequences
        teacher_response = teacher_sampler.sample(
            batch,
            SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=1.0, prompt_logprobs=1, logprobs=GKD_TOPK),
        )
        input_data = [seq.new_input_feature for response in teacher_response for seq in response.sequences]
        input_ids_list = []
        for data in input_data:
            input_ids_list.append(data.pop('input_ids', None))

        # Convert teacher logprobs to tensor format for GKDLoss
        teacher_output = convert_topk_prompt_logprobs(
            [resp.prompt_logprobs for resp in teacher_response],
            [[sequence.logprobs for sequence in resp.sequences] for resp in teacher_response],
        )

        # Student forward + GKD backward
        student_model.forward_backward(inputs=input_data, **teacher_output)
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
