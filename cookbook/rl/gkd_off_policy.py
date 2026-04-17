"""GKD Off-Policy Distillation via Ray.

Off-policy knowledge distillation: the student learns to match the teacher's
token distribution on pre-existing reference responses from the dataset.

Pipeline:
    1. DataLoader supplies full-text batches (prompt + reference answer).
    2. Teacher vLLM sampler computes top-k prompt logprobs on the sequences.
    3. Student MegatronModel runs forward_backward() with GKDLoss.

Key difference from on-policy:
    - No student sampler needed (responses already in the dataset).
    - No weight sync needed (student doesn't sample).
    - Faster per-step (no generation latency), but less exploration.

Architecture (Ray):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Driver (CPU)                                                    │
    │  dataloader ──► full-text batch (prompt + reference answer)    │
    │  teacher_sampler.sample(prompt_logprobs=k) ──► teacher lps     │
    │  student_model.forward_backward(teacher_output=...) ──► GKD    │
    └─────────────────────────────────────────────────────────────────┘
                        │
         vLLMSampler + MegatronModel
          (teacher)       (student)

Environment variables (all optional):
    STUDENT_MODEL_ID  – (default: ms://Qwen/Qwen3-0.6B)
    TEACHER_MODEL_ID  – (default: ms://Qwen/Qwen3-8B)
    MODEL_GPUS        – GPUs for student model        (default: 8)
    SAMPLER_GPUS      – GPUs for teacher vLLM sampler (default: 8)
    BATCH_SIZE        – global batch size             (default: 16)
    MAX_STEPS         – total optimisation steps      (default: 1000)
    LR                – learning rate                 (default: 5e-5)
    GKD_BETA          – JSD beta (0=fwd KL, 1=rev KL) (default: 0.5)
    GKD_TEMPERATURE   – distillation temperature      (default: 1.0)
    GKD_TOPK          – top-k vocab for teacher logprobs (default: 64)
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
from twinkle.model import MegatronModel
from twinkle.preprocessor import GSM8KProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', 'ms://Qwen/Qwen3-0.6B')
TEACHER_MODEL_ID = os.environ.get('TEACHER_MODEL_ID', 'ms://Qwen/Qwen3-8B')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 16))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 1000))
LEARNING_RATE = float(os.environ.get('LR', 5e-5))

GKD_BETA = float(os.environ.get('GKD_BETA', 0.5))
GKD_TEMPERATURE = float(os.environ.get('GKD_TEMPERATURE', 1.0))
GKD_TOPK = int(os.environ.get('GKD_TOPK', 64))
ADAPTER_NAME = 'default'
SYSTEM_PROMPT = ('You are a helpful math assistant. Solve the problem step by step and put '
                 'your final answer within #### <number>')


# ── Dataset ───────────────────────────────────────────────────────────────────

def create_dataset():
    """Full-text dataset with prompt + reference answer for off-policy distillation."""
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Template', model_id=STUDENT_MODEL_ID, max_length=1024)
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT, add_assistant=True))
    return dataset


# ── Utility ───────────────────────────────────────────────────────────────────

def convert_topk_prompt_logprobs(
    topk_prompt_logprobs_batch: List[Optional[List[List[tuple]]]],
) -> dict:
    """Convert vLLM topk_prompt_logprobs to GKDLoss teacher_output format.

    Args:
        topk_prompt_logprobs_batch: [batch] each is topk_prompt_logprobs for one request.
            Shape: [seq_len, topk] per request, where each position is List[(token_id, logprob)].

    Returns:
        Dict with teacher logprobs/indices tensors.
    """
    batch_logprobs = []
    batch_indices = []

    for seq_topk in topk_prompt_logprobs_batch:
        seq_logprobs = []
        seq_indices = []
        if seq_topk is not None:
            for pos_topk in seq_topk:
                if pos_topk is None:
                    # First position is None, fill with placeholder
                    seq_logprobs.append([0.0] * GKD_TOPK)
                    seq_indices.append([0] * GKD_TOPK)
                else:
                    seq_logprobs.append([lp for _, lp in pos_topk])
                    seq_indices.append([tid for tid, _ in pos_topk])
        batch_logprobs.append(seq_logprobs)
        batch_indices.append(seq_indices)

    # Pad to same seq_len within batch
    max_len = max(len(seq) for seq in batch_logprobs) if batch_logprobs else 1

    for i in range(len(batch_logprobs)):
        pad_len = max_len - len(batch_logprobs[i])
        if pad_len > 0:
            batch_logprobs[i].extend([[0.0] * GKD_TOPK] * pad_len)
            batch_indices[i].extend([[0] * GKD_TOPK] * pad_len)

    # Roll to align with labels (first position has no valid logprobs)
    return {
        'teacher_topk_logprobs': torch.roll(torch.tensor(batch_logprobs, dtype=torch.float32), shifts=-1, dims=1),
        'teacher_topk_indices': torch.roll(torch.tensor(batch_indices, dtype=torch.long), shifts=-1, dims=1),
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
    student_model = MegatronModel(
        model_id=STUDENT_MODEL_ID,
        device_mesh=model_mesh,
        remote_group='student_model',
    )
    student_model.add_adapter_to_model(
        ADAPTER_NAME,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules='all-linear'),
    )
    student_model.set_optimizer('default', lr=LEARNING_RATE)
    student_model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS)
    student_model.set_loss(GKDLoss(beta=GKD_BETA, temperature=GKD_TEMPERATURE))
    student_model.set_template('Template', model_id=STUDENT_MODEL_ID)

    # ── Teacher vLLM sampler (for prompt logprobs) ─────────────────────────────
    teacher_sampler = vLLMSampler(
        model_id=TEACHER_MODEL_ID,
        engine_args={'gpu_memory_utilization': 0.85, 'max_model_len': 10240, 'logprobs_mode': 'raw_logprobs', 'max_logprobs': 64},
        device_mesh=sampler_mesh,
        remote_group='teacher_sampler',
    )
    teacher_sampler.set_template(Template, model_id=TEACHER_MODEL_ID)

    # ── DataLoader (full-text: prompt + reference answer) ──────────────────────
    dataloader = DataLoader(
        dataset=create_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        remote_group='student_model',
    )

    logger.info(f'GKD Off-Policy | student={STUDENT_MODEL_ID}  teacher={TEACHER_MODEL_ID}')
    logger.info(f'  beta={GKD_BETA}  T={GKD_TEMPERATURE}  topk={GKD_TOPK}')

    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        # Teacher vLLM computes top-k prompt logprobs on the reference sequences
        # max_tokens=0: don't generate new content, just compute logprobs on input
        teacher_response = teacher_sampler.sample(
            batch,
            SamplingParams(max_tokens=0, temperature=1.0, prompt_logprobs=GKD_TOPK, num_samples=1),
        )

        input_data = [seq.new_input_feature for response in teacher_response for seq in response.sequences]

        # Convert teacher logprobs to tensor format for GKDLoss
        teacher_output = convert_topk_prompt_logprobs(
            [resp.topk_prompt_logprobs for resp in teacher_response],
        )

        # Student forward + GKD backward
        student_model.forward_backward(inputs=input_data, **teacher_output)
        student_model.clip_grad_and_step()

        if optim_step > 0 and optim_step % 10 == 0:
            metric = student_model.calculate_metric(is_training=True)
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metric}')

        if optim_step > 0 and optim_step % 50 == 0:
            student_model.save(f'gkd-offpolicy-ckpt-{optim_step}')

        optim_step += 1

    student_model.save('gkd-offpolicy-final')
    logger.info('GKD off-policy training completed.')


if __name__ == '__main__':
    main()
