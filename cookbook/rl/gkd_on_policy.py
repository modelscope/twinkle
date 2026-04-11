"""GKD On-Policy Multimodal Distillation via Ray.

On-policy knowledge distillation on OlympiadBench multimodal math/physics:
student vLLM generates responses, teacher vLLM provides top-k prompt logprobs,
then student model learns to match the teacher's token distribution.

Supports three OlympiadBench subsets:
- OE_MM_maths_zh_CEE: Multimodal math problems (Chinese CEE)
- OE_MM_physics_zh_CEE: Multimodal physics problems (Chinese CEE)
- OE_TO_maths_zh_CEE: Text-only math problems (Chinese CEE)

Pipeline:
    1. Sync student model weights to student vLLM sampler.
    2. Student vLLM sampler generates completions on-the-fly.
    3. Teacher vLLM sampler computes top-k prompt logprobs on generated sequences.
    4. Student TransformersModel runs forward_backward() with GKDLoss.

Architecture (Ray):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Driver (CPU)                                                    │
    │  ckpt_manager.sync_weights() ──► sync LoRA to student sampler  │
    │  student_sampler.sample() ──► on-policy completions            │
    │  teacher_sampler.sample(prompt_logprobs=k) ──► teacher lps     │
    │  student_model.forward_backward(teacher_output=...) ──► GKD    │
    └─────────────────────────────────────────────────────────────────┘
         │               │                    │
    DataLoader      vLLMSampler ×2     TransformersModel
                  student + teacher      (model GPUs)

Environment variables (all optional):
    STUDENT_MODEL_ID  – (default: ms://Qwen/Qwen3.5-4B)
    TEACHER_MODEL_ID  – (default: ms://Qwen/Qwen3.5-9B)
    MODEL_GPUS        – GPUs for student model               (default: 4)
    SAMPLER_GPUS      – GPUs for each vLLM sampler           (default: 4)
    MAX_NEW_TOKENS    – max completion tokens                (default: 2048)
    BATCH_SIZE        – global prompt-level batch size       (default: 16)
    MAX_STEPS         – total optimisation steps             (default: 1000)
    LR                – learning rate                        (default: 5e-5)
    N_SAMPLES         – samples per prompt                   (default: 1)
    GKD_BETA          – JSD beta (0=fwd KL, 1=rev KL)        (default: 0.5)
    GKD_TEMPERATURE   – distillation temperature             (default: 1.0)
    GKD_TOPK          – top-k vocab for teacher logprobs     (default: 64)
"""

import os
from typing import List, Optional

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import DatasetMeta, LazyDataset
from twinkle.loss import GKDLoss
from twinkle.model import TransformersModel
from twinkle.preprocessor.olympiad_bench import OlympiadBenchProcessor
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
TEACHER_MODEL_ID = os.environ.get('TEACHER_MODEL_ID', 'ms://Qwen/Qwen3.5-9B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 2))
NUM_GPUS = MODEL_GPUS + 2*SAMPLER_GPUS

MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 2048))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 1000))
LEARNING_RATE = float(os.environ.get('LR', 5e-5))
N_SAMPLES = int(os.environ.get('N_SAMPLES', 1))

GKD_BETA = float(os.environ.get('GKD_BETA', 0.5))
GKD_TEMPERATURE = float(os.environ.get('GKD_TEMPERATURE', 1.0))
GKD_TOPK = int(os.environ.get('GKD_TOPK', 64))
ADAPTER_NAME = 'default'

# OlympiadBench subsets
SUBSETS = [
    'OE_MM_maths_zh_CEE',
    'OE_MM_physics_zh_CEE',
    'OE_TO_maths_zh_CEE',
]


# ── Dataset ───────────────────────────────────────────────────────────────────

def create_dataset():
    """OlympiadBench multimodal dataset; student vLLM will generate completions on-policy."""
    ds = DatasetMeta('ms://AI-ModelScope/OlympiadBench', subset_name=SUBSETS[0], split='train')
    dataset = LazyDataset(ds)
    dataset.map(OlympiadBenchProcessor(language='zh'), dataset_meta=ds)

    for subset in SUBSETS[1:]:
        ds = DatasetMeta('ms://AI-ModelScope/OlympiadBench', subset_name=subset, split='train')
        dataset.add_dataset(ds)
        dataset.map(OlympiadBenchProcessor(language='zh'), dataset_meta=ds)

    dataset.set_template('Qwen3_5Template', model_id=STUDENT_MODEL_ID, max_length=2048, enable_thinking=False)
    dataset.mix_dataset(interleave=True)
    return dataset


# ── Utility ───────────────────────────────────────────────────────────────────

def convert_topk_prompt_logprobs(
    topk_prompt_logprobs_batch: List[List[Optional[List[tuple]]]],
) -> dict:
    """Convert vLLM topk_prompt_logprobs to GKDLoss teacher_output format.

    Args:
        topk_prompt_logprobs_batch: List of per-input topk_prompt_logprobs.
            Each is List[Optional[List[(token_id, logprob)]]] of shape [seq_len, topk].
        device: Target device for tensors.

    Returns:
        Dict with 'topk_logprobs' [batch, seq_len, topk] and
        'topk_indices' [batch, seq_len, topk] tensors.
    """
    batch_logprobs = []
    batch_indices = []

    for seq_topk in topk_prompt_logprobs_batch:
        seq_logprobs = []
        seq_indices = []
        for pos_topk in seq_topk:
            if pos_topk is None:
                seq_logprobs.append([0.0] * len(seq_topk[1]) if len(seq_topk) > 1 and seq_topk[1] else [0.0])
                seq_indices.append([0] * len(seq_topk[1]) if len(seq_topk) > 1 and seq_topk[1] else [0])
            else:
                seq_logprobs.append([lp for _, lp in pos_topk])
                seq_indices.append([tid for tid, _ in pos_topk])
        batch_logprobs.append(seq_logprobs)
        batch_indices.append(seq_indices)

    # Pad to same seq_len within batch
    max_len = max(len(seq) for seq in batch_logprobs)
    topk = GKD_TOPK

    for i in range(len(batch_logprobs)):
        pad_len = max_len - len(batch_logprobs[i])
        if pad_len > 0:
            batch_logprobs[i].extend([[0.0] * topk] * pad_len)
            batch_indices[i].extend([[0] * topk] * pad_len)

    return {
        'teacher_topk_logprobs': torch.roll(torch.tensor(batch_logprobs, dtype=torch.float32), shifts=-1, dims=1),
        'teacher_topk_indices': torch.roll(torch.tensor(batch_indices, dtype=torch.long), shifts=-1, dims=1),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    device_groups = [
        DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='cuda'),
        DeviceGroup(name='student_sampler', ranks=SAMPLER_GPUS, device_type='cuda'),
        DeviceGroup(name='teacher_sampler', ranks=SAMPLER_GPUS, device_type='cuda'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
    )

    # ── Student model (trainable) ──────────────────────────────────────────────
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules='all-linear')

    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        student_model = MegatronModel(
            model_id=STUDENT_MODEL_ID,
            device_mesh=model_mesh,
            remote_group='student_model',
        )
    else:
        from transformers import Qwen3_5ForConditionalGeneration
        student_model = TransformersModel(
            model_id=STUDENT_MODEL_ID,
            model_cls=Qwen3_5ForConditionalGeneration,
            device_mesh=model_mesh,
            remote_group='student_model',
        )

    student_model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)

    if USE_MEGATRON:
        student_model.set_optimizer('default', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
        student_model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    else:
        student_model.set_optimizer('AdamW', lr=LEARNING_RATE)
        student_model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)

    student_model.set_loss(GKDLoss(beta=GKD_BETA, temperature=GKD_TEMPERATURE), adapter_name=ADAPTER_NAME)
    student_model.set_template('Qwen3_5Template', model_id=STUDENT_MODEL_ID, adapter_name=ADAPTER_NAME, enable_thinking=False)

    # ── Student vLLM sampler (for on-policy generation) ────────────────────────
    student_sampler = vLLMSampler(
        model_id=STUDENT_MODEL_ID,
        # enable_lora=True used with ckpt_manager.sync_weights(merge_and_sync=False)
        # meaning only sync lora weights, if merge_and_sync=True,
        # lora will be merged into the base model and sync all weights to vLLM
        engine_args={
            'gpu_memory_utilization': 0.75,
            'max_model_len': 8192,
            'enable_lora': True,
            'max_loras': 1,
            'limit_mm_per_prompt': {'image': 3},
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='student_sampler',
    )
    student_sampler.set_template('Qwen3_5Template', model_id=STUDENT_MODEL_ID, enable_thinking=False)

    # ── Teacher vLLM sampler (for prompt logprobs) ───────────────────────────────
    teacher_sampler = vLLMSampler(
        model_id=TEACHER_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.75,
            'max_model_len': 8192,
            'logprobs_mode': 'raw_logprobs',
            'max_logprobs': 64,
            'limit_mm_per_prompt': {'image': 3},
        },
        device_mesh=sampler_mesh,
        remote_group='teacher_sampler',
    )
    teacher_sampler.set_template('Qwen3_5Template', model_id=TEACHER_MODEL_ID, enable_thinking=False)

    # ── DataLoader (prompt-only) ───────────────────────────────────────────────
    dataloader = DataLoader(
        dataset=create_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='student_model',
    )

    # ── Checkpoint manager for weight sync ──────────────────────────────────────
    ckpt_manager = CheckpointEngineManager(model=student_model, sampler=student_sampler)

    logger.info(get_device_placement())
    logger.info(f'GKD On-Policy | student={STUDENT_MODEL_ID}  teacher={TEACHER_MODEL_ID}')
    logger.info(f'  beta={GKD_BETA}  T={GKD_TEMPERATURE}  topk={GKD_TOPK}')

    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        # 1. Sync student model weights to student sampler
        # enable_lora=True used with ckpt_manager.sync_weights(merge_and_sync=False)
        # meaning only sync lora weights, if merge_and_sync=True,
        # lora will be merged into the base model and sync all weights to vLLM
        ckpt_manager.sync_weights(merge_and_sync=False)
        student_sampler.reset_prefix_cache()

        # 2. Student vLLM generates completions
        sample_response = student_sampler.sample(batch, SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=1.0, num_samples=N_SAMPLES))
        input_data = [seq.new_input_feature for response in sample_response for seq in response.sequences]

        # 3. Teacher vLLM computes top-k prompt logprobs on generated sequences
        teacher_response = teacher_sampler.sample(
            input_data,
            SamplingParams(max_tokens=0, temperature=1.0, prompt_logprobs=GKD_TOPK),
        )

        # 4. Convert teacher logprobs to tensor format for GKDLoss
        # teacher_response is List[SampleResponse], extract topk_prompt_logprobs from each
        teacher_output = convert_topk_prompt_logprobs(
            [resp.topk_prompt_logprobs for resp in teacher_response],
        )

        # 5. Student forward + GKD backward
        student_model.forward_backward(inputs=input_data, adapter_name=ADAPTER_NAME, **teacher_output)
        student_model.clip_grad_and_step(adapter_name=ADAPTER_NAME)

        if optim_step > 0 and optim_step % 1 == 0:
            metric = student_model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME)
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metric}')

        if optim_step > 0 and optim_step % 50 == 0:
            student_model.save(f'gkd-onpolicy-ckpt-{optim_step}', adapter_name=ADAPTER_NAME)

        optim_step += 1

    student_model.save('gkd-onpolicy-final', adapter_name=ADAPTER_NAME)
    logger.info('GKD on-policy training completed.')


if __name__ == '__main__':
    main()
