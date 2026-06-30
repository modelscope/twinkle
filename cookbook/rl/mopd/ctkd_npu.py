import os
from typing import List, Optional

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CTKDLoss
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', 'ms://Qwen/qwen3.5-2B')
TEACHER_MODEL_ID = os.environ.get('TEACHER_MODEL_ID', 'ms://Qwen/Qwen3-0.6B')
TEACHER_MODEL_ID_2 = os.environ.get('TEACHER_MODEL_ID_2', 'ms://Qwen/qwen2.5-0.5b-instruct')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 1))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))
SHARED_TEACHER_GPUS = bool(os.environ.get('SHARED_TEACHER_GPUS', False))
NUM_GPUS = MODEL_GPUS + (SAMPLER_GPUS if SHARED_TEACHER_GPUS else SAMPLER_GPUS * 2)
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 10))
LEARNING_RATE = float(os.environ.get('LR', 1e-4))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 2))

CTKD_TEMPERATURE = float(os.environ.get('CTKD_TEMPERATURE', 1.0))
CTKD_MAX_LENGTH = int(os.environ.get('CTKD_MAX_LENGTH', 4))
CTKD_BETA = float(os.environ.get('CTKD_BETA', 0.9))
CTKD_GAMMA = float(os.environ.get('CTKD_GAMMA', 0.1))
CTKD_LOSS_TYPE = os.environ.get('CTKD_LOSS_TYPE', 'pkl')  # 'pkl' or 'hkl'
CTKD_TOPK = int(os.environ.get('CTKD_TOPK', 64))
ADAPTER_NAME = 'default'
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', 2048))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 2048))
N_SAMPLES = int(os.environ.get('N_SAMPLES', 1))
SHARED_TEACHER_GPUS = bool(os.environ.get('SHARED_TEACHER_GPUS', False))
# ── Utility ───────────────────────────────────────────────────────────────────

def convert_topk_prompt_logprobs(
    topk_prompt_logprobs_batch: List[List[Optional[List[tuple]]]],
    topk: int = 64,
) -> dict:
    """Convert vLLM topk_prompt_logprobs to CTKDLoss teacher_output format.

    Args:
        topk_prompt_logprobs_batch: List of per-input topk_prompt_logprobs.
            Each is List[Optional[List[(token_id, logprob)]]] of shape [seq_len, topk].
        topk: Number of top-k logits to extract.

    Returns:
        Dict with 'teacher_topk_logprobs' [batch, seq_len, topk] and
        'teacher_topk_indices' [batch, seq_len, topk] tensors.
    """
    batch_logprobs = []
    batch_indices = []

    for seq_topk in topk_prompt_logprobs_batch:
        seq_logprobs = []
        seq_indices = []
        for pos_topk in seq_topk:
            if pos_topk is None:
                # First position is None, fill with placeholder
                seq_logprobs.append([0.0] * topk)
                seq_indices.append([0] * topk)
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
            batch_logprobs[i].extend([[0.0] * topk] * pad_len)
            batch_indices[i].extend([[0] * topk] * pad_len)

    # Roll to align with labels (first position has no valid logprobs)
    return {
        'teacher_topk_logprobs': torch.roll(torch.tensor(batch_logprobs, dtype=torch.float32), shifts=-1, dims=1),
        'teacher_topk_indices': torch.roll(torch.tensor(batch_indices, dtype=torch.long), shifts=-1, dims=1),
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

def create_dataset():
    """创建用于蒸馏的全文(prompt + response)数据集。

    数据集使用 student tokenizer 编码。Teacher 会将文本解码后，
    使用自己的 tokenizer 重新编码，以实现跨 tokenizer 知识蒸馏。
    """
    dataset = Dataset(DatasetMeta(DATASET_ID, data_slice=range(10000)))
    dataset.set_template('Template', model_id=STUDENT_MODEL_ID, max_length=MAX_LENGTH)
    dataset.encode(load_from_cache_file=True)
    return dataset
def train():
    """Main training function for MOPD with CTKDLoss."""
    # Initialize device groups based on shared mode
    if SHARED_TEACHER_GPUS:
        # Shared mode: both teacher samplers use the same GPU resources
        device_groups = [
            DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='npu'),
            DeviceGroup(name='teacher_sampler', ranks=SAMPLER_GPUS, device_type='npu'),
        ]
    else:
        # Separate mode: each teacher sampler gets its own GPU resources
        device_groups = [
            DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='npu'),
            DeviceGroup(name='teacher_sampler', ranks=SAMPLER_GPUS, device_type='npu'),
            DeviceGroup(name='teacher_sampler_2', ranks=SAMPLER_GPUS, device_type='npu'),
        ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
    )

    # ── Student model (trainable) ──────────────────────────────────────────────
    student_model = TransformersModel(
        model_id=STUDENT_MODEL_ID,
        device_mesh=model_mesh,
        remote_group='student_model',
    )

    # LoRA configuration for efficient fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules='all-linear',
    )
    student_model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    student_model.set_optimizer('AdamW', lr=LEARNING_RATE, weight_decay=0.01)
    student_model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=LEARNING_RATE * 0.1)

# ── Configure CTKDLoss ─────────────────────────────────────────────────────
    # Get tokenizers for cross-tokenizer alignment
    from transformers import AutoTokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, trust_remote_code=True)
    teacher_tokenizer_2 = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID_2, trust_remote_code=True)

    loss_fn = CTKDLoss(
        student_tokenizer=student_tokenizer,
        teacher_tokenizer_group=[teacher_tokenizer, teacher_tokenizer_2],
        max_length=CTKD_MAX_LENGTH,
        beta=CTKD_BETA,
        gamma=CTKD_GAMMA,
        loss_type=CTKD_LOSS_TYPE,
        temperature=CTKD_TEMPERATURE,
    )
    student_model.set_loss(loss_fn, adapter_name=ADAPTER_NAME)
    student_model.set_template('QwenTemplate', model_id=STUDENT_MODEL_ID, adapter_name=ADAPTER_NAME)

    # Log configuration
    logger.info(f'GPU Configuration: MODEL_GPUS={MODEL_GPUS}, SAMPLER_GPUS={SAMPLER_GPUS}, SHARED_TEACHER_GPUS={SHARED_TEACHER_GPUS}')
    logger.info(f'Total GPUs required: {NUM_GPUS}')

    # Log projection matrix statistics
    stats = loss_fn.get_mapping_statistics()
    logger.info(f'CTKD Projection Matrix Statistics: {stats}')

    #── Teacher vLLM samplers (for logits) ─────────────────────────────────────
    if SHARED_TEACHER_GPUS:
        # Shared mode: use different instance_id to avoid naming conflicts
        teacher_sampler = vLLMSampler(
            model_id=TEACHER_MODEL_ID,
            engine_args={
                'gpu_memory_utilization': 0.75,
                'max_model_len': 4096,
                'logprobs_mode': 'raw_logprobs',
                'max_logprobs': CTKD_TOPK,
            },
            device_mesh=sampler_mesh,
            remote_group='teacher_sampler',
            instance_id='teacher_1'
        )
        teacher_sampler.set_template('QwenTemplate', model_id=TEACHER_MODEL_ID)

        teacher_sampler_2 = vLLMSampler(
            model_id=TEACHER_MODEL_ID_2,
            engine_args={
                'gpu_memory_utilization': 0.75,
                'max_model_len': 4096,
                'logprobs_mode': 'raw_logprobs',
                'max_logprobs': CTKD_TOPK,
            },
            device_mesh=sampler_mesh,
            remote_group='teacher_sampler',  # Same remote_group but different instance_id
            instance_id='teacher_2'
        )
        teacher_sampler_2.set_template('QwenTemplate', model_id=TEACHER_MODEL_ID_2)
    else:
        # Separate mode: each teacher sampler has its own GPU resources
        teacher_sampler = vLLMSampler(
            model_id=TEACHER_MODEL_ID,
            engine_args={
                'gpu_memory_utilization': 0.75,
                'max_model_len': 4096,
                'logprobs_mode': 'raw_logprobs',
                'max_logprobs': CTKD_TOPK,
            },
            device_mesh=sampler_mesh,
            remote_group='teacher_sampler',
        )
        teacher_sampler.set_template('QwenTemplate', model_id=TEACHER_MODEL_ID)

        teacher_sampler_2 = vLLMSampler(
            model_id=TEACHER_MODEL_ID_2,
            engine_args={
                'gpu_memory_utilization': 0.75,
                'max_model_len': 4096,
                'logprobs_mode': 'raw_logprobs',
                'max_logprobs': CTKD_TOPK,
            },
            device_mesh=sampler_mesh,
            remote_group='teacher_sampler_2',
        )
        teacher_sampler_2.set_template('QwenTemplate', model_id=TEACHER_MODEL_ID_2)

    # ── DataLoader (full-text: prompt + response) ──────────────────────────────
    # Dataset is pre-encoded with student tokenizer
    # Teacher will decode text and re-encode with its own tokenizer
    dataloader = DataLoader(
        dataset=create_dataset(),  # 调用函数获取数据集对象
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='student_model',
    )

    logger.info(get_device_placement())
    logger.info(f'MOPD CTKD Training | student={STUDENT_MODEL_ID}  teachers={TEACHER_MODEL_ID}, {TEACHER_MODEL_ID_2}')
    logger.info(f'  T={CTKD_TEMPERATURE}  loss_type={CTKD_LOSS_TYPE}  topk={CTKD_TOPK}')
    logger.info(f'  batch_size={BATCH_SIZE}  lr={LEARNING_RATE}  max_steps={MAX_STEPS}')

    # ── Training Loop ──────────────────────────────────────────────────────────
    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        # 在 Ray 模式下，batch 是一个可调用对象，需要调用它来获取实际数据
        if callable(batch):
            batch = batch()

        # 1. Decode student-encoded tokens back to text for teacher
        # batch contains input_ids encoded with student tokenizer
        # We need to decode them and re-encode with teacher tokenizer
        from twinkle.data_format import Trajectory

        teacher_inputs = []
        for item in batch:
            # Decode student tokens back to text, then re-encode with teacher tokenizer
            text = student_tokenizer.decode(item['input_ids'], skip_special_tokens=False)
            teacher_encoded = teacher_tokenizer.encode(text, add_special_tokens=False)
            teacher_inputs.append({
                'input_ids': teacher_encoded,
                'labels': item.get('labels', [-100] * len(teacher_encoded)),
            })

        # 2. Teachers compute top-k logprobs on the full sequences
        # max_tokens=0: don't generate new content, just compute logits on input
        teacher_response = teacher_sampler.sample(
            teacher_inputs,  # Trajectory format - teacher will encode with its tokenizer
            SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=CTKD_TOPK),  # max_tokens=1 for logprobs only
        )

        teacher_response_2 = teacher_sampler_2.sample(
            teacher_inputs,  # Same input for second teacher
            SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=CTKD_TOPK),
        )

        # 3. Convert teacher responses to input format (already encoded with teacher tokenizers)
        teacher_input_data = [seq.new_input_feature for resp in teacher_response for seq in resp.sequences]
        teacher_input_data_2 = [seq.new_input_feature for resp in teacher_response_2 for seq in resp.sequences]

        # 4. Prepare teacher output for CTKDLoss (for both teachers)
        topk_data = convert_topk_prompt_logprobs(
            [resp.topk_prompt_logprobs for resp in teacher_response],
            topk=CTKD_TOPK,
        )

        topk_data_2 = convert_topk_prompt_logprobs(
            [resp.topk_prompt_logprobs for resp in teacher_response_2],
            topk=CTKD_TOPK,
        )

        # Get teacher input_ids and labels from teacher_input_data
        import torch.nn.utils.rnn as rnn_utils
        teacher_input_ids_list = [torch.tensor(item['input_ids']) for item in teacher_input_data]
        teacher_labels_list = [torch.tensor(item['labels']) for item in teacher_input_data]

        teacher_input_ids_list_2 = [torch.tensor(item['input_ids']) for item in teacher_input_data_2]
        teacher_labels_list_2 = [torch.tensor(item['labels']) for item in teacher_input_data_2]

        # Pad sequences to the same length
        teacher_input_ids = rnn_utils.pad_sequence(teacher_input_ids_list, batch_first=True)
        teacher_labels = rnn_utils.pad_sequence(teacher_labels_list, batch_first=True, padding_value=-100)

        teacher_input_ids_2 = rnn_utils.pad_sequence(teacher_input_ids_list_2, batch_first=True)
        teacher_labels_2 = rnn_utils.pad_sequence(teacher_labels_list_2, batch_first=True, padding_value=-100)

        # Create teacher_output dict for both teachers
        teacher_output = {
            'teacher_labels': [teacher_labels, teacher_labels_2],
            'teacher_input_ids': [teacher_input_ids, teacher_input_ids_2],
            'teacher_topk_logprobs_group': [topk_data['teacher_topk_logprobs'], topk_data_2['teacher_topk_logprobs']],
            'teacher_topk_indices_group': [topk_data['teacher_topk_indices'], topk_data_2['teacher_topk_indices']],
        }

        # 5. Student forward + CTKD backward
        # batch is already encoded with student tokenizer
        student_model.forward_backward(
            inputs=batch,
            adapter_name=ADAPTER_NAME,
            return_logits=True,
            **teacher_output,
        )
        student_model.clip_grad_and_step(adapter_name=ADAPTER_NAME)

        # 5. Logging
        if optim_step > 0 and optim_step % 10 == 0:
            metric = student_model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME)
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metric}')

        # 6. Checkpoint
        if optim_step > 0 and optim_step % 100 == 0:
            student_model.save(f'mopd-ctkd-ckpt-{optim_step}', adapter_name=ADAPTER_NAME)

        optim_step += 1

    # Save final checkpoint
    student_model.save('mopd-ctkd-final', adapter_name=ADAPTER_NAME)
    logger.info('MOPD CTKD training completed.')


if __name__ == '__main__':
    train()