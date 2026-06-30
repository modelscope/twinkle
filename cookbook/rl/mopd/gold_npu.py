import os
from typing import List

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.loss import GOLDLoss
from twinkle.loss.gold_config import GOLDConfig
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', 'ms://Qwen/Qwen3-0.6B')
TEACHER_MODEL_ID = os.environ.get('TEACHER_MODEL_ID', 'ms://Qwen/qwen2.5-0.5b-instruct')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 1))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 10))
LEARNING_RATE = float(os.environ.get('LR', 1e-4))

GKD_TEMPERATURE = float(os.environ.get('GKD_TEMPERATURE', 1.0))
ADAPTER_NAME = 'default'


# ── NPU Environment Setup ─────────────────────────────────────────────────────

def setup_npu_env():
    """Setup NPU environment variables for Ascend devices."""
    # 1. 设置 NPU 日志输出
    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"

    # 2. 设置正确的 CANN 路径（必须匹配实际安装路径！）
    # 请执行以下命令确认实际路径：
    #   ls /usr/local/Ascend/ascend-toolkit/ | grep -E '5\.1\.RC[0-9]+'
    # 假设输出为 5.1.RC1，则：
    os.environ["ASCEND_AICPU_PATH"] = "/usr/local/Ascend/ascend-toolkit/5.1.RC1"

    # 3. 更新 LD_LIBRARY_PATH（包含 lib64 目录）
    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.environ.get('LD_LIBRARY_PATH', '')}:"
        f"{os.environ['ASCEND_AICPU_PATH']}/lib64"
    )

def convert_topk_prompt_logprobs(
    topk_prompt_logprobs_batch: List,
) -> dict:
    """将 vLLM 的 topk_prompt_logprobs 转换为 GOLDLoss 所需的 teacher_output 格式。

    这个方法的作用是：
    1. vLLM 返回的 teacher_response 中包含 topk_prompt_logprobs，这是一个嵌套列表结构
    2. GOLDLoss 需要的是 teacher_topk_logprobs 和 teacher_topk_indices 两个张量
    3. 本方法将嵌套列表转换为对齐的张量格式，用于知识蒸馏损失计算

    Args:
        topk_prompt_logprobs_batch: 每个输入的 topk_prompt_logprobs 列表。
            每个元素是 List[Optional[List[(token_id, logprob)]]]，
            形状为 [seq_len, topk]，即每个序列位置有 topk 个 (token_id, logprob) 对。

    Returns:
        包含以下键的字典：
        - 'teacher_topk_logprobs': [batch, seq_len, topk] 形状的张量，存储教师模型的 top-k 对数概率
        - 'teacher_topk_indices': [batch, seq_len, topk] 形状的张量，存储对应的 token ID
    """
    batch_logprobs = []  # 存储整个 batch 的对数概率
    batch_indices = []   # 存储整个 batch 的 token 索引

    # 遍历 batch 中的每个序列
    for seq_topk in topk_prompt_logprobs_batch:
        if seq_topk is None:
            # 处理 None 情况（没有可用的 prompt logprobs）
            # 用占位符填充：单个位置，64 个 top-k 值
            seq_logprobs = [[0.0] * 64]
            seq_indices = [[0] * 64]
        else:
            seq_logprobs = []
            seq_indices = []
            # 遍历序列中的每个位置
            for pos_topk in seq_topk:
                if pos_topk is None:
                    # 第一个位置是 None（因为第一个 token 没有前文），用占位符填充
                    seq_logprobs.append([0.0] * 64)
                    seq_indices.append([0] * 64)
                else:
                    # 提取 (token_id, logprob) 对中的 logprob
                    seq_logprobs.append([lp for _, lp in pos_topk])
                    # 提取 (token_id, logprob) 对中的 token_id
                    seq_indices.append([tid for tid, _ in pos_topk])
        batch_logprobs.append(seq_logprobs)
        batch_indices.append(seq_indices)

    # 将 batch 内的所有序列填充到相同的 seq_len
    max_len = max(len(seq) for seq in batch_logprobs) if batch_logprobs else 1
    topk = 64

    # 对较短的序列进行填充
    for i in range(len(batch_logprobs)):
        pad_len = max_len - len(batch_logprobs[i])
        if pad_len > 0:
            # 用 0 填充到最大长度
            batch_logprobs[i].extend([[0.0] * topk] * pad_len)
            batch_indices[i].extend([[0] * topk] * pad_len)

    # 滚动张量以对齐 labels（第一个位置没有有效的 logprobs，需要移到末尾）
    # 这是因为第 i 个位置的 logprobs 是预测第 i+1 个 token 的概率分布
    return {
        'teacher_topk_logprobs': torch.roll(torch.tensor(batch_logprobs, dtype=torch.float32), shifts=-1, dims=1),
        'teacher_topk_indices': torch.roll(torch.tensor(batch_indices, dtype=torch.long), shifts=-1, dims=1),
    }


def train():
    """Main training function for MOPD with GOLDLoss on NPU."""
    # Setup NPU environment
    setup_npu_env()

    # Initialize device groups
    device_groups = [
        DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='npu'),
        DeviceGroup(name='teacher_sampler', ranks=SAMPLER_GPUS, device_type='npu'),
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

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules='all-linear',
    )
    student_model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=2)
    student_model.set_optimizer('AdamW', lr=LEARNING_RATE)
    student_model.set_lr_scheduler('CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=MAX_STEPS)

    # ── Configure GOLDLoss ─────────────────────────────────────────────────────
    gold_config = GOLDConfig(
        # ULD loss configuration
        use_uld_loss=True,
        use_extended_uld=True,
        uld_crossentropy_weight=0.0,  # Set > 0 to add cross-entropy loss
        uld_distillation_weight=1.0,
        uld_student_temperature=GKD_TEMPERATURE,
        uld_teacher_temperature=GKD_TEMPERATURE,
        uld_skip_student_eos=True,
        uld_skip_teacher_eos=True,
        # Hybrid loss (optional, for cross-tokenizer distillation)
        uld_use_hybrid_loss=True,
        beta=0.5,  # JSD interpolation coefficient
    )

    # Get tokenizers for ULD alignment
    from transformers import AutoTokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, trust_remote_code=True)

    loss_fn = GOLDLoss(
        config=gold_config,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )
    student_model.set_loss(loss_fn, adapter_name=ADAPTER_NAME)
    student_model.set_template('Template', model_id=STUDENT_MODEL_ID, adapter_name=ADAPTER_NAME)

    # ── Teacher vLLM sampler (for logits) ──────────────────────────────────────
    teacher_sampler = vLLMSampler(
        model_id=TEACHER_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.65,
            'max_model_len': 4096,
            'logprobs_mode': 'raw_logprobs',
            'max_logprobs': 64,
        },
        device_mesh=sampler_mesh,
        remote_group='teacher_sampler',
    )
    teacher_sampler.set_template('Template', model_id=TEACHER_MODEL_ID)

    # ── DataLoader (full-text: prompt + response) ──────────────────────────────
    dataset = Dataset(
        dataset_meta=DatasetMeta(
            'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji',
            data_slice=range(1000)
        )
    )
    dataset.set_template('Template', model_id=STUDENT_MODEL_ID)
    dataset.encode()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        remote_group='student_model',
    )

    logger.info(f'MOPD NPU Training | student={STUDENT_MODEL_ID}  teacher={TEACHER_MODEL_ID}')
    logger.info(f'  T={GKD_TEMPERATURE}  batch_size={BATCH_SIZE}  lr={LEARNING_RATE}')

    # ── Training Loop ──────────────────────────────────────────────────────────
    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        # 1. Teacher computes logits on the full sequences
        # max_tokens=0: don't generate new content, just compute logits on input
        # Use prompt_logprobs to get teacher logprobs for distillation

        teacher_response = teacher_sampler.sample(
            batch,
            SamplingParams(max_tokens=0, temperature=1.0, prompt_logprobs=64),
        )

        # 2. Convert teacher response to input format
        input_data = [seq.new_input_feature for resp in teacher_response for seq in resp.sequences]

        # 3. Prepare teacher output for GOLDLoss
        # GOLDLoss expects teacher_logits, teacher_labels, and teacher_input_ids
        # Convert topk_prompt_logprobs to the format expected by GOLDLoss

        # Convert topk_prompt_logprobs to teacher_logits format
        topk_data = convert_topk_prompt_logprobs(
            [resp.topk_prompt_logprobs for resp in teacher_response]
        )

        # Get teacher input_ids and labels from input_data
        # Convert lists to tensors first
        # Handle variable sequence lengths by padding to the maximum length
        import torch.nn.utils.rnn as rnn_utils

        # Get all input_ids and labels
        input_ids_list = [torch.tensor(item['input_ids']) for item in input_data]
        labels_list = [torch.tensor(item['labels']) for item in input_data]

        # Pad sequences to the same length
        teacher_input_ids = rnn_utils.pad_sequence(input_ids_list, batch_first=True)
        teacher_labels = rnn_utils.pad_sequence(labels_list, batch_first=True, padding_value=-100)

        # Create teacher_output dict with topk format
        # GOLDLoss has been modified to support topk format
        teacher_output = {
            'teacher_labels': teacher_labels,
            'teacher_input_ids': teacher_input_ids,
            'teacher_topk_logprobs': topk_data['teacher_topk_logprobs'],
            'teacher_topk_indices': topk_data['teacher_topk_indices'],
        }

        # 4. Student forward + GOLD backward
        student_model.forward_backward(
            inputs=input_data,
            adapter_name=ADAPTER_NAME,
            return_logits=True,
            **teacher_output,
        )
        student_model.clip_grad_and_step(adapter_name=ADAPTER_NAME)

        # 5. Logging
        if optim_step > 0 and optim_step % 20 == 0:
            metric = student_model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME)
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metric}')

        # 6. Checkpoint
        if optim_step > 0 and optim_step % 100 == 0:
            student_model.save(f'mopd-npu-ckpt-{optim_step}', adapter_name=ADAPTER_NAME)

        optim_step += 1

    # Save final checkpoint
    student_model.save('mopd-npu-final', adapter_name=ADAPTER_NAME)
    logger.info('MOPD NPU training completed.')


if __name__ == '__main__':
    train()