"""Cross-Token Knowledge Distillation Training (On-Policy, Two Teachers).

This script implements the on-policy X-Token training pipeline using
CrossTokenLoss with multiple teachers. The student model generates
completions on-the-fly, and each teacher distills its knowledge on the
student's own output distribution — matching the on-policy distillation
paradigm. Teacher losses are weighted and summed inside CrossTokenLoss.

This is the two-teacher counterpart of ``cross_token_trainer_on_policy.py``
(single-teacher). The teacher list is configurable via ``TEACHER_MODEL_IDS``
(comma-separated), so the same script also supports N teachers.

Reference:
    "X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation"
    (https://arxiv.org/pdf/2605.21699)
    "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
    (https://arxiv.org/abs/2306.13649)

On-Policy Pipeline:
    1. Sync student model weights to student vLLM sampler.
    2. Student vLLM sampler generates completions on-the-fly.
    3. Decode student-generated text, re-encode with each teacher tokenizer.
    4. Each teacher TransformersModel computes full logits.
    5. Student TransformersModel runs forward_backward() with CrossTokenLoss
       on its own generated data (per-teacher KD weighted & summed).

Architecture (Ray):
    +-----------------------------------------------------------------+
    | Driver (CPU)                                                    |
    |  ckpt_manager.sync_weights() --> sync LoRA to student sampler   |
    |  student_vllm.sample() --> on-policy completions                |
    |  decode-re-encode for each teacher tokenizer                    |
    |  teacher_i.forward_only() --> teacher_i full logits             |
    |  student.forward_backward(teacher_output=...) --> Loss          |
    +-----------------------------------------------------------------+
          |               |                    |
     DataLoader      vLLMSampler         TransformersModel x3
                   (student gen)       (student + 2 teachers)

Environment variables (all optional):
    STUDENT_MODEL_ID          – Student model path
    TEACHER_MODEL_IDS         – Comma-separated teacher model paths
                                (default: Llama-3.2-3B-Instruct,Qwen2.5-7B-Instruct)
    TEACHER_GPUS              – Comma-separated GPUs per teacher
                                (default: 1,1)
    TEACHER_WEIGHTS           – Comma-separated per-teacher distillation weights
                                (default: 0.2,0.8 for 2 teachers; equal otherwise)
    DATASET_ID                – Prompt-only dataset (JSONL file)
    MODEL_GPUS                – GPUs for student model (default: 1)
    STUDENT_SAMPLER_GPUS      – GPUs for student vLLM sampler (default: 1)
    DEVICE_IDS                – Comma-separated physical device ids, e.g. 0,2,4,6.
                                Ranks are mapped to these cards in order
                                (groups take the first MODEL_GPUS / next
                                STUDENT_SAMPLER_GPUS / remaining TEACHER_GPUS ids).
    BATCH_SIZE                – Global batch size (default: 2)
    MAX_STEPS                 – Total optimisation steps (default: 5)
    LR                        – Learning rate (default: 1e-5)
    GRADIENT_ACCUMULATION_STEPS – Gradient accumulation (default: 4)
    LOSS_TYPE                 – 'pkl' or 'hkl' (default: 'pkl')
    TEMPERATURE               – Distillation temperature (default: 1.2)
    KL_LOSS_WEIGHT            – Weight for KD loss (default: 0.5)
    CE_LOSS_WEIGHT            – Weight for CE loss (default: 1.0)
    MAX_LENGTH                – Max span length for multi-token (default: 4)
    BETA                      – Base weight for projection (default: 0.95)
    GAMMA                     – Decay rate for multi-token weights (default: 0.1)
    GAMMA_KL                  – Weight for common-KL in H-KL (default: 0.5)
    GAMMA_ULD                 – Weight for ULD in H-KL (default: 0.0)
    VOCAB_TOPK                – Top-k vocab for P-KL subset (default: 64)
    UNCOMMON_TOPK             – Top-k for uncommon L1 in H-KL (default: 8192)
    MAX_LENGTH_SEQ            – Max sequence length for dataset (default: 2048)
    MAX_NEW_TOKENS            – Max tokens for student generation (default: 2048)
    N_SAMPLES                 – Number of samples per prompt (default: 1)
    EVAL_ENABLED              – Enable fixed-prompt greedy eval snapshots (default: true)
    EVAL_INTERVAL             – Eval every N optimisation steps (default: 50; 0=off)
    EVAL_MAX_NEW_TOKENS       – Max tokens for eval generation (default: 512)
    EVAL_PROMPTS              – Hand-written eval prompts, one per line
    EVAL_N_TRAIN              – Eval prompts sampled from training JSONL (default: 2)
    EVAL_TRAIN_OFFSET         – Skip first N training rows when sampling (default: 10000)
    EVAL_LOG_FILE             – Eval snapshot log file (default: eval_log.txt)
"""

import os

import torch
import torch.nn.utils.rnn as rnn_utils
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CrossTokenLoss
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', '/nas/disk1/Qwen3-1.7B')
# 教师为逗号分隔列表(支持 N 个教师);每个教师独立 tokenizer/投影矩阵/权重。
# 默认教师1 保留单教师脚本的 Llama,教师2 选异词表 Qwen2.5(与默认学生 Qwen3
# 同族)——一异一同覆盖两种蒸馏场景。
TEACHER_MODEL_IDS = [
    p.strip() for p in os.environ.get(
        'TEACHER_MODEL_IDS',
        'ms://Llama-3.2-3B-Instruct,ms://Qwen3.5-4B',
    ).split(',')
    if p.strip()
]
# 每教师卡数(逗号分隔,与 TEACHER_MODEL_IDS 一一对应)
TEACHER_GPUS = [
    int(g) for g in os.environ.get('TEACHER_GPUS', '1,1').split(',')
    if g.strip()
]
if len(TEACHER_MODEL_IDS) != len(TEACHER_GPUS):
    raise ValueError(
        f'TEACHER_MODEL_IDS ({len(TEACHER_MODEL_IDS)} teachers) must match '
        f'TEACHER_GPUS ({len(TEACHER_GPUS)} entries)'
    )
NUM_TEACHERS = len(TEACHER_MODEL_IDS)
# 每教师蒸馏权重(逗号分隔,可选)。未设置时默认 2 教师为 0.2/0.8:
# 多教师实验结论——Qwen3.5-4B 对中文响应的响应区打分系统性优于
# Llama-3.2-3B(响应区 mean logp -2.21 vs -2.89),弱教师(Llama)降权;
# 其他教师数量回退等权(CrossTokenLoss 归一化)。
_weights_str = os.environ.get('TEACHER_WEIGHTS', '').strip()
if _weights_str:
    TEACHER_WEIGHTS = [
        float(w) for w in _weights_str.split(',') if w.strip()
    ]
else:
    TEACHER_WEIGHTS = [0.2, 0.8] if NUM_TEACHERS == 2 else None
if TEACHER_WEIGHTS is not None and len(TEACHER_WEIGHTS) != NUM_TEACHERS:
    raise ValueError(
        f'TEACHER_WEIGHTS ({len(TEACHER_WEIGHTS)} entries) must match '
        f'number of teachers ({NUM_TEACHERS})'
    )
DATASET_ID = os.environ.get(
    'DATASET_ID', 'ms://modelscope/gsm8k'
)

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 1))
STUDENT_SAMPLER_GPUS = int(os.environ.get('STUDENT_SAMPLER_GPUS', 1))
NUM_GPUS = MODEL_GPUS + STUDENT_SAMPLER_GPUS + sum(TEACHER_GPUS)

# Optional: specify physical device ids (e.g. DEVICE_IDS=0,2,4,6). Must be set
# before twinkle.initialize() so the Ray workers inherit the platform's
# visible-devices env var (ASCEND_RT_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES),
# which maps local rank -> physical card.
DEVICE_IDS = os.environ.get('DEVICE_IDS', '13，14，15，16')
if DEVICE_IDS:
    # 兼容中文全角逗号与空格（如 DEVICE_IDS=4，6，7），并尽早校验格式
    DEVICE_IDS = DEVICE_IDS.replace('，', ',').replace(' ', '')
    try:
        [int(d) for d in DEVICE_IDS.split(',') if d]
    except ValueError as exc:
        raise ValueError(
            f'Invalid DEVICE_IDS={DEVICE_IDS!r}: must be comma-separated integers'
        ) from exc
    from twinkle.utils.platforms import Platform

    visible_env = Platform.get_platform().visible_device_env()
    old = os.environ.get(visible_env)
    if old and old != DEVICE_IDS:
        logger.warning(
            f'Overriding {visible_env}={old!r} with DEVICE_IDS={DEVICE_IDS!r}'
        )
    os.environ[visible_env] = DEVICE_IDS

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 2))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 100))
LEARNING_RATE = float(os.environ.get('LR', 2e-5))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 4))

LOSS_TYPE = os.environ.get('LOSS_TYPE', 'hkl')
# 蒸馏温度:0.8 → 1.2(软化教师分布)。教师(Llama-3.2-Instruct)的高概率区
# 含公式化拒绝文本("I'm sorry, I can't...")——低温时 KD 把学生拉向该
# 区域(reward hacking:eval 中 Q4/Q7/Q8 学成拒绝回答、teacher_score 反升,
# 而 Q1 实质回答退化 -0.48)。升温削弱公式化区的相对优势;与
# KL_LOSS_WEIGHT=0.5 组合后有效 KD 强度 = 0.5×KL×1.2² ≈ 0.72×KL,
# 与原先 1.0×KL×0.8² ≈ 0.64×KL 基本持平——只软化分布,不减弱整体信号。
TEMPERATURE = float(os.environ.get('TEMPERATURE', 1.2))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', 4))
BETA = float(os.environ.get('BETA', 0.95))
GAMMA = float(os.environ.get('GAMMA', 0.1))
# H-KL 权重按 NeMo 对齐:NeMo 的 H-KL 只有 common-KL + uncommon-L1
# (无 ULD/熵项),gamma_kl=0.5、gamma_uld=0 → gamma_l1 = 1-0.5-0 = 0.5,
# KL/L1 等权。
GAMMA_KL = float(os.environ.get('GAMMA_KL', 0.5))
GAMMA_ULD = float(os.environ.get('GAMMA_ULD', 0.0))
VOCAB_TOPK = int(os.environ.get('VOCAB_TOPK', 512))
UNCOMMON_TOPK = int(os.environ.get('UNCOMMON_TOPK', 8192))
MAX_LENGTH_SEQ = int(os.environ.get('MAX_LENGTH_SEQ', 2048))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 2048))
N_SAMPLES = int(os.environ.get('N_SAMPLES', 1))
# KD 权重 1.0 → 0.5:CE 项(学生对自己生成 token 的 CE,weight=1)锚定
# 学生自身分布;KD 权重减半以减缓被教师公式化区域(拒绝/模板文本)劫持
# 的漂移速度。与 TEMPERATURE=1.2 组合后有效 KD 强度基本不变(见上)。
KL_LOSS_WEIGHT = float(os.environ.get('KL_LOSS_WEIGHT', 0.5))
CE_LOSS_WEIGHT = float(os.environ.get('CE_LOSS_WEIGHT', 1.0))
DYNAMIC_LOSS_SCALING = os.environ.get(
    'DYNAMIC_LOSS_SCALING', 'false'
).lower() in ('true', '1', 'yes')
# KL 方向:默认 true = 正向 KL(学生‖教师)(Σ p·ln(p/q),对教师低分有
# 区分度)——反向 KL(教师‖学生)在精确投影 + 教师分布散落时被 eps 地板
# 放大到 20+,无区分度。NeMo 用反向但配合软投影,我们场景必须正向。
REVERSE_KL = os.environ.get('REVERSE_KL', 'true').lower() in ('true', '1', 'yes')
# 短生成过滤:生成 token 数 < MIN_GEN_TOKENS 的样本跳过训练。
# 短生成(1-2 个模板 token)产生无信号噪声步(proj_mass=0 → KD/CE≈0 微负),
# 也是学生"提前停止"逃避教师评价的路径——过滤后该行为不再获得训练信号。
MIN_GEN_TOKENS = int(os.environ.get('MIN_GEN_TOKENS', 5))
# 生成侧强制下限(vLLM min_tokens):生成至少 N 个 token 后才允许 EOS/停止。
# stop-hacking 的生成侧对策——从源头阻止"早停",不依赖 loss 侧干预。
# 阈值是"地板"不是"目标":30 低于数据集中最短合理响应(标题类 40-60),
# 只挡"逃避式"早停,不扭曲正常任务;任务想写长就长,由模型自己决定。
MIN_NEW_TOKENS = int(os.environ.get('MIN_NEW_TOKENS', 50))

ADAPTER_NAME = 'default'


# ── Dataset ───────────────────────────────────────────────────────────────────

def create_dataset():
    """Create a prompt-only dataset for on-policy distillation.

    The dataset only contains prompts; the student model generates completions
    on-the-fly. The teacher model computes logprobs on student-generated
    sequences.
    """
    dataset = Dataset(DatasetMeta(DATASET_ID, data_slice=range(10000)))
    dataset.set_template(
        'Template', model_id=STUDENT_MODEL_ID, max_length=MAX_LENGTH_SEQ,
        enable_thinking=False,
    )
    dataset.encode(load_from_cache_file=True)
    return dataset


# ── Utility ───────────────────────────────────────────────────────────────────

def prepare_teacher_inputs_from_student_gen(
        sample_response: list,
        student_tokenizer,
        teacher_tokenizer,
) -> list:
    """Decode student-generated sequences and re-encode for a teacher.

    Args:
        sample_response: List of vLLM response objects from student_sampler.sample().
        student_tokenizer: Student model tokenizer (for decoding).
        teacher_tokenizer: One teacher model tokenizer (for re-encoding).

    Returns:
        List of new_input_feature dicts with teacher-tokenized 'input_ids'.
    """
    teacher_inputs = []
    for resp in sample_response:
        for seq in resp.sequences:
            full_ids = seq.new_input_feature['input_ids']
            text = student_tokenizer.decode(full_ids, skip_special_tokens=False)

            # 不加特殊 token:教师序列必须与学生的字符跨度一一对应。
            # add_special_tokens=True 会给 Llama 加 <|begin_of_text|>(15 字符),
            # 教师所有字符跨度整体偏移 → chunk 对齐系统性错位
            # (学生响应区对齐到教师模板区,KL 触底)。
            teacher_ids = teacher_tokenizer.encode(text, add_special_tokens=False)
            teacher_ids = teacher_ids[:MAX_LENGTH_SEQ]

            teacher_inputs.append({
                'input_ids': teacher_ids,
                'labels': teacher_ids[:],
            })
    return teacher_inputs


def _as_list(ids):
    """Convert token ids (tensor / list / array) to a plain Python list."""
    if isinstance(ids, torch.Tensor):
        return ids.tolist()
    return list(ids)


def _to_py_list(obj):
    """递归把 ndarray/np 标量转成 Python 原生类型。

    vLLM v1 的 input_processor 对 prompt_ids 做 truthiness 判断
    (``if prompt_ids and ...``),ndarray 会抛 ValueError。训练 batch 经
    HF datasets 读取时 input_ids 已是 list,故训练路径正常;eval 直接
    ``template.encode()`` 得到的是 ndarray,喂给 sample() 前必须转换。
    """
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_py_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py_list(v) for v in obj]
    return obj


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    """Main training loop for on-policy cross-tokenizer distillation (multi-teacher)."""

    # ── Initialize device groups ──────────────────────────────────────────────
    device_groups = [
        DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='npu'),
        DeviceGroup(
            name='student_sampler', ranks=STUDENT_SAMPLER_GPUS, device_type='npu',
        ),
    ]
    # 每个教师独立 DeviceGroup / DeviceMesh
    device_groups += [
        DeviceGroup(name=f'teacher_{i}', ranks=gpus, device_type='npu')
        for i, gpus in enumerate(TEACHER_GPUS)
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=STUDENT_SAMPLER_GPUS, dp_size=STUDENT_SAMPLER_GPUS,
    )
    teacher_meshes = [
        DeviceMesh.from_sizes(world_size=gpus, dp_size=gpus)
        for gpus in TEACHER_GPUS
    ]

    # Ray 不会自动检测昇腾 NPU 资源：ResourceManager 要求节点上报
    # NPU 资源数 >= nproc_per_node，否则报
    # "Not enough resources, required nodes: 1, available: 0"。
    # 本地启动 Ray 时显式声明 NPU 自定义资源；部分 Ray 版本会忽略
    # ray.init(resources=...)，此时自动退化到 `ray start --resources` 重启。
    # 若通过 RAY_ADDRESS 连接外部集群，请在 `ray start` 时声明
    # --resources='{"NPU": <卡数>}'。
    import ray
    if not os.environ.get('RAY_ADDRESS'):
        ray.init(resources={'NPU': NUM_GPUS}, ignore_reinit_error=True)
        have_npu = float(ray.cluster_resources().get('NPU', 0))
        if have_npu < NUM_GPUS:
            logger.warning(
                f'ray.init(resources=...) 未生效（NPU: {have_npu:.0f}/{NUM_GPUS}），'
                '重启本地 Ray 并用 --resources 声明 NPU 资源'
            )
            import subprocess
            subprocess.run(['ray', 'stop'], capture_output=True)
            subprocess.run(
                ['ray', 'start', '--head', '--num-gpus=0',
                 f'--resources={{"NPU": {NUM_GPUS}}}', '--disable-usage-stats',
                 '--include-dashboard=false'],
                check=True,
            )
            ray.init(address='auto', ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)
        have_npu = float(ray.cluster_resources().get('NPU', 0))
        if have_npu < NUM_GPUS:
            raise RuntimeError(
                f'Ray 集群（RAY_ADDRESS={os.environ["RAY_ADDRESS"]}）未声明 '
                f'NPU 资源（NPU: {have_npu:.0f}，需要 {NUM_GPUS}）。请重启集群：\n'
                f'  ray stop && ray start --head --num-gpus=0 '
                f'--resources=\'{{"NPU": {NUM_GPUS}}}\''
            )

    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
    )

    # ── Student model (trainable) ─────────────────────────────────────────────
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
    student_model.add_adapter_to_model(
        ADAPTER_NAME, lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    student_model.set_optimizer('AdamW', lr=LEARNING_RATE, weight_decay=0.01)
    student_model.set_lr_scheduler(
        'CosineAnnealingLR', T_max=MAX_STEPS, eta_min=LEARNING_RATE * 0.1,
    )

    # ── Configure CrossTokenLoss ──────────────────────────────────────────────
    from transformers import AutoTokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL_ID, trust_remote_code=True,
    )
    teacher_tokenizers = [
        AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
        for tid in TEACHER_MODEL_IDS
    ]

    loss_fn = CrossTokenLoss(
        student_tokenizer=student_tokenizer,
        teacher_tokenizer_group=teacher_tokenizers,
        teacher_weights=TEACHER_WEIGHTS,
        max_length=MAX_LENGTH,
        beta=BETA,
        gamma=GAMMA,
        loss_type=LOSS_TYPE,
        temperature=TEMPERATURE,
        gamma_kl=GAMMA_KL,
        gamma_uld=GAMMA_ULD,
        vocab_topk=VOCAB_TOPK,
        uncommon_topk=UNCOMMON_TOPK,
        reverse_kl=REVERSE_KL,
        kl_loss_weight=KL_LOSS_WEIGHT,
        ce_loss_weight=CE_LOSS_WEIGHT,
        dynamic_loss_scaling=DYNAMIC_LOSS_SCALING,
        device=torch.device('npu:0'),
    )
    student_model.set_loss(loss_fn, adapter_name=ADAPTER_NAME)
    student_model.set_template(
        'Template', model_id=STUDENT_MODEL_ID, adapter_name=ADAPTER_NAME,
        enable_thinking=False,
    )

    # Log projection matrix statistics (per teacher)
    for i, tid in enumerate(TEACHER_MODEL_IDS):
        stats = loss_fn.get_mapping_statistics(i)
        logger.info(
            f'CrossToken Projection Matrix Statistics (teacher_{i}={tid}): '
            f'{stats}'
        )
        coverage_ratio = stats['exact_matched'] / stats['total_student_tokens']
        logger.info(
            f'Vocabulary coverage ratio (teacher_{i}): {coverage_ratio:.2%}'
        )
        if coverage_ratio < 0.3:
            logger.warning(
                f"Low vocabulary coverage ({coverage_ratio:.2%}) vs "
                f"teacher_{i} ({tid}), consider using models with "
                "similar tokenizers"
            )
        if LOSS_TYPE == 'hkl':
            logger.info(
                f'H-KL mode: gamma_kl={GAMMA_KL}, gamma_uld={GAMMA_ULD} '
                f'(teacher_{i})'
            )
            logger.info(
                f'  Unmatched tokens (ULD) vs teacher_{i}: '
                f'{stats["unmatched"]}/{stats["total_student_tokens"]}'
            )

    # ── Student vLLM sampler (for on-policy generation) ───────────────────────
    student_sampler = vLLMSampler(
        model_id=STUDENT_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.75,
            'max_model_len': 4096,
            'enable_lora': True,
            'max_lora_rank': 8,
        },
        device_mesh=sampler_mesh,
        remote_group='student_sampler',
    )
    student_sampler.set_template(
        'Template', model_id=STUDENT_MODEL_ID, enable_thinking=False,
    )

    # ── Teacher TransformersModels (for full logits, one per teacher) ─────────
    teacher_models = [
        TransformersModel(
            model_id=tid,
            device_mesh=mesh,
            remote_group=f'teacher_{i}',
        )
        for i, (tid, mesh) in enumerate(zip(TEACHER_MODEL_IDS, teacher_meshes))
    ]
    for teacher_model, tid in zip(teacher_models, TEACHER_MODEL_IDS):
        teacher_model.set_template(
            'Template', model_id=tid, enable_thinking=False,
        )

    # ── Checkpoint manager for weight sync ────────────────────────────────────
    ckpt_manager = CheckpointEngineManager(
        model=student_model, sampler=student_sampler,
    )

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataloader = DataLoader(
        dataset=create_dataset(),
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='student_model',
    )

    logger.info(get_device_placement())
    logger.info(
        f'CrossToken On-Policy Training | '
        f'student={STUDENT_MODEL_ID}  teacher={TEACHER_MODEL_IDS}'
    )
    logger.info(
        f'  teacher_weights={loss_fn.teacher_weights}'
        f'  teacher_gpus={TEACHER_GPUS}'
    )
    logger.info(
        f'  loss_type={LOSS_TYPE}  T={TEMPERATURE}  vocab_topk={VOCAB_TOPK}'
    )
    logger.info(
        f'  beta={BETA}  gamma={GAMMA}  max_length={MAX_LENGTH}'
    )
    if LOSS_TYPE == 'hkl':
        logger.info(
            f'  gamma_kl={GAMMA_KL}  gamma_uld={GAMMA_ULD}  '
            f'uncommon_topk={UNCOMMON_TOPK}'
        )
    logger.info(
        f'  batch_size={BATCH_SIZE}  lr={LEARNING_RATE}  '
        f'max_steps={MAX_STEPS}  max_new_tokens={MAX_NEW_TOKENS}  '
        f'n_samples={N_SAMPLES}'
    )
    if DYNAMIC_LOSS_SCALING:
        logger.info('  dynamic_loss_scaling=enabled')

    # ── Training Loop (On-Policy) ─────────────────────────────────────────────
    optim_step = 1
    for batch in dataloader:
        if optim_step > MAX_STEPS:
            break
        if callable(batch):
            batch = batch()

        # ── Step 1: Sync student weights to sampler ───────────────────────────
        ckpt_manager.sync_weights(merge_and_sync=False)

        # ── Step 2: Student generates completions ────────────────────────────
        sample_response = student_sampler.sample(
            batch,
            SamplingParams(
                max_tokens=MAX_NEW_TOKENS,
                min_tokens=MIN_NEW_TOKENS,
                temperature=1.0,
                num_samples=N_SAMPLES,
            ),
        )

        # ── Step 2.5: 过滤短生成样本 ────────────────────────────────────
        # 生成 < MIN_GEN_TOKENS 的样本跳过(噪声步 + 阻止学生"提前停止"
        # 逃避教师评价——短输出不再获得训练信号)。
        pairs = [
            (resp, seq)
            for resp in sample_response
            for seq in resp.sequences
        ]
        valid_pairs = []
        for resp, seq in pairs:
            gen_len = (
                    len(seq.new_input_feature['input_ids'])
                    - len(resp.prompt_token_ids)
            )
            valid_pairs.append(gen_len >= MIN_GEN_TOKENS)
        if not any(valid_pairs):
            logger.warning(
                f'[Step {optim_step}] all samples filtered (short generation), '
                'skipping step'
            )
            continue

        # Extract generated sequences (prompt + student-generated response)
        input_data = [
            seq.new_input_feature
            for (resp, seq), v in zip(pairs, valid_pairs)
            if v
        ]

        # ── Step 3: Prepare teacher inputs (decode + re-encode) ──────────────
        # 每个教师各自重编码(不同 tokenizer/词表),与 valid_pairs 同序
        # (resp→seq 遍历),过滤短生成样本
        teacher_inputs_list = []
        for teacher_tokenizer in teacher_tokenizers:
            t_inputs = prepare_teacher_inputs_from_student_gen(
                sample_response, student_tokenizer, teacher_tokenizer,
            )
            teacher_inputs_list.append(
                [t for t, v in zip(t_inputs, valid_pairs) if v]
            )

        # ── Step 4: Teacher forward → full logits (per teacher) ──────────────
        teacher_logits_list = []
        teacher_input_ids_list = []
        for i, teacher_model in enumerate(teacher_models):
            t_outputs = teacher_model.forward_only(
                inputs=teacher_inputs_list[i],
                return_logits=True,
                temperature=1.0,
                disable_lora=True,
                adapter_name='',
            )
            t_outputs = t_outputs()  # remote_function → actual result
            teacher_logits_list.append(t_outputs['logits'])

            t_prompt_ids = [
                torch.tensor(feat['input_ids'], dtype=torch.long)
                for feat in teacher_inputs_list[i]
            ]
            teacher_input_ids_list.append(
                rnn_utils.pad_sequence(
                    t_prompt_ids, batch_first=True, padding_value=0,
                )
            )

        # ── Step 5: Package teacher output ───────────────────────────────────
        # Use student-generated input_ids as labels, but only for the response
        # part: prompt positions are masked to -100 so both CE and the KD
        # chunk mask (already gated on labels) distill only the response.
        # Matches the framework convention (vllm_sampler sets prompt labels
        # to -100) and keeps template-structure positions (\n, <|im_start|>,
        # user/assistant) from dominating the loss.
        student_labels_list = []
        student_ids_list = []
        for (resp, seq), v in zip(pairs, valid_pairs):
            if not v:
                continue  # 短生成样本已过滤
            full_ids = seq.new_input_feature['input_ids']
            labels = list(full_ids)
            for p in range(len(resp.prompt_token_ids)):
                labels[p] = -100
            student_labels_list.append(torch.tensor(labels, dtype=torch.long))
            student_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        student_labels = rnn_utils.pad_sequence(
            student_labels_list, batch_first=True, padding_value=-100,
        )
        student_ids = rnn_utils.pad_sequence(
            student_ids_list, batch_first=True, padding_value=0,
        )

        teacher_output = {
            'teacher_logits_group': teacher_logits_list,
            'teacher_input_ids_group': teacher_input_ids_list,
            'teacher_labels': [student_labels] * NUM_TEACHERS,
        }

        # ── Step 6: Student forward + CrossToken backward ────────────────────
        student_outputs = student_model.forward_backward(
            inputs=input_data,
            adapter_name=ADAPTER_NAME,
            return_logits=True,
            **teacher_output,
        )

        student_model.clip_grad_and_step(adapter_name=ADAPTER_NAME)

        # ── Logging ───────────────────────────────────────────────────────────
        if optim_step > 0 and optim_step % 2 == 0:
            metric = student_model.calculate_metric(
                is_training=True, adapter_name=ADAPTER_NAME,
            )
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metric}')

        # ── Checkpoint ────────────────────────────────────────────────────────
        if optim_step > 0 and optim_step % 100 == 0:
            student_model.save(
                f'cross-token-on-policy-ckpt-{optim_step}',
                adapter_name=ADAPTER_NAME,
            )

        optim_step += 1

    # ── Save final checkpoint ─────────────────────────────────────────────────
    student_model.save(
        'cross-token-on-policy-final', adapter_name=ADAPTER_NAME,
    )

    logger.info(
        f'CrossToken on-policy training completed after {optim_step} steps.'
    )


if __name__ == '__main__':
    train()
