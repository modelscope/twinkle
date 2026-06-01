"""Streaming SFT with QualityPreprocessor + OdpsIterableDataset (Ray mode).

Architecture (8 GPUs single-node):
    GPU 0-3: LoRA SFT training (4x DP)
    GPU 4-7: vLLMSampler Ray actor (same model, for QualityPreprocessor)

QualityPreprocessor phases (intent, IFD, refine) use SamplerBackend
which calls vLLMSampler directly via Ray (no HTTP overhead).

Two output files are produced:
  - trained_data.jsonl: rows that pass QualityPreprocessor and are consumed by training
  - dropped_data.jsonl: rows dropped by QualityPreprocessor (with step annotation)

Launch:
    python cookbook/exp/train_streaming_sft.py
"""
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import Features, Value
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.preprocessor import QualityPreprocessor, SamplerBackend

logger = get_logger()

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
MODEL_LOCAL_PATH = os.environ.get('MODEL_LOCAL_PATH', 'Qwen/Qwen3.5-4B')
TEMPLATE_NAME = 'Qwen3_5Template'
MAX_LENGTH = 40000

# ── GPU allocation ───────────────────────────────────────────────────────────
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 16))
LEARNING_RATE = float(os.environ.get('LR', 1e-4))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRAD_ACCUM', 2))
LOG_INTERVAL = 20
SAVE_INTERVAL = 500
NUM_STEPS = int(os.environ.get('NUM_STEPS', 5000))

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = './output/streaming_sft'
TRAINED_DATA_PATH = os.path.join(OUTPUT_DIR, 'trained_data.jsonl')
DROPPED_DATA_PATH = os.path.join(OUTPUT_DIR, 'dropped_data.jsonl')
ADAPTER_NAME = 'default'

# ── Data source (test mode: Chinese-DeepSeek-R1-Distill-data-110k) ───────────
CN_R1_DISTILL_REPO = 'ms://AI-ModelScope/Chinese-DeepSeek-R1-Distill-data-110k'
DATASET_LIMIT = int(os.environ.get('DATASET_LIMIT', 1000))
DATASET_USE_CACHE = os.environ.get('DATASET_USE_CACHE', '1') == '1'

_TARGET_FEATURES = Features({
    'id': Value('string'),
    'source': Value('string'),
    'messages': [{'role': Value('string'), 'content': Value('string')}],
})
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


class CNR1DistillSFTProcessor(Preprocessor):
    """CN-R1-Distill raw row → full SFT messages: ``[user: input, assistant: <think>cot</think>response]``."""

    _SOURCE = 'Chinese-DeepSeek-R1-Distill-data-110k'

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows_list = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows_list:
            query = (row.get('input') or '').strip()
            cot = (row.get('reasoning_content') or '').strip()
            response = (row.get('content') or '').strip()
            if not query or not response:
                continue
            response = _THINK_RE.sub('', response).strip() if cot else response
            assistant = f'<think>{cot}</think>{response}' if cot else response
            row_id = hashlib.md5((query + assistant).encode('utf-8')).hexdigest()[:16]
            out.append({
                'id': f'{self._SOURCE}__{row_id}',
                'source': self._SOURCE,
                'messages': [
                    {'role': 'user', 'content': query},
                    {'role': 'assistant', 'content': assistant},
                ],
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages'])

# ── QualityPreprocessor config ───────────────────────────────────────────────
SENSITIVE_WORDS_FILE = str(
    Path(__file__).resolve().parent.parent.parent / 'sensitive_words.txt')
IFD_THRESHOLD = float(os.environ.get('IFD_THRESHOLD', 0.8))
REFINE_TEMPERATURE = float(os.environ.get('REFINE_TEMPERATURE', 0.6))
REFINE_MAX_TOKENS = int(os.environ.get('REFINE_MAX_TOKENS', 4096))


def build_dataset(backend: SamplerBackend) -> Dataset:
    """Build dataset from CN_R1_DISTILL_REPO with full QualityPreprocessor pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = Dataset()
    meta = DatasetMeta(
        dataset_id=CN_R1_DISTILL_REPO, split='train',
        data_slice=range(DATASET_LIMIT),
    )
    dataset.add_dataset(meta)
    cols = list(dataset.datasets[meta.get_id()].column_names)
    dataset.map(
        CNR1DistillSFTProcessor,
        dataset_meta=meta,
        remove_columns=cols,
        load_from_cache_file=DATASET_USE_CACHE,
        features=_TARGET_FEATURES,
    )
    template = Qwen3_5Template(model_id=MODEL_ID, max_length=MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False)

    qp = QualityPreprocessor(
        # Shared LLM backend (vLLMSampler via Ray, no HTTP)
        backend=backend,
        # Phase 1.5: message sanity
        message_sanity_filter=True,
        sensitive_words_file=SENSITIVE_WORDS_FILE,
        # Phase 2: structural
        hard_filter=True,
        refuse_filter=True,
        dead_loop_filter=True,
        # Phase 3: character quality
        token_soup_filter=True,
        special_chars_max_ratio=0.5,
        minhash_dedup=False,
        # Phase 12: IFD hard-example filter
        ifd_template=template,
        ifd_threshold=IFD_THRESHOLD,
        # Phase 13: response refinement
        refine_temperature=REFINE_TEMPERATURE,
        refine_max_tokens=REFINE_MAX_TOKENS,
        refine_max_workers=8,
        # Diagnostics
        dropped_log_path=DROPPED_DATA_PATH,
    )
    dataset.map(qp, load_from_cache_file=False)

    dataset.set_template(
        TEMPLATE_NAME,
        model_id=MODEL_ID,
        max_length=MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    dataset.encode()
    dataset.save_as(TRAINED_DATA_PATH, format='jsonl', mode='training')

    return dataset


def save_checkpoint(model: TransformersModel, checkpoint_name: str, dataloader: DataLoader):
    model.save(
        checkpoint_name,
        output_dir=OUTPUT_DIR,
        adapter_name=ADAPTER_NAME,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def train():
    # ── Ray mode: GPUs 0-3 for training, GPUs 4-7 for vLLMSampler ────────────
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU', gpus_per_worker=2),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS // 2, tp_size=2)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # ── vLLMSampler on GPUs 4-7 (Ray actor, no HTTP overhead) ────────────────
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.6,
            'max_model_len': MAX_LENGTH,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE_NAME, model_id=MODEL_ID)
    backend = SamplerBackend(sampler)
    logger.info(f'vLLMSampler ready on GPUs {MODEL_GPUS}-{NUM_GPUS - 1}')

    # ── Dataset with full QualityPreprocessor (uses SamplerBackend) ───────────
    dataset = build_dataset(backend)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    # ── Model (LoRA on 4 GPUs) ────────────────────────────────────────────────
    model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
    )

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model(
        ADAPTER_NAME, lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=50,
        num_training_steps=NUM_STEPS)

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {NUM_STEPS}, model GPUs: {MODEL_GPUS}, sampler GPUs: {SAMPLER_GPUS}')

    optimizer_group = model.optimizer_group[ADAPTER_NAME]

    for batch in dataloader:
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        cur_step = optimizer_group.cur_step

        if cur_step % LOG_INTERVAL == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Step {cur_step}/{NUM_STEPS}, metric: {metric}')

        if cur_step % SAVE_INTERVAL == 0:
            save_checkpoint(model, f'step-{cur_step}', dataloader)

        if cur_step >= NUM_STEPS:
            break

    save_checkpoint(model, 'last-checkpoint', dataloader)
    dataset.flush_save()
    logger.info(f'Training complete. Trained data saved to: {TRAINED_DATA_PATH}')
    logger.info(f'Dropped data saved to: {DROPPED_DATA_PATH}')


if __name__ == '__main__':
    train()
