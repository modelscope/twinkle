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
from twinkle_agentic.preprocessor import (
    QualityPreprocessor, SamplerBackend,
    IntentClassifier, ResponseRefiner, ScoreFilter,
    HardFilter, RefuseFilter, DeadLoopFilter, TokenSoupFilter, MessageSanityFilter,
    FixUnicodeFilter, RemoveRepeatSentencesFilter,
    WordRepeatFilter, CharRepeatFilter, SpecialCharsFilter, AlphanumericFilter,
    FlaggedWordsFilter, MinHashDedupFilter,
)
from twinkle_agentic.preprocessor.score_filter import (
    ChrMinScorer, PassNScorer, ParaphraseScorer,
)

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
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
LEARNING_RATE = float(os.environ.get('LR', 1e-4))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRAD_ACCUM', 8))
LOG_INTERVAL = 20
SAVE_INTERVAL = 500
NUM_STEPS = int(os.environ.get('NUM_STEPS', 5000))

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = './output/streaming_sft'
TRAINED_DATA_PATH = os.path.join(OUTPUT_DIR, 'trained_data.jsonl')
DROPPED_DATA_PATH = os.path.join(OUTPUT_DIR, 'dropped_data.jsonl')
ADAPTER_NAME = 'default'

# ── Data source ──────────────────────────────────────────────────────────────
CN_R1_DISTILL_REPO = 'ms://AI-ModelScope/Chinese-DeepSeek-R1-Distill-data-110k'
DATASET_TOTAL = int(os.environ.get('DATASET_TOTAL', 1000))  # 0 = all
DATASET_USE_CACHE = os.environ.get('DATASET_USE_CACHE', '0') == '1'
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


class CNR1DistillSFTProcessor(Preprocessor):
    """Chinese-DeepSeek-R1-Distill-data-110k → SFT messages format."""

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows_list = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows_list:
            query = (row.get('input') or '').strip()
            cot = (row.get('reasoning_content') or '').strip()
            response = (row.get('content') or '').strip()
            if not query or not response:
                continue
            if cot:
                response = _THINK_RE.sub('', response).strip()
                assistant_content = f'<think>{cot}</think>{response}'
            else:
                assistant_content = response
            messages = [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': assistant_content},
            ]
            rid = hashlib.md5(query.encode()).hexdigest()[:16]
            out.append({
                'id': f'cnr1__{rid}',
                'source': 'Chinese-DeepSeek-R1-Distill-data-110k',
                'messages': messages,
                'user_data': {'key_rounds': [1]},
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'messages', 'user_data'])

# ── QualityPreprocessor config ───────────────────────────────────────────────
SENSITIVE_WORDS_FILE = str(
    Path(__file__).resolve().parent.parent.parent / 'sensitive_words.txt')
# chr_min cutoff: keep round if chr_min < threshold (low chr_min = hard).
CHR_MIN_THRESHOLD = float(os.environ.get('CHR_MIN_THRESHOLD', 0.5))
REFINE_TEMPERATURE = float(os.environ.get('REFINE_TEMPERATURE', 0.6))
REFINE_MAX_TOKENS = int(os.environ.get('REFINE_MAX_TOKENS', 4096))

# ── Pass@4 LLM-as-judge (grades each diagnostic rollout vs GT) ───────────────
# Set JUDGE_MODEL='' to disable; otherwise judge runs over every diagnostic round.
JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'qwen3.7-max')
JUDGE_BASE_URL = os.environ.get('JUDGE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
JUDGE_API_KEY = os.environ.get('JUDGE_API_KEY', 'EMPTY')
JUDGE_TEMPERATURE = float(os.environ.get('JUDGE_TEMPERATURE', 0.3))
JUDGE_MAX_TOKENS = int(os.environ.get('JUDGE_MAX_TOKENS', 32000))
JUDGE_MAX_WORKERS = int(os.environ.get('JUDGE_MAX_WORKERS', 16))


def build_dataset(backend: SamplerBackend) -> Dataset:
    """Load CN-R1-Distill from ModelScope, convert to SFT format, run QualityPreprocessor."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = Dataset()
    data_slice = range(DATASET_TOTAL) if DATASET_TOTAL > 0 else None
    meta = DatasetMeta(dataset_id=CN_R1_DISTILL_REPO, split='train',
                       data_slice=data_slice)
    dataset.add_dataset(meta)
    cols = list(dataset.datasets[meta.get_id()].column_names)
    dataset.map(
        CNR1DistillSFTProcessor,
        dataset_meta=meta,
        remove_columns=cols,
        load_from_cache_file=DATASET_USE_CACHE,
    )
    template = Qwen3_5Template(model_id=MODEL_ID, max_length=MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False)

    qp = QualityPreprocessor(
        pipeline=[
            # Phase 1-5: deterministic structural filters
            HardFilter(),
            RefuseFilter(),
            DeadLoopFilter(),
            TokenSoupFilter(),
            MessageSanityFilter(),
            # Phase 6-7: text normalization (mappers)
            FixUnicodeFilter(),
            RemoveRepeatSentencesFilter(),
            # Phase 8-10: repetition & character quality
            WordRepeatFilter(),
            CharRepeatFilter(),
            SpecialCharsFilter(max_ratio=0.6),
            AlphanumericFilter(),
            FlaggedWordsFilter(),
            # MinHashDedupFilter(),
            IntentClassifier(),
            ScoreFilter(
                template=template,
                backend=backend,
                scorers=[
                    ChrMinScorer(),
                    # PassNScorer(
                    #     backend=backend,
                    #     judge_model=JUDGE_MODEL or None,
                    #     judge_base_url=JUDGE_BASE_URL,
                    #     judge_api_key=JUDGE_API_KEY,
                    #     n=4,
                    #     min_pass=0,
                    #     sample_temperature=0.7,
                    #     sample_max_tokens=4096,
                    #     judge_temperature=JUDGE_TEMPERATURE,
                    #     judge_max_tokens=JUDGE_MAX_TOKENS,
                    #     judge_max_workers=JUDGE_MAX_WORKERS,
                    # ),
                    # ParaphraseScorer(
                    #     backend=backend,
                    #     template=template,
                    # ),
                ],
                # trace_dir=os.path.join(OUTPUT_DIR, 'score_traces'),
            ),
            # Phase 13: response refinement
            # ResponseRefiner(
            #     backend=backend,
            #     temperature=REFINE_TEMPERATURE,
            #     max_tokens=REFINE_MAX_TOKENS,
            #     max_workers=8,
            # ),
        ],
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
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS // 2, fsdp_size=2)
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

    for cur_step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()

        if cur_step % LOG_INTERVAL == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Step {cur_step}/{NUM_STEPS}, metric: {metric}')

        if cur_step % SAVE_INTERVAL == 0:
            save_checkpoint(model, f'step-{cur_step}', dataloader)

        if cur_step >= NUM_STEPS:
            break

    save_checkpoint(model, 'last-checkpoint', dataloader)
    logger.info(f'Training complete. Trained data saved to: {TRAINED_DATA_PATH}')
    logger.info(f'Dropped data saved to: {DROPPED_DATA_PATH}')


if __name__ == '__main__':
    train()
