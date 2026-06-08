"""Streaming SFT with QualityPreprocessor on a streaming IterableDataset (Ray mode).

Architecture (8 GPUs single-node):
    GPU 0-3: LoRA SFT training (4x DP)
    GPU 4-7: vLLMSampler Ray actor (same model, for QualityPreprocessor)

QualityPreprocessor phases (intent, IFD, refine) use SamplerBackend
which calls vLLMSampler directly via Ray (no HTTP overhead).

Two output files are produced:
  - trained_data.jsonl: write-through of rows actually consumed by training
  - dropped_data.jsonl: rows dropped by QualityPreprocessor (with step annotation)

Launch:
    python cookbook/exp/train_streaming_sft.py
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List
from functools import partial
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset
from twinkle.dataset.base import DatasetMeta
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.preprocessor import (
    QualityPreprocessor, SamplerBackend,
    IntentClassifier, ResponseRefiner, ScoreFilter,
    HardFilter, RefuseFilter, DeadLoopFilter, TokenSoupFilter, MessageSanityFilter,
    SpecialCharsFilter, PIIPresidioFilter, ModelFilter, DedupFilter,
    ToolCallNormalizer,
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
LOG_INTERVAL = 1
SAVE_INTERVAL = 500
NUM_STEPS = int(os.environ.get('NUM_STEPS', 5000))

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = './output/streaming_sft'
TRAINED_DATA_PATH = os.path.join(OUTPUT_DIR, 'trained_data.jsonl')
DROPPED_DATA_PATH = os.path.join(OUTPUT_DIR, 'dropped_data.jsonl')
ADAPTER_NAME = 'default'

# ── Data source ──────────────────────────────────────────────────────────────
CSV_PATH = os.environ.get(
    'CSV_PATH', '/mnt/workspace/yzhao/tastelikefeet/bc/ds_csv/data/20260531.csv')
DATASET_TOTAL = int(os.environ.get('DATASET_TOTAL', 1000))  # 0 = full materialized dataset
# Worker count for HF Dataset.map(num_proc=N); spawn start method is forced in twinkle.dataset.base.
MAP_NUM_PROC = int(os.environ.get('MAP_NUM_PROC', 1))


def _canonicalize_tool_call(tc: Any) -> Dict[str, Any]:
    """Coerce ``tool_calls[i]`` to a fixed-schema dict for stable Arrow inference.

    Keeps ``function.arguments`` as the OpenAI-native JSON string so every row
    sees a uniform ``string`` field; any string→dict decoding is the
    chat_template's concern (see ``Template._apply_chat_template``).

    The decoded form is enforced to be a JSON object so the chat_template's
    ``|items`` filter never receives list/scalar/null — those originate from
    dirty CSV rows and are coerced to ``{}`` here, the ingestion boundary.
    """
    tc = tc if isinstance(tc, dict) else {}
    fn = tc.get('function') if isinstance(tc.get('function'), dict) else {}
    args = fn.get('arguments')
    if isinstance(args, dict):
        args_str = json.dumps(args, ensure_ascii=False)
    elif isinstance(args, str) and args.strip():
        try:
            decoded = json.loads(args)
        except json.JSONDecodeError:
            decoded = {}
        if not isinstance(decoded, dict):
            decoded = {}
        args_str = json.dumps(decoded, ensure_ascii=False)
    else:
        args_str = '{}'
    return {
        'id': str(tc.get('id') or ''),
        'type': str(tc.get('type') or 'function'),
        'function': {
            'name': str(fn.get('name') or ''),
            'arguments': args_str,
        },
    }


def _stream_csv_rows(csv_path: str, max_rows: int = 0) -> Iterator[Dict[str, Any]]:
    """Stream the custom CSV: each line is `ts,model,req_id,messages_json` (no quoting).

    The first 3 fields are scalar; the remainder of the line is a JSON array of
    chat messages, possibly containing commas — so we split on the first 3 commas only.
    ``max_rows`` caps the yielded rows at ingestion time so Arrow never materializes
    the unused tail.
    """
    emitted = 0
    with open(csv_path, 'rb') as f:
        bad_bytes = 0
        for raw in f:
            try:
                line = raw.decode('utf-8').rstrip('\n').rstrip('\r')
            except UnicodeDecodeError:
                bad_bytes += 1
                continue
            if not line:
                continue
            parts = line.split(',', 3)
            if len(parts) < 4:
                continue
            ts, _model, req_id, msgs_raw = parts
            try:
                raw_msgs = json.loads(msgs_raw)
            except json.JSONDecodeError:
                continue
            messages: List[Dict[str, Any]] = []
            for m in raw_msgs:
                role = m.get('role', '')
                content = m.get('content')
                # User content arrives as [{'type':'text','text':...}, ...]; flatten to plain string.
                if isinstance(content, list):
                    content = ''.join(
                        p.get('text', '') for p in content
                        if isinstance(p, dict) and p.get('type') == 'text')
                if content is None:
                    content = ''
                if not isinstance(content, str):
                    continue
                raw_tcs = m.get('tool_calls') if role == 'assistant' else None
                tc_list = [_canonicalize_tool_call(tc) for tc in raw_tcs] if raw_tcs else []
                if role == 'assistant':
                    if not content and not tc_list:
                        continue
                    if m.get('reasoning_content'):
                        content = f"<think>{m['reasoning_content']}</think>{content}"
                elif role == 'tool':
                    pass
                elif not content:
                    continue
                # tool_calls stored as JSON string (empty -> ''): keeps Arrow schema as a
                # stable Value(string) regardless of empty-list / heterogeneous-struct shards.
                # Template._apply_chat_template decodes it back to list before jinja render.
                messages.append({
                    'role': role,
                    'content': content,
                    'tool_calls': json.dumps(tc_list, ensure_ascii=False) if tc_list else '',
                    'tool_call_id': str(m.get('tool_call_id') or '') if role == 'tool' else '',
                })
            if not messages:
                continue
            yield {
                'id': f'csv__{ts}__{req_id}',
                'source': Path(csv_path).stem,
                'model_id': _model,
                'messages': messages,
                'user_data': {},
            }
            emitted += 1
            if max_rows and emitted >= max_rows:
                break


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
    """Materialize the local CSV, convert to SFT messages format, run QualityPreprocessor.

    Switched from streaming IterableDataset to in-memory Dataset so HF
    `Dataset.map(num_proc=N)` can parallelize the QualityPreprocessor pipeline.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Custom CSV format (commas inside JSON) — feed framework via callable, not csv loader.
    meta = DatasetMeta(
        dataset_id=Path(CSV_PATH).stem,
        data=partial(_stream_csv_rows, csv_path=CSV_PATH, max_rows=DATASET_TOTAL),
    )
    dataset = Dataset(meta)
    # template kept for future re-enablement of ScoreFilter; unused in current pipeline.
    _ = Qwen3_5Template(model_id=MODEL_ID, max_length=MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False)

    qp = QualityPreprocessor(
        pipeline=[
            ModelFilter(),
            HardFilter(
                min_user_chars_cjk=14, min_user_chars=24,
                system_deny_keywords=[
                    '角色扮演', '扮演', '人设', 'roleplay', 'role play', 'cosplay',
                    '群聊模拟', '虚拟角色', '二次元', 'OC设定',
                ],
            ),
            ToolCallNormalizer(),
            RefuseFilter(),
            DeadLoopFilter(),
            MessageSanityFilter(sensitive_words_file='.temp/sensitive_words.txt'),
            SpecialCharsFilter(max_ratio=0.6),
            TokenSoupFilter(max_chars=8000),
            IntentClassifier(),
            # Phase 12: conversation-level dedup (max 3 per system+user signature)
            DedupFilter(max_per_sig=1),
            # ScoreFilter temporarily disabled — reuses Ray vLLMSampler backend
            # which is incompatible with HF Dataset.map(num_proc>1) workers.
            # ScoreFilter(
            #     template=template,
            #     backend=backend,
            #     scorers=[
            #         ChrMinScorer(),
            #         # PassNScorer(
            #         #     backend=backend,
            #         #     judge_model=JUDGE_MODEL or None,
            #         #     judge_base_url=JUDGE_BASE_URL,
            #         #     judge_api_key=JUDGE_API_KEY,
            #         #     n=4,
            #         #     min_pass=0,
            #         #     sample_temperature=0.7,
            #         #     sample_max_tokens=4096,
            #         #     judge_temperature=JUDGE_TEMPERATURE,
            #         #     judge_max_tokens=JUDGE_MAX_TOKENS,
            #         #     judge_max_workers=JUDGE_MAX_WORKERS,
            #         # ),
            #         # ParaphraseScorer(
            #         #     backend=backend,
            #         #     template=template,
            #         # ),
            #     ],
            #     # trace_dir=os.path.join(OUTPUT_DIR, 'score_traces'),
            # ),
            # PIIPresidioFilter(languages=('en', 'zh')),
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
    dataset.map(qp, num_proc=MAP_NUM_PROC, load_from_cache_file=False)
    dataset.save_as('output/streaming_sft/filtered.jsonl')

    dataset.set_template(
        TEMPLATE_NAME,
        model_id=MODEL_ID,
        max_length=MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    dataset.encode(num_proc=MAP_NUM_PROC, load_from_cache_file=False)

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
        
        print([len(m['input_ids']) for m in batch])
        if cur_step == 17:
            print()
        
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
