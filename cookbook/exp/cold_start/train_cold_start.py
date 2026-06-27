import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, PackingDataset
from twinkle.dataset.base import DatasetMeta
from twinkle.model import MegatronModel
from twinkle_agentic.preprocessor import (
    QualityPreprocessor, SamplerBackend,
    IntentClassifier, HardFilter, RefuseFilter, DeadLoopFilter, TokenSoupFilter, MessageSanityFilter,
    SpecialCharsFilter, ModelFilter, DedupFilter,
    MessageNormalizer,
)

logger = get_logger()

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID = 'ms://Qwen/Qwen3-4B'
TEMPLATE_NAME = 'Template'
MAX_LENGTH = 80000

# ── GPU allocation ───────────────────────────────────────────────────────────
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 8))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 0))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 1))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRAD_ACCUM', 4))
LOG_INTERVAL = 1
SAVE_INTERVAL = 500
NUM_STEPS = int(os.environ.get('NUM_STEPS', 5000))

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = './output/streaming_sft'
TRAINED_DATA_PATH = os.path.join(OUTPUT_DIR, 'trained_data.jsonl')
DROPPED_DATA_PATH = os.path.join(OUTPUT_DIR, 'dropped_data.jsonl')
ADAPTER_NAME = 'default'

# ── Data source ──────────────────────────────────────────────────────────────
CSV_PATH = os.environ.get('CSV_PATH')
DATASET_TOTAL = int(os.environ.get('DATASET_TOTAL', 10000))  # 0 = full materialized dataset
# Worker count for HF Dataset.map(num_proc=N); spawn start method is forced in twinkle.dataset.base.
MAP_NUM_PROC = int(os.environ.get('MAP_NUM_PROC', 16))


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
                'user_data': [],
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
    dataset = PackingDataset(meta)

    qp = QualityPreprocessor(
        pipeline=[
            ModelFilter(),
            MessageNormalizer(),
            HardFilter(
                min_user_chars_cjk=14, min_user_chars=24,
                system_deny_keywords=[
                    '角色扮演', '扮演', '人设', 'roleplay', 'role play', 'cosplay',
                    '群聊模拟', '虚拟角色', '二次元', 'OC设定',
                ],
                max_rounds=30,
            ),
            RefuseFilter(),
            DeadLoopFilter(),
            MessageSanityFilter(sensitive_words_file='.temp/sensitive_words.txt'),
            SpecialCharsFilter(max_ratio=0.6),
            TokenSoupFilter(max_chars=8000),
            IntentClassifier(),
            # ScoreFilter(
            #     template=template,
            #     backend=backend,
            #     scorers=[
            #         ChrMinScorer(),
            #     ],
            # ),
            # PIIPresidioFilter(languages=('en', 'zh')),
        ],
        dropped_log_path=DROPPED_DATA_PATH,
    )
    dataset.map(qp, num_proc=8, load_from_cache_file=True)
    dataset.map(
        QualityPreprocessor(pipeline=[DedupFilter()]),
        num_proc=1,
        batch_size=len(dataset.dataset),
        load_from_cache_file=True,
    )

    print(len(dataset.dataset))
    dataset.set_template(
        TEMPLATE_NAME,
        model_id=MODEL_ID,
        max_length=MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    dataset.encode(num_proc=16, load_from_cache_file=True)
    dataset.pack_dataset()
    return dataset


def save_checkpoint(model: MegatronModel, checkpoint_name: str, dataloader: DataLoader):
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
        # DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU', gpus_per_worker=2),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=1, cp_size=8)
    # sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS // 2, tp_size=2)
    twinkle.initialize(mode='local', nproc_per_node=NUM_GPUS, groups=device_groups,
                       global_device_mesh=model_mesh, lazy_collect=False)

    # ── vLLMSampler on GPUs 4-7 (Ray actor, no HTTP overhead) ────────────────
    # sampler = vLLMSampler(
    #     model_id=MODEL_ID,
    #     engine_args={
    #         'gpu_memory_utilization': 0.6,
    #         'max_model_len': MAX_LENGTH,
    #     },
    #     device_mesh=sampler_mesh,
    #     remote_group='sampler',
    # )
    # sampler.set_template(TEMPLATE_NAME, model_id=MODEL_ID)
    # backend = SamplerBackend(sampler)
    # logger.info(f'vLLMSampler ready on GPUs {MODEL_GPUS}-{NUM_GPUS - 1}')

    # ── Dataset with full QualityPreprocessor (uses SamplerBackend) ───────────
    dataset = build_dataset(None)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
    )

    # ── Model (LoRA on 4 GPUs) ────────────────────────────────────────────────
    model = MegatronModel(
        model_id=MODEL_ID,
        device_mesh=model_mesh,
        # remote_group='model',
        # attn_implementation='flash_attention_2',
    )

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model(
        ADAPTER_NAME, lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer(optimizer_cls='default', lr=LEARNING_RATE)
    model.set_lr_scheduler(
        scheduler_cls='default',
        lr_warmup_steps=2,
        lr_decay_steps=len(dataloader))

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
