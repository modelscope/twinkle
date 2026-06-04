"""LoRA embedding training: query ↔ CM-v2-compressed thinking_content (Transformers + Megatron).

Pipeline:
  - 4 GPUs (``sampler`` group) load ``ms://twinkle-kit/Qwen3.5-4B-CM-v2`` via
    :class:`vLLMSampler` and run as a frozen online compressor.
  - 4 GPUs (``model`` group) load the same checkpoint with a LoRA adapter and
    train an embedding head against InfoNCE.
  - Each row from :func:`dataset_think.get_dataset` provides ``(query, cot)``;
    every step compresses ``cot`` through CM-v2 (with the production
    Condenser system+user prompt) and treats ``(query, compressed_cot)`` as
    the anchor/positive pair. In-batch + cross-DP samples become negatives.

Switch ``BACKEND`` between ``'transformers'`` and ``'megatron'``; the rest of
the script is backend-agnostic.

Launch:
    python cookbook/exp/train_embedding_lora_ddp.py
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import swanlab
import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import InputFeature, SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.loss import InfonceLoss
from twinkle.metric import EmbeddingMetric
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle.utils.parallel import PosixFileLock

# allow importing the sibling dataset_think module without packaging
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_think import get_dataset  # noqa: E402

logger = get_logger()

# -- Backend selection --------------------------------------------------------
BACKEND: Literal['transformers', 'megatron'] = 'transformers'

MODEL_ID = os.environ.get('MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
TEMPLATE_NAME = 'Qwen3_5Template'

# -- GPU placement ------------------------------------------------------------
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

# -- Embedding training hyper-params ------------------------------------------
EMB_MAX_LENGTH = 4096
HARD_NEGATIVES = None  # rely on in-batch negatives only
TEMPERATURE = 0.05
LORA_RANK = 16
ADAPTER_NAME = 'default'

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 1
LOG_INTERVAL = 2
SAVE_INTERVAL = 4000
NUM_EPOCHS = 1

# None → use full _BASE_SIZES from dataset_think; int to subsample.
TOTAL_SAMPLES: Optional[int] = None

# -- Online-compression knobs (CM-v2 inference) -------------------------------
MIN_COT_CHARS = 256                           # skip too-short cot rows entirely
COMPRESS_RATIO = 2.0                          # used to derive the prompt char budget
COMPRESS_MAX_TOKENS = 2048
COMPRESS_TEMPERATURE = 0.2
COMPRESS_TOP_P = 0.5
COMPRESS_MAX_MODEL_LEN = 32768

OUTPUT_DIR = f'./output/embedding_lora_{BACKEND}'
RESPONSE_LOG = os.environ.get('RESPONSE_LOG', f'./output/embedding_lora_{BACKEND}/responses.jsonl')

_response_lock: Optional[PosixFileLock] = None


def _log_responses(query_resp, cot_resp, idx: int, query_raw: str = '', cot_raw: str = ''):
    """Append a (query_compressed, cot_compressed) pair to the JSONL log file.

    Uses PosixFileLock for multi-process safety.
    """
    global _response_lock
    if _response_lock is None:
        os.makedirs(os.path.dirname(RESPONSE_LOG) or '.', exist_ok=True)
        _response_lock = PosixFileLock(RESPONSE_LOG + '.lock')

    q_seq = query_resp.sequences[0] if query_resp.sequences else None
    c_seq = cot_resp.sequences[0] if cot_resp.sequences else None
    record = {
        'idx': idx,
        'query_raw': query_raw,
        'cot_raw': cot_raw,
        'query_compressed': q_seq.decoded if q_seq else None,
        'cot_compressed': c_seq.decoded if c_seq else None,
    }
    line = json.dumps(record, ensure_ascii=False, default=str) + '\n'
    with _response_lock:
        with open(RESPONSE_LOG, 'a', encoding='utf-8') as f:
            f.write(line)

# Production CM-v2 prompt (kept verbatim — same as cookbook/sample/sample.py).
CONDENSER_SYSTEM = """You are a text compression assistant. A downstream model will read your compressed output to decide whether the detail it needs is inside this block; if yes, it will fetch and read the original passage.

Downstream model workflow:
Read your compressed output -> Decide whether needed info is in this block -> If yes -> Fetch original.

Therefore your compression MUST NOT lose major information from the source.

Output format:

```text
## Summary
Overview plus facts STRONGLY RELATED to the Query, stated explicitly.

## More
A collapsed index; expansion required to see specific information.
```

Rules:
1. Telegraphic style — drop function words ("the", "a", "is", "are", "of", ...); colons and commas mean "is" / "has".
2. Summary MUST contain the passage's primary topic + 2–4 concrete core facts drawn from the source (entities, numbers, dates, relations). If a Query is given, order Query-relevant facts first, but STILL include other core facts within the budget. A Query is an ORDERING HINT, NOT a filter.
3. Summary MUST NOT be meta-commentary about the Query. Forbidden patterns: "no X mention", "Query info: absent", "passage covers Y only", "does not contain ...", "no relevant info", or summaries that are only abstract category words like "structure/order/usage" with no facts. If the passage is unrelated to the Query, you still summarize the passage normally.
4. More is an INDEX of category keywords, NOT inline data. Enumerate what CAN be recovered from the source (e.g. "birthplace, death place, age"); do NOT paste dates/numbers/names inline. Make sure all category of useful facts are introduced here.
5. Output language MUST match the source language.
6. Do NOT fabricate. Do NOT omit major information. Any fact not in the source MUST NOT appear in your output.

Now begin.
"""

CONDENSER_USER = (
    'Downstream model will read your compressed block to decide whether to '
    'expand it. Compress faithfully: preserve the passage topic + core facts. '
    'Do NOT invent facts. Do NOT drop major facts. Do NOT write meta-commentary '
    'about the Query (never write "Query info: absent", "no X mention", etc.); '
    'if the passage does not address the Query, still summarize the passage.\n\n'
    '## Query (ordering hint only — still summarize the whole passage)\n{query}\n\n'
    '## Target length\n'
    'Compress AS MUCH AS faithfully possible. HARD CEILING: {budget} chars '
    '(~50% of the source). If core facts fit in far fewer chars, output fewer. '
    'Never exceed the ceiling.\n\n'
    '## Passage\n{text}')


def build_model(device_mesh: DeviceMesh):
    if BACKEND == 'transformers':
        from twinkle.model import TransformersModel
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
            remote_group='model',
            ddp_config={'find_unused_parameters': True},
        )
        from twinkle.patch.no_split_modules import NoSplitModulesPatch
        model.apply_patch(NoSplitModulesPatch({'Qwen3_5DecoderLayer'}))
        return model
    if BACKEND == 'megatron':
        from twinkle.model import MegatronModel
        return MegatronModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
            remote_group='model',
            mixed_precision='bf16',
            variable_seq_lengths=True,
        )
    raise ValueError(f'Unknown BACKEND={BACKEND!r}')


def setup_optimizer(model, total_steps: int):
    if BACKEND == 'transformers':
        model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler(
            scheduler_cls='CosineWarmupScheduler',
            num_warmup_steps=50,
            num_training_steps=total_steps,
        )
        return
    if BACKEND == 'megatron':
        model.set_optimizer(optimizer_cls='default', lr=LEARNING_RATE)
        model.set_lr_scheduler(
            scheduler_cls='default',
            lr_warmup_steps=50,
            lr_decay_steps=total_steps,
        )
        return
    raise ValueError(f'Unknown BACKEND={BACKEND!r}')


def save_checkpoint(model, name: str):
    model.save(name, output_dir=OUTPUT_DIR, adapter_name=ADAPTER_NAME)


# --------------------------------------------------------------------- Loop
EMBED_QUERY_Q = (
    'What problem does this passage need to solve, and what kind of skill or '
    'method is required? Compress into a retrieval-friendly need description.')
EMBED_QUERY_COT = (
    'Extract the reusable skill: trigger conditions, key steps, and expected '
    'output. Compress into a standardized procedure for retrieval.')


def _extract_query_cot(row: Dict[str, Any]):
    """Extract (user_content, reasoning_content) from a messages-format row."""
    messages = row.get('messages') or []
    query, cot = '', ''
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role') or ''
        if role == 'user' and not query:
            query = (m.get('content') or '').strip()
        elif role == 'assistant':
            cot = (m.get('reasoning_content') or '').strip()
            break
    return query, cot


def _build_compress_prompts(rows: List[Dict[str, Any]]) -> tuple:
    """Build prompts for compressing both query and cot per row.

    Returns (prompts, valid_indices, raw_pairs) where prompts is flat-interleaved
    [query_0, cot_0, query_1, cot_1, ...], valid_indices tracks which
    rows passed the min-length filter, and raw_pairs is [(query, cot), ...].
    """
    prompts: List[Dict[str, Any]] = []
    valid_indices: List[int] = []
    raw_pairs: List[tuple] = []
    for i, row in enumerate(rows):
        query, cot = _extract_query_cot(row)
        if not query or len(cot) < MIN_COT_CHARS:
            continue
        valid_indices.append(i)
        raw_pairs.append((query, cot))
        for text, qtpl in ((query, EMBED_QUERY_Q), (cot, EMBED_QUERY_COT)):
            budget = max(1, int(len(text) / COMPRESS_RATIO))
            user = CONDENSER_USER.format(query=qtpl, budget=budget, text=text)
            prompts.append({'messages': [
                {'role': 'system', 'content': CONDENSER_SYSTEM},
                {'role': 'user', 'content': user},
            ]})
    return prompts, valid_indices, raw_pairs


def _get_first_feature(response, template: Template, role: str) -> Optional[Dict[str, Any]]:
    """Encode decoded text from first sampled sequence via template."""
    seqs = getattr(response, 'sequences', None) or []
    if not seqs:
        return None
    text = getattr(seqs[0], 'decoded', None)
    if not text:
        return None
    if role == 'anchor':
        feat = template.encode({'messages': [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': 'Match the correct response here.'}]})
        feat['labels'] = [1]
    else:
        feat = template.encode({'messages': [{'role': 'user', 'content': 'Match the correct query here.'}, {'role': 'assistant', 'content': text}]})
        feat['labels'] = [0]
    return feat


def train():
    # -------- Ray + device groups --------------------------------------------
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler',
                    ranks=list(range(MODEL_GPUS, NUM_GPUS)),
                    device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)

    # -------- Data -----------------------------------------------------------
    dataset = get_dataset(total=TOTAL_SAMPLES, load_from_cache_file=True)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    total_steps = len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS

    # -------- Trainable embedding model + LoRA -------------------------------
    model = build_model(model_mesh)
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2, lora_dropout=0.05,
        target_modules='all-linear')
    model.add_adapter_to_model(
        ADAPTER_NAME, lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

    model.set_processor(InputProcessor)
    model.set_loss(
        InfonceLoss,
        temperature=TEMPERATURE,
        use_batch=True,
        hard_negatives=HARD_NEGATIVES,
    )
    setup_optimizer(model, total_steps)
    model.add_metric(EmbeddingMetric, is_training=True)

    # -------- Frozen CM-v2 sampler (online compressor) -----------------------
    emb_template = Template(model_id=MODEL_ID, max_length=EMB_MAX_LENGTH, enable_thinking=False)
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': COMPRESS_MAX_MODEL_LEN,
            'enable_lora': False,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE_NAME, model_id=MODEL_ID, enable_thinking=False, truncation_strategy='delete', max_length=COMPRESS_MAX_TOKENS)
    compress_params = SamplingParams(
        max_tokens=COMPRESS_MAX_TOKENS,
        temperature=COMPRESS_TEMPERATURE,
        top_p=COMPRESS_TOP_P,
        num_samples=1,
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {total_steps}')

    swanlab.init(project='twinkle', config={
        'backend': BACKEND,
        'model_id': MODEL_ID,
        'batch_size': BATCH_SIZE,
        'lr': LEARNING_RATE,
        'lora_rank': LORA_RANK,
        'temperature': TEMPERATURE,
        'emb_max_length': EMB_MAX_LENGTH,
        'compress_ratio': COMPRESS_RATIO,
        'compress_max_tokens': COMPRESS_MAX_TOKENS,
    })

    # -------- Train loop -----------------------------------------------------
    def _sample_batch(raw_batch):
        """Sample compress prompts and build embedding features. Runs in prefetch thread."""
        compress_prompts, valid_indices, raw_pairs = _build_compress_prompts(raw_batch)
        if not compress_prompts:
            return None
        responses = sampler.sample(compress_prompts, compress_params)

        # Retry truncated responses up to 3 times
        retry_indices = []
        for ri, resp in enumerate(responses):
            seq = resp.sequences[0] if resp.sequences else None
            if seq and seq.stop_reason == 'length':
                retry_indices.append(ri)

        for attempt in range(3):
            if not retry_indices:
                break
            print(f'retry: {attempt}')
            retry_prompts = [compress_prompts[ri] for ri in retry_indices]
            pad_count = (SAMPLER_GPUS - len(retry_prompts) % SAMPLER_GPUS) % SAMPLER_GPUS
            padded_prompts = retry_prompts + [retry_prompts[i % len(retry_prompts)] for i in range(pad_count)] if pad_count else retry_prompts
            retry_responses = sampler.sample(padded_prompts, compress_params)
            still_truncated = []
            for j, ri in enumerate(retry_indices):
                new_resp = retry_responses[j]
                new_seq = new_resp.sequences[0] if new_resp.sequences else None
                if new_seq and new_seq.stop_reason != 'length':
                    responses[ri] = new_resp
                else:
                    still_truncated.append(ri)
            retry_indices = still_truncated

        if retry_indices:
            for ri in retry_indices:
                side = 'query' if ri % 2 == 0 else 'cot'
                idx = valid_indices[ri // 2]
                seq = responses[ri].sequences[0] if responses[ri].sequences else None
                print(f'[max_length hit after 3 retries] side={side}, batch_idx={idx}, '
                      f'decoded_len={len(seq.decoded) if seq and seq.decoded else 0}')
                raise

        emb_features: List[Dict[str, Any]] = []
        for i in range(0, len(responses), 2):
            q_raw, c_raw = raw_pairs[i // 2]
            _log_responses(responses[i], responses[i + 1], valid_indices[i // 2],
                           query_raw=q_raw, cot_raw=c_raw)
            feat_q = _get_first_feature(responses[i], emb_template, role='anchor')
            feat_c = _get_first_feature(responses[i + 1], emb_template, role='positive')
            emb_features.append(feat_q)
            emb_features.append(feat_c)

        if len(emb_features) < 4:
            raise ValueError(f'Not enough valid pairs in batch: {len(emb_features) // 2} < 2')
        return emb_features

    cur_step = 0
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    for epoch in range(NUM_EPOCHS):
        batch_iter = iter(dataloader)
        # Prefetch first batch
        prefetch_future = None
        first_batch = next(batch_iter, None)
        if first_batch is not None:
            prefetch_future = prefetch_executor.submit(_sample_batch, first_batch)

        for raw_batch in batch_iter:
            # Get current features from prefetch
            emb_features = prefetch_future.result() if prefetch_future else None
            # Submit next batch to sampler (overlaps with model training below)
            prefetch_future = prefetch_executor.submit(_sample_batch, raw_batch)

            if emb_features is None:
                continue

            model.forward_backward(inputs=emb_features, task='embedding')
            model.clip_grad_and_step()
            cur_step += 1

            if cur_step % LOG_INTERVAL == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(
                    f'Epoch {epoch} Step {cur_step}/{total_steps}, metric: {metric}')
                log_dict = {k: float(v) for k, v in metric.items() if v}
                log_dict['epoch'] = epoch
                swanlab.log(log_dict, step=cur_step)
            if cur_step % SAVE_INTERVAL == 0:
                save_checkpoint(model, f'step_{cur_step}')

        # Drain the last prefetched batch
        if prefetch_future is not None:
            emb_features = prefetch_future.result()
            if emb_features is not None:
                model.forward_backward(inputs=emb_features, task='embedding')
                model.clip_grad_and_step()
                cur_step += 1

        save_checkpoint(model, f'epoch-{epoch}')
    prefetch_executor.shutdown(wait=False)
    save_checkpoint(model, 'last-checkpoint')


if __name__ == '__main__':
    train()
