"""Full-parameter embedding training on pre-compressed dataset.

Reads the pre-compressed HF Dataset produced by make_embedding_dataset.py,
encodes features, trains with InfoNCE loss.

Architecture (4 GPUs):
  - Ranks 0-3: Trainable embedding model, InfoNCE loss.

Launch:
    python cookbook/exp/embedding/train_embedding_full_ddp.py
"""
import os
import time
from typing import Any, Dict, List, Literal, Optional

import swanlab

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.loss import InfonceLoss
from twinkle.metric import EmbeddingMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.template import Qwen3_5Template, Template

logger = get_logger()

# -- Backend selection --------------------------------------------------------
BACKEND: Literal['transformers', 'megatron'] = 'transformers'

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')

# -- GPU placement ------------------------------------------------------------
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 8))

# -- Embedding training hyper-params ------------------------------------------
EMB_MAX_LENGTH = 8192
HARD_NEGATIVES = None
TEMPERATURE = 0.07

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 1
LOG_INTERVAL = 2
SAVE_INTERVAL = 2000
NUM_EPOCHS = 1

# -- Dataset path (output of make_embedding_dataset.py) -----------------------
DATASET_PATH = os.environ.get('EMB_DATASET_PATH', 'ms://twinkle-kit/qth-embedding')
MIX_SHUFFLE_SEED = 42

# -- Resume from checkpoint ---------------------------------------------------
RESUME_CHECKPOINT = os.environ.get('RESUME_CHECKPOINT', '')
RESUME_STEP = int(os.environ.get('RESUME_STEP', 0))

# -- Output -------------------------------------------------------------------
OUTPUT_DIR = f'./output/embedding_full_{BACKEND}'


# =============================================================================
# Model builders
# =============================================================================

def build_model(device_mesh: DeviceMesh):
    model_id = RESUME_CHECKPOINT if RESUME_CHECKPOINT else MODEL_ID
    if BACKEND == 'transformers':
        model = TransformersModel(
            model_id=model_id,
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
            num_warmup_steps=200,
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
    model.save(name, output_dir=OUTPUT_DIR)


# =============================================================================
# Feature encoding
# =============================================================================

def _get_first_feature(decoded_text: str, template: Template, role: str) -> Optional[Dict[str, Any]]:
    if not decoded_text:
        return None
    if role == 'anchor':
        feat = template.encode({'messages': [
            {'role': 'user', 'content': decoded_text},
            {'role': 'assistant', 'content': 'Match the correct response here.'},
        ]})
        if feat is None:
            return None
        feat['labels'] = [1]
    else:
        feat = template.encode({'messages': [
            {'role': 'user', 'content': 'Match the correct query here.'},
            {'role': 'assistant', 'content': decoded_text},
        ]})
        if feat is None:
            return None
        feat['labels'] = [0]
    return feat


def _encode_batch(
    rows: List[Dict[str, Any]],
    emb_template: Template,
) -> List[Dict[str, Any]]:
    """Encode pre-compressed texts into embedding features."""
    features: List[Dict[str, Any]] = []
    for row in rows:
        anchor_text = row['anchor_text']
        positive_text = row['positive_text']
        negative_texts = row.get('negative_texts') or []

        feat_q = _get_first_feature(anchor_text, emb_template, role='anchor')
        feat_c = _get_first_feature(positive_text, emb_template, role='positive')
        if not feat_q or not feat_c:
            continue
        features.append(feat_q)
        features.append(feat_c)
        for neg_text in negative_texts:
            feat_neg = _get_first_feature(neg_text, emb_template, role='positive')
            if feat_neg:
                features.append(feat_neg)
    return features


# =============================================================================
# Main training
# =============================================================================

def train():
    device_groups = [
        DeviceGroup(name='model',
                    ranks=list(range(MODEL_GPUS)),
                    device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS, groups=device_groups)

    # -- Load pre-compressed dataset ------------------------------------------
    from twinkle.dataset import Dataset as TwinkleDataset, DatasetMeta
    logger.info(f'[data] loading pre-compressed dataset from {DATASET_PATH}')
    dataset = TwinkleDataset(DatasetMeta(dataset_id=DATASET_PATH), download_mode='force_redownload')
    dataset = dataset.dataset.shuffle(seed=MIX_SHUFFLE_SEED)
    logger.info(f'[data] {len(dataset)} rows loaded')

    # -- Compute steps --------------------------------------------------------
    rows_per_step = BATCH_SIZE
    total_steps = (len(dataset) // rows_per_step) * NUM_EPOCHS
    optimizer_steps = total_steps // GRADIENT_ACCUMULATION_STEPS

    # -- Model ----------------------------------------------------------------
    model = build_model(model_mesh)
    model.set_processor(InputProcessor)
    model.set_loss(InfonceLoss, temperature=TEMPERATURE, use_batch=True,
                   hard_negatives=HARD_NEGATIVES)
    setup_optimizer(model, optimizer_steps)
    model.add_metric(EmbeddingMetric, is_training=True)

    emb_template = Qwen3_5Template(
        model_id=MODEL_ID, max_length=EMB_MAX_LENGTH,
        enable_thinking=False, truncation_strategy='delete')

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {total_steps}, optimizer steps: {optimizer_steps}')

    swanlab.init(project='twinkle', config={
        'backend': BACKEND,
        'model_id': MODEL_ID,
        'batch_size': BATCH_SIZE,
        'lr': LEARNING_RATE,
        'temperature': TEMPERATURE,
        'emb_max_length': EMB_MAX_LENGTH,
        'dataset_path': DATASET_PATH,
    })

    # -- Train loop -----------------------------------------------------------
    cur_step = 0
    _skip_rows = RESUME_STEP * rows_per_step  # approximate rows to skip

    for epoch in range(NUM_EPOCHS):
        for start in range(0, len(dataset), rows_per_step):
            if start < _skip_rows:
                continue

            batch_rows = dataset[start:start + rows_per_step]
            # HF Dataset slicing returns dict of lists; convert to list of dicts
            n_rows = len(batch_rows['anchor_text'])
            rows_list = [{k: batch_rows[k][i] for k in batch_rows}
                         for i in range(n_rows)]

            t0 = time.monotonic()
            features = _encode_batch(rows_list, emb_template)
            t_encode = time.monotonic() - t0

            if len(features) < 4:
                continue

            t1 = time.monotonic()
            model.forward_backward(inputs=features, task='embedding')
            model.clip_grad_and_step(
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
            t_train = time.monotonic() - t1
            cur_step += 1

            if cur_step % LOG_INTERVAL == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(
                    f'Epoch {epoch} Step {cur_step}/{total_steps}, '
                    f'metric: {metric} | '
                    f'encode={t_encode:.2f}s train={t_train:.2f}s')
                log_dict = {}
                for k, v in metric.items():
                    if not v:
                        continue
                    try:
                        log_dict[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                log_dict['epoch'] = epoch
                log_dict['encode_sec'] = round(t_encode, 3)
                log_dict['train_sec'] = round(t_train, 3)
                swanlab.log(log_dict, step=cur_step)
            if cur_step % SAVE_INTERVAL == 0:
                save_checkpoint(model, f'step_{cur_step}')

    save_checkpoint(model, 'last-checkpoint')
    # Force sync: resolve any pending lazy remote calls (save) before exit
    model.calculate_metric(is_training=True)
    logger.info(f'Training complete. Final step: {cur_step}')


if __name__ == '__main__':
    train()
