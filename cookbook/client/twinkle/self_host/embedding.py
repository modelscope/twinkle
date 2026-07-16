# Twinkle Client - Embedding (InfoNCE contrastive) Training Example
#
# Client counterpart of the ray-local core-lib example
# ``cookbook/exp/embedding/train_embedding_full_ddp.py`` and the client section
# of ``docs/source_zh/使用指引/Embedding训练.md``. Instead of building a local
# ``TransformersModel``/``MegatronModel`` + Ray, it drives the SAME training flow
# over HTTP through the Twinkle client:
#
#   set_processor('InputProcessor')
#     -> set_loss('InfonceLoss', ...)
#     -> set_optimizer('Adam', ...)
#     -> add_metric('EmbeddingMetric', is_training=True)
#     -> loop: forward_backward(inputs=mb, task='embedding') + clip_grad_and_step
#     -> calculate_metric(is_training=True)
#
# Data format (anchor/positive contrastive pairs):
#   inputs: [anchor_0, positive_0, anchor_1, positive_1, ...]
#   labels:  [      1,          0,        1,          0, ...]
#   (labels==1 marks a new group's anchor; labels==0 marks its positive)
#
# The server must be running first (see server.py / server_config.yaml). Only the
# model service is required (no sampler). Works against both the transformers and
# megatron backends: each single-sequence feature carries ``position_ids`` so the
# megatron mrope model gets valid positions (transformers derives them internally).

import dotenv

dotenv.load_dotenv('.env')

from typing import Any, Dict, List

from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.template import Qwen3_5Template
from twinkle_client.model import MultiLoraTransformersModel

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
ADAPTER_NAME = 'emb_adapter'
TEMPERATURE = 0.07
LEARNING_RATE = 1e-4
MAX_STEPS = 10
MAX_LENGTH = 512

# Small self-contained anchor/positive corpus. Each anchor's positive shares its
# meaning, so InfoNCE has a clear contrastive signal to learn from.
PAIRS: List[tuple] = [
    ('What is the capital of France?', 'Paris is the capital city of France.'),
    ('How do plants make their food?', 'Plants use photosynthesis to produce food from sunlight.'),
    ('What is the boiling point of water?', 'Water boils at 100 degrees Celsius at sea level.'),
    ('Who wrote the play Hamlet?', 'William Shakespeare wrote the tragedy Hamlet.'),
    ('What is the speed of light?', 'Light travels at about 300,000 kilometers per second.'),
    ('What language is spoken in Brazil?', 'The main language spoken in Brazil is Portuguese.'),
]


def build_minibatch(tokenizer) -> List[Dict[str, Any]]:
    """Build one interleaved [anchor, positive, ...] embedding minibatch.

    Each sample is a plain-dict feature with:
      * ``input_ids``      -- tokenized text (plain Python list, JSON-serializable)
      * ``attention_mask`` -- all ones
      * ``labels``         -- a single scalar: 1 for an anchor, 0 for its positive
      * ``position_ids``   -- ``range(len)`` so the megatron mrope model gets valid
                              positions (transformers derives them internally)
    """
    minibatch: List[Dict[str, Any]] = []
    for anchor_text, positive_text in PAIRS:
        for text, is_anchor in ((anchor_text, True), (positive_text, False)):
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            minibatch.append({
                'input_ids': list(input_ids),
                'attention_mask': [1] * len(input_ids),
                'labels': [1 if is_anchor else 0],
                'position_ids': list(range(len(input_ids))),
            })
    return minibatch


def train():
    # Step 1: connect to the running Twinkle server.
    init_twinkle_client(base_url='http://127.0.0.1:8000', api_key='EMPTY_TOKEN')

    # Step 2: build the client model with a fresh LoRA adapter.
    model = MultiLoraTransformersModel(model_id=MODEL_ID)
    model.add_adapter_to_model(ADAPTER_NAME, LoraConfig(target_modules='all-linear'))
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)

    # Step 3: configure the embedding-training pipeline (ORDER MATTERS).
    # Pass class names as strings; the server resolves them to core-lib classes.
    model.set_processor('InputProcessor')
    model.set_loss('InfonceLoss', temperature=TEMPERATURE, use_batch=True, hard_negatives=None)
    # 'Adam' works on both transformers and megatron backends ('AdamW' is
    # transformers-only; megatron expects 'Adam'/'default'/'MegatronOptimizer').
    model.set_optimizer('Adam', lr=LEARNING_RATE)
    model.add_metric('EmbeddingMetric', is_training=True)

    # Step 4: build a small contrastive minibatch with a local tokenizer.
    template = Qwen3_5Template(model_id=MODEL_ID, max_length=MAX_LENGTH, enable_thinking=False)
    minibatch = build_minibatch(template.tokenizer)
    logger.info(f'Built minibatch: {len(minibatch)} samples ({len(PAIRS)} anchor/positive pairs)')

    # Step 5: training loop. task='embedding' selects the embedding pooling +
    # InfoNCE path on the server (TransformersEmbeddingPatch/MegatronEmbeddingPatch
    # is applied and rolled back automatically inside the forward).
    for step in range(MAX_STEPS):
        model.forward_backward(inputs=minibatch, task='embedding')
        model.clip_grad_and_step(max_grad_norm=1.0)
        metric = model.calculate_metric(is_training=True)
        metric = getattr(metric, 'result', metric)
        logger.info(f'Step {step}: {metric}')

    logger.info('Embedding training finished. Healthy training shows pos_sim rising '
                'and neg_sim stable/decreasing.')


if __name__ == '__main__':
    train()
