# Twinkle Client - Pointwise Reranker Training Example
#
# Trains a reranker that gives every (query, document) pair one relevance score:
#
#   add_adapter_to_model(LoRA on the backbone + the 'score' head)
#     -> set_processor('InputProcessor')
#     -> set_loss('PointwiseRerankerLoss')      # binary cross entropy on the score
#     -> set_optimizer('Adam', ...)
#     -> loop: forward_backward(inputs=batch, task='reranker') + clip_grad_and_step
#     -> save(...)
#
# Data format: query and document are joined into ONE text per sample, and
# ``labels`` = [1] for a relevant document or [0] for an irrelevant one (int only).
#
# Before running (from the repository root):
#   1. python cookbook/client/twinkle/prepare_seq_cls_model.py --num-labels 1 --output-dir output/Qwen3.5-4B-reranker
#   2. ray start --head --num-gpus=2
#   3. python -m twinkle.server launch --config cookbook/client/server/transformer/server_config_seq_cls.yaml

import dotenv
import os
from peft import LoraConfig

from twinkle import get_logger
from twinkle import init_twinkle_client
from twinkle.template import Qwen3_5Template

dotenv.load_dotenv('.env')
logger = get_logger()

# ========== Configuration ==========
MODEL = 'Qwen3.5-4B-reranker'           # the model route in server_config_seq_cls.yaml
TOKENIZER_ID = 'ms://Qwen/Qwen3.5-4B'   # the prepared base keeps the original tokenizer
LEARNING_RATE = 1e-4
MAX_STEPS = 10

# Each query has one relevant and one irrelevant document.
SAMPLES = [
    ('What is the capital of France?', 'Paris is the capital city of France.', 1),
    ('What is the capital of France?', 'Bananas are rich in potassium.', 0),
    ('How do plants make their food?', 'Plants use photosynthesis to produce food from sunlight.', 1),
    ('How do plants make their food?', 'The train arrived at the station on time.', 0),
    ('Who wrote the play Hamlet?', 'William Shakespeare wrote the tragedy Hamlet.', 1),
    ('Who wrote the play Hamlet?', 'Water boils at 100 degrees Celsius at sea level.', 0),
    ('What language is spoken in Brazil?', 'The main language spoken in Brazil is Portuguese.', 1),
    ('What language is spoken in Brazil?', 'A dog barked at the mailman.', 0),
]


def build_batch(tokenizer):
    """One feature per (query, document) pair; ``labels`` is its 0/1 relevance."""
    batch = []
    for query, document, relevance in SAMPLES:
        input_ids = tokenizer.encode(f'Query: {query}\nDocument: {document}')
        batch.append({
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': [relevance],
        })
    return batch


def train():
    # Step 1: connect to the running Twinkle server.
    client = init_twinkle_client(
        base_url=os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000'),
        api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
    )

    # Step 2: create a LoRA adapter. 'score' is the one-output relevance head: give it
    # its own LoRA too, otherwise the head stays at its random initialization.
    model = client.model(MODEL)
    model.add_adapter_to_model('reranker', LoraConfig(target_modules=['q_proj', 'v_proj', 'score']))
    model.set_template('Template')

    # Step 3: configure processor, loss and optimizer.
    model.set_processor('InputProcessor')
    model.set_loss('PointwiseRerankerLoss')
    model.set_optimizer('Adam', lr=LEARNING_RATE)

    # Step 4: tokenize the pairs locally.
    tokenizer = Qwen3_5Template(model_id=TOKENIZER_ID).tokenizer
    batch = build_batch(tokenizer)

    # Step 5: training loop. task='reranker' returns one score per pair.
    for step in range(MAX_STEPS):
        model.forward_backward(inputs=batch, task='reranker')
        model.clip_grad_and_step(max_grad_norm=1.0)
        metric = model.calculate_metric(is_training=True)
        logger.info(f'Step {step}: {metric.result}')

    # Step 6: relevant documents should now score higher, then save the adapter.
    scores = model.forward_only(inputs=batch, task='reranker', return_logits=True).result['logits']
    for (query, document, relevance), score in zip(SAMPLES, scores):
        logger.info(f'relevance={relevance} score={float(score[0]):+.3f} | {query} -> {document}')
    saved = model.save('reranker-final', save_optimizer=True)
    logger.info(f'Saved checkpoint: {saved.twinkle_path}')


if __name__ == '__main__':
    train()
