# Twinkle Client - Sequence Classification Training Example
#
# Trains a three-class text classifier (animal / fruit / vehicle) over HTTP:
#
#   add_adapter_to_model(LoRA on the backbone + the 'score' head)
#     -> set_processor('InputProcessor')
#     -> set_loss('SeqClsLoss', problem_type='single_label_classification', num_labels=3)
#     -> set_optimizer('Adam', ...)
#     -> loop: forward_backward(inputs=batch, task='seq_cls') + clip_grad_and_step
#     -> save(...)
#
# Data format: one text per sample, ``labels`` = [class_id] with an int class_id
# in [0, num_labels).
#
# Before running (from the repository root):
#   1. python cookbook/client/twinkle/prepare_seq_cls_model.py --num-labels 3 --output-dir output/Qwen3.5-4B-seq-cls
#   2. ray start --head --num-gpus=2
#   3. python -m twinkle.server launch --config cookbook/client/server/transformer/server_config_seq_cls.yaml

import dotenv
import numpy as np
import os
from peft import LoraConfig

from twinkle import get_logger
from twinkle import init_twinkle_client
from twinkle.template import Qwen3_5Template

dotenv.load_dotenv('.env')
logger = get_logger()

# ========== Configuration ==========
MODEL = 'Qwen3.5-4B-seq-cls'            # the model route in server_config_seq_cls.yaml
TOKENIZER_ID = 'ms://Qwen/Qwen3.5-4B'   # the prepared base keeps the original tokenizer
NUM_LABELS = 3                          # must match num_labels in the server config
LEARNING_RATE = 1e-4
MAX_STEPS = 10

LABELS = ['animal', 'fruit', 'vehicle']
SAMPLES = [
    ('The cat is sleeping on the sofa.', 'animal'),
    ('A dog barked at the mailman.', 'animal'),
    ('The horse galloped across the field.', 'animal'),
    ('An eagle flew over the mountains.', 'animal'),
    ('I ate a crisp red apple.', 'fruit'),
    ('Bananas are rich in potassium.', 'fruit'),
    ('She squeezed fresh oranges for juice.', 'fruit'),
    ('The grapes on the vine are ripe.', 'fruit'),
    ('He drove his car to work.', 'vehicle'),
    ('The train arrived at the station on time.', 'vehicle'),
    ('A bus stopped at the corner.', 'vehicle'),
    ('The airplane took off from the runway.', 'vehicle'),
]


def build_batch(tokenizer):
    """One feature per text; ``labels`` holds the class index of that text."""
    batch = []
    for text, label in SAMPLES:
        input_ids = tokenizer.encode(text)
        batch.append({
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': [LABELS.index(label)],
        })
    return batch


def train():
    # Step 1: connect to the running Twinkle server.
    client = init_twinkle_client(
        base_url=os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000'),
        api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
    )

    # Step 2: create a LoRA adapter. 'score' is the classification head: give it its
    # own LoRA too, otherwise the head stays at its random initialization.
    model = client.model(MODEL)
    model.add_adapter_to_model('seq_cls', LoraConfig(target_modules=['q_proj', 'v_proj', 'score']))
    model.set_template('Template')

    # Step 3: configure processor, loss and optimizer.
    model.set_processor('InputProcessor')
    model.set_loss('SeqClsLoss', problem_type='single_label_classification', num_labels=NUM_LABELS)
    model.set_optimizer('Adam', lr=LEARNING_RATE)

    # Step 4: tokenize the samples locally.
    tokenizer = Qwen3_5Template(model_id=TOKENIZER_ID).tokenizer
    batch = build_batch(tokenizer)

    # Step 5: training loop. task='seq_cls' returns one [num_labels] score row per text.
    for step in range(MAX_STEPS):
        output = model.forward_backward(inputs=batch, task='seq_cls')
        model.clip_grad_and_step(max_grad_norm=1.0)
        logger.info(f'Step {step}: loss={output.result["loss"]:.4f}')

    # Step 6: check the predictions, then save the adapter and optimizer state.
    logits = model.forward_only(inputs=batch, task='seq_cls', return_logits=True).result['logits']
    for (text, label), row in zip(SAMPLES, logits):
        logger.info(f'{text!r}: predicted={LABELS[int(np.argmax(row))]}, expected={label}')
    saved = model.save('seq-cls-final', save_optimizer=True)
    logger.info(f'Saved checkpoint: {saved.twinkle_path}')


if __name__ == '__main__':
    train()
