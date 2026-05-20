"""DDP LoRA SFT for the policy on hotpotqa_distractor_reannotated_sft_12k.jsonl.

The JSONL is the output of ``cookbook/rl/make_condensed_sft.py``: each row
already carries ``messages`` (system / user / assistant with textual
``<tool_call>`` blocks / tool) plus an OpenAI-shape ``tools`` schema, ready
for ``Qwen3_5Template`` to render. ``enable_thinking=False`` matches the
RL runtime contract.

Launch:
    torchrun --nproc_per_node=8 cookbook/rl/train_condensed_sft_ddp.py
"""
from pathlib import Path

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / 'hotpotqa_distractor_reannotated_sft_12k.jsonl')
TEMPLATE_NAME = 'Qwen3_5Template'
# Multi-hop with compressed context + multi-turn extract_condensed CoT;
# raw audit: most samples land well under 16k after condensation.
MAX_LENGTH = 32000

DP_SIZE = 8
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 2
LOG_INTERVAL = 20
NUM_EPOCHS = 2

OUTPUT_DIR = './output/condensed_sft_ddp'
RESUME_FROM_CHECKPOINT = None
RESUME_ONLY_MODEL = False
IGNORE_DATA_SKIP = False
ADAPTER_NAME = 'default'

device_mesh = DeviceMesh.from_sizes(dp_size=DP_SIZE)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def build_dataset(num_samples: int = None) -> Dataset:
    meta_kwargs = {}
    if num_samples is not None:
        meta_kwargs['data_slice'] = range(num_samples)
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_PATH, **meta_kwargs))
    # ``truncation_strategy='delete'`` drops overlong rows instead of slicing —
    # a sliced multi-turn trajectory would lose `\boxed{}` and break SFT signal.
    dataset.set_template(
        TEMPLATE_NAME,
        model_id=MODEL_ID,
        max_length=MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False)
    dataset.encode(load_from_cache_file=True, num_proc=16)
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
    dataset = build_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    model = TransformersModel(model_id=MODEL_ID, ddp_config={'find_unused_parameters': True})
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=50,
        num_training_steps=len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS)

    if RESUME_FROM_CHECKPOINT:
        checkpoint_path = Path(RESUME_FROM_CHECKPOINT).expanduser().resolve()
        kwargs = {'adapter_name': ADAPTER_NAME} if ADAPTER_NAME else {}
        progress = model.resume_from_checkpoint(
            str(checkpoint_path), resume_only_model=RESUME_ONLY_MODEL, **kwargs)
        if not IGNORE_DATA_SKIP:
            dataloader.resume_from_checkpoint(progress['consumed_train_samples'])

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader) * NUM_EPOCHS}')

    optimizer_group = model.optimizer_group[ADAPTER_NAME]

    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            model.forward_backward(inputs=batch)
            model.clip_grad_and_step()
            cur_step = optimizer_group.cur_step
            if cur_step % LOG_INTERVAL == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(f'Epoch {epoch} Step {cur_step}/{len(dataloader) * NUM_EPOCHS}, metric: {metric}')
        save_checkpoint(model, f'epoch-{epoch}', dataloader)
    save_checkpoint(model, 'last-checkpoint', dataloader)


if __name__ == '__main__':
    train()
