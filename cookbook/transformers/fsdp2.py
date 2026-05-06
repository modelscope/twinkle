from pathlib import Path

from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_ID = 'ms://swift/self-cognition'
TEMPLATE_NAME = 'Qwen3_5Template'
MODEL_NAME = 'twinkle大模型'
MODEL_AUTHOR = 'ModelScope社区'
FSDP_SIZE = 2
DP_SIZE = 4
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 2
LOG_INTERVAL = 20
EVAL_INTERVAL = 40
EVAL_SAMPLES = 100
TRAIN_SAMPLES = 1000

OUTPUT_DIR = './output/fsdp2'
RESUME_FROM_CHECKPOINT = None
RESUME_ONLY_MODEL = False
IGNORE_DATA_SKIP = False
ADAPTER_NAME = 'default'

# Construct a device_mesh
device_mesh = DeviceMesh.from_sizes(fsdp_size=FSDP_SIZE, dp_size=DP_SIZE)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def build_dataset(num_samples: int) -> Dataset:
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(num_samples)))
    dataset.set_template(TEMPLATE_NAME, model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor(MODEL_NAME, MODEL_AUTHOR))
    dataset.encode()
    return dataset


def save_checkpoint(model: TransformersModel, checkpoint_name: str, dataloader: DataLoader):
    model.save(
        checkpoint_name,
        output_dir=OUTPUT_DIR,
        adapter_name=ADAPTER_NAME,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def evaluate(model):
    dataloader = DataLoader(dataset=build_dataset(EVAL_SAMPLES), batch_size=BATCH_SIZE)
    for batch in tqdm(dataloader):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    return model.calculate_metric(is_training=False)


def train():
    dataset = build_dataset(TRAIN_SAMPLES)
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    # Use a TransformersModel
    model = TransformersModel(model_id=MODEL_ID)
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Add a lora to model, with name `default`
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))

    if RESUME_FROM_CHECKPOINT:
        checkpoint_path = Path(RESUME_FROM_CHECKPOINT).expanduser().resolve()
        kwargs = {}
        if ADAPTER_NAME:
            kwargs['adapter_name'] = ADAPTER_NAME
        progress = model.resume_from_checkpoint(
            str(checkpoint_path), resume_only_model=RESUME_ONLY_MODEL, **kwargs)
        if not IGNORE_DATA_SKIP:
            dataloader.resume_from_checkpoint(progress['consumed_train_samples'])

    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    optimizer_group = model.optimizer_group[ADAPTER_NAME]
    best_loss = float('inf')
    # lora: 8G * 8
    # full: 18G * 8
    for batch in dataloader:
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        cur_step = optimizer_group.cur_step
        if cur_step % LOG_INTERVAL == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {cur_step} of {len(dataloader)}, metric: {metric}')
        if cur_step > 0 and cur_step % EVAL_INTERVAL == 0:
            metrics = evaluate(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = cur_step
            current_loss = float(metrics['loss'])
            if current_loss < best_loss:
                save_checkpoint(model, f'checkpoint-{cur_step}', dataloader)
                best_loss = current_loss
    save_checkpoint(model, 'last-checkpoint', dataloader)


if __name__ == '__main__':
    train()
