from pathlib import Path

from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

from resume_utils import resume_from_checkpoint

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_ID = 'ms://swift/self-cognition'
FSDP_SIZE = 2
DP_SIZE = 4
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 2
LOG_INTERVAL = 20
EVAL_INTERVAL = 40

OUTPUT_DIR = './output/fsdp2'
RESUME_FROM_CHECKPOINT = None
RESUME_ONLY_MODEL = False
IGNORE_DATA_SKIP = False
ADAPTER_NAME = 'default'

# Construct a device_mesh
device_mesh = DeviceMesh.from_sizes(fsdp_size=FSDP_SIZE, dp_size=DP_SIZE)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def eval(model):
    # 100 Samples
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(100)))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics


def train():
    # 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(1000)))
    # Set template to prepare encoding
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID)
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    # Use a TransformersModel
    model = TransformersModel(model_id=MODEL_ID)
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Add a lora to model, with name `default`
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))

    consumed_train_samples = 0
    if RESUME_FROM_CHECKPOINT:
        checkpoint_path = Path(RESUME_FROM_CHECKPOINT).expanduser().resolve()
        consumed_train_samples = resume_from_checkpoint(
            model=model,
            dataloader=dataloader,
            checkpoint_path=checkpoint_path,
            resume_only_model=RESUME_ONLY_MODEL,
            ignore_data_skip=IGNORE_DATA_SKIP,
            adapter_name=ADAPTER_NAME,
        )

    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    # lora: 8G * 8
    # full: 18G * 8
    for step, batch in enumerate(dataloader):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        consumed_train_samples += BATCH_SIZE
        cur_step = model.optimizer_group[ADAPTER_NAME].cur_step
        if cur_step % LOG_INTERVAL == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {cur_step} of {len(dataloader)}, metric: {metric}')
        if cur_step > 0 and cur_step % EVAL_INTERVAL == 0:
            metrics = eval(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = cur_step
            if loss_metric > float(metrics['loss']):
                model.save(
                    f'checkpoint-{cur_step}',
                    output_dir=OUTPUT_DIR,
                    adapter_name=ADAPTER_NAME,
                    save_optimizer=True,
                    consumed_train_samples=consumed_train_samples,
                )
                loss_metric = float(metrics['loss'])
    model.save(
        'last-checkpoint',
        output_dir=OUTPUT_DIR,
        adapter_name=ADAPTER_NAME,
        save_optimizer=True,
        consumed_train_samples=consumed_train_samples,
    )


if __name__ == '__main__':
    train()
