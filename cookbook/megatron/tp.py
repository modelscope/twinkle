from pathlib import Path

from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_ID = 'ms://swift/self-cognition'
TEMPLATE_NAME = 'Qwen3_5Template'
MODEL_NAME = 'twinkle大模型'
MODEL_AUTHOR = 'ModelScope社区'
DP_SIZE = 2
TP_SIZE = 2
PP_SIZE = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOG_INTERVAL = 5
EVAL_INTERVAL = 20
EVAL_SAMPLES = 100
TRAIN_SAMPLES = 1000

OUTPUT_DIR = './output/megatron_tp'
RESUME_FROM_CHECKPOINT = None
RESUME_ONLY_MODEL = False
IGNORE_DATA_SKIP = False
ADAPTER_NAME = 'default'

device_mesh = DeviceMesh.from_sizes(dp_size=DP_SIZE, tp_size=TP_SIZE, pp_size=PP_SIZE)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def build_dataset(num_samples: int) -> Dataset:
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(num_samples)))
    dataset.set_template(TEMPLATE_NAME, model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor(MODEL_NAME, MODEL_AUTHOR))
    dataset.encode()
    return dataset


def save_checkpoint(model: MegatronModel, checkpoint_name: str, dataloader: DataLoader):
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
    return model.calculate_metric(is_training=False)


def train():
    dataset = build_dataset(TRAIN_SAMPLES)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    model = MegatronModel(model_id=MODEL_ID)

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Comment this to use full-parameter training
    model.add_adapter_to_model(ADAPTER_NAME, lora_config)
    model.set_optimizer(optimizer_cls='default', lr=LEARNING_RATE)
    model.set_lr_scheduler(scheduler_cls='default', lr_warmup_steps=5, lr_decay_steps=len(dataloader))

    start_step = 0
    if RESUME_FROM_CHECKPOINT:
        checkpoint_path = Path(RESUME_FROM_CHECKPOINT).expanduser().resolve()
        kwargs = {}
        if ADAPTER_NAME:
            kwargs['adapter_name'] = ADAPTER_NAME
        progress = model.resume_from_checkpoint(
            str(checkpoint_path), resume_only_model=RESUME_ONLY_MODEL, **kwargs)
        if not IGNORE_DATA_SKIP:
            dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
            start_step = progress['cur_step']

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')

    best_loss = float('inf')

    for step, batch in enumerate(dataloader, start=start_step):
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        if step % LOG_INTERVAL == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        if step > 0 and step % EVAL_INTERVAL == 0:
            metrics = evaluate(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = step
            current_loss = float(metrics['loss'])
            if current_loss < best_loss:
                save_checkpoint(model, f'checkpoint-{step}', dataloader)
                best_loss = current_loss
    save_checkpoint(model, 'last-checkpoint', dataloader)


if __name__ == '__main__':
    train()
