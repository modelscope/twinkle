import os

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.preprocessor import SelfCognitionProcessor

# Build a device mesh for the verified NPU LoRA smoke.
MODEL_ID = os.environ.get('TWINKLE_LOCAL_MODEL_DIR', 'ms://Qwen/Qwen3-4B')
DATASET_PATH = os.environ.get(
    'TWINKLE_LOCAL_DATASET_PATH',
    'ms://swift/self-cognition',
)
MAX_STEPS = int(os.environ.get('TWINKLE_MAX_STEPS', '10'))
TRAIN_SAMPLES = int(os.environ.get('TWINKLE_TRAIN_SAMPLE_LIMIT', '160'))
BATCH_SIZE = int(os.environ.get('TWINKLE_BATCH_SIZE', '16'))

# 8 cards: dp=2, tp=2, pp=2
device_mesh = DeviceMesh.from_sizes(dp_size=2, tp_size=2, pp_size=2)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


def build_dataloader() -> DataLoader:
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_PATH, data_slice=range(TRAIN_SAMPLES)))
    dataset.set_template('Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    return DataLoader(dataset=dataset, batch_size=BATCH_SIZE)


def train():
    dataloader = build_dataloader()

    model = MegatronModel(model_id=MODEL_ID)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config)
    model.set_optimizer(optimizer_cls='default', lr=1e-4)

    # Keep the scheduler compatible with the shortened smoke run.
    lr_decay_steps = max(MAX_STEPS, 2)
    model.set_lr_scheduler(
        scheduler_cls='default',
        lr_warmup_steps=1,
        lr_decay_steps=lr_decay_steps,
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(
        'LoRA NPU smoke config: '
        f'model_id={MODEL_ID}, dataset={DATASET_PATH}, batch_size={BATCH_SIZE}, '
        f'train_samples={TRAIN_SAMPLES}, max_steps={MAX_STEPS}'
    )
    logger.info(f'dataloader_steps={len(dataloader)}')

    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        metric = model.calculate_metric(is_training=True)
        logger.info(f'step={step} metric={metric}')
        if step + 1 >= MAX_STEPS:
            break

    model.save('last-checkpoint')


if __name__ == '__main__':
    train()
