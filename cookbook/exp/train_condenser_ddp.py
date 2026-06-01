"""Ray LoRA SFT for the condenser model on condense_300K.

Launch:
    python cookbook/exp/train_condenser_ddp.py
"""
from pathlib import Path

from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_ID = 'ms://twinkle-kit/condense_300K'
TEMPLATE_NAME = 'Qwen3_5Template'

DP_SIZE = 8
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 4
LOG_INTERVAL = 20
EVAL_INTERVAL = 200
EVAL_SAMPLES = 100
NUM_EPOCHS = 1

OUTPUT_DIR = './output/condenser_ddp'
RESUME_FROM_CHECKPOINT = None
RESUME_ONLY_MODEL = False
IGNORE_DATA_SKIP = False
ADAPTER_NAME = 'default'

def build_dataset(num_samples: int = None) -> Dataset:
    meta_kwargs = {'split': 'train'}
    if num_samples is not None:
        meta_kwargs['data_slice'] = range(num_samples)
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, **meta_kwargs))
    dataset.set_template(TEMPLATE_NAME, model_id=MODEL_ID, max_length=40000, enable_thinking=False, truncation_strategy='delete')
    dataset.encode(load_from_cache_file=True, num_proc=16)
    return dataset


def train():
    device_groups = [DeviceGroup(name='model', ranks=DP_SIZE, device_type='GPU')]
    model_mesh = DeviceMesh.from_sizes(world_size=DP_SIZE, dp_size=2, fsdp_size=4)
    twinkle.initialize(mode='ray', nproc_per_node=DP_SIZE, groups=device_groups, global_device_mesh=model_mesh)

    dataset = build_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=model_mesh, remote_group='model')

    model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules='all-linear')
    # model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=50, num_training_steps=len(dataloader) * NUM_EPOCHS)

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')

    for i in range(NUM_EPOCHS):
        for cur_step, batch in enumerate(dataloader):
            model.forward_backward(inputs=batch)
            model.clip_grad_and_step()
            if cur_step % LOG_INTERVAL == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(f'Step {cur_step}/{len(dataloader) * NUM_EPOCHS}, metric: {metric}')
            if cur_step % 4000 == 0:
                model.save(f'step_{cur_step}', output_dir=OUTPUT_DIR)
    model.save('last_checkpoint', output_dir=OUTPUT_DIR)


if __name__ == '__main__':
    train()
