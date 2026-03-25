import math
from functools import partial

import numpy as np
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger,Platform
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.model.transformers.models import TwinkleQwen3_5ForCausalLM
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASETS = 'ms://swift/self-cognition'

device_group = [DeviceGroup(
    name='default',
    ranks=[0, 1, 2, 3],
    device_type=Platform.get_platform().device_prefix(),
)]

# FSDP + SP validation over 4 GPUs: dp=2, fsdp=2 (SP only affects input slicing)
device_mesh = DeviceMesh(
    device_type='cuda',
    mesh=np.arange(4).reshape(2, 2),
    mesh_dim_names=('dp', 'fsdp'),
    ulysses_size=2,
)

twinkle.initialize(
    mode='local',
    nproc_per_node=4,
    global_device_mesh=device_mesh,
    lazy_collect=False,
)


def eval(model):
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=range(100)),
        batch_size=4,
        device_mesh=device_mesh,
    )
    for _, batch in enumerate(dataloader):
        model.forward_only(inputs=batch, adapter_name='default')
        model.calculate_loss(adapter_name='default')
    return model.calculate_metric(is_training=False, adapter_name='default')


def create_dataset(data_slice=None):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASETS, data_slice=range(500)))
    dataset.set_template('Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    return dataset


def train():
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=None),
        batch_size=8,
        device_mesh=device_mesh,
    )

    model = TransformersModel(
        model_id=MODEL_ID,
        model_cls=TwinkleQwen3_5ForCausalLM,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        attn_implementation='flash_attention_2'
    )

    lora_config = LoraConfig(target_modules='all-linear', lora_dropout=0.0)
    model.add_adapter_to_model('default', lora_config)
    grad_accumulation_steps = model.optimizer_group['default'].gradient_accumulation_steps
    num_optimizer_steps = math.ceil(len(dataloader) / grad_accumulation_steps)
    log_every_optimizer_steps = 20
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=num_optimizer_steps,
        adapter_name='default',
    )

    logger.info(model.get_train_configs(adapter_name='default'))
    logger.info(
        f'Total micro steps: {len(dataloader)}, optimizer steps: {num_optimizer_steps}, '
        f'gradient_accumulation_steps: {grad_accumulation_steps}')

    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch, adapter_name='default')
        model.clip_grad_and_step(adapter_name='default')
        optimizer_step = step // grad_accumulation_steps
        is_optimizer_boundary = (step + 1) % grad_accumulation_steps == 0
        if is_optimizer_boundary and optimizer_step % log_every_optimizer_steps == 0:
            metric = model.calculate_metric(is_training=True, adapter_name='default')
            optimizer_step = metric.get('iters', optimizer_step)
            logger.info(
                f'Current is optimizer step {optimizer_step} of {num_optimizer_steps} '
                f'(micro step {step} of {len(dataloader)}), metric: {metric}')
    model.save('last-checkpoint', interval=1)


if __name__ == '__main__':
    train()
