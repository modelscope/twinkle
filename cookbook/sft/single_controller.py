from functools import partial

import numpy as np
from peft import LoraConfig

import twinkle
from twinkle import get_device_placement, get_logger, DeviceGroup, Platform, DeviceMesh
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel

logger = get_logger()

device_group = [
    DeviceGroup(
        name='model',
        ranks=[0,1,2,3],
        device_type=Platform.get_platform().device_prefix(),
    )
]


device_mesh = DeviceMesh(
    device_type='cuda',
    mesh=np.array([[0,1], [2,3]]),
    mesh_dim_names=('dp', 'fsdp')
)

twinkle.initialize(mode='ray', nproc_per_node=4, groups=device_group, global_device_mesh=device_mesh)


def create_dataset(data_slice=None):
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition',data_slice=data_slice))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', truncation_strategy='left', max_length=1024)
    dataset.map('SelfCognitionProcessor')
    dataset.encode(batched=True)
    return dataset


def eval(model: TransformersModel):
    dataloader = DataLoader(dataset=partial(create_dataset, data_slice=range(20)), batch_size=8, drop_last=True)
    for step, batch in enumerate(dataloader):
        print(step, len(batch))
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric()
    return metrics()

def train():
    dataloader = DataLoader(dataset=partial(create_dataset, data_slice=None), batch_size=8, remote_group='model')

    model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', remote_group='model')

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=16)
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    loss_metric = 99.0
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch)
        if step % 16 == 0:
            logger.info(f'Current is step {step // 16}, loss: {output()}')
        model.clip_grad_and_step()
        if step % 50 == 0 and step > 0:
            metrics = eval(model)
            logger.info(f'Current is step {step // 16}, metrics: {metrics}')
            metrics['step'] = step
            if loss_metric > metrics['loss']:
                model.save(f'checkpoint-{step}')
                loss_metric = metrics['loss']


if __name__ == '__main__':
    train()
