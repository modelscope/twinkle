import numpy as np
from peft import LoraConfig

import twinkle
from twinkle import get_device_placement, get_logger, DeviceMesh, DeviceGroup, Platform
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


# device_mesh = DeviceMesh(
#     device_type='cuda',
#     mesh=np.array([[0,1], [2,3]]),
#     mesh_dim_names=('dp', 'fsdp')
# )

device_mesh = DeviceMesh(
   device_type='cuda',
   mesh=np.array([0,1,2,3]),
   mesh_dim_names=('dp',)
)

twinkle.initialize(mode='ray', groups=device_group, global_device_mesh=device_mesh)


def create_dataset():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math'), streaming=True)
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    return dataset


def train():
    dataloader = DataLoader(dataset=create_dataset, batch_size=8, remote_group='model')

    model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', remote_group='model')

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=16)
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch)
        if step % 16 == 0:
            logger.info(f'Current is step {step // 16}, loss: {output}')
        model.clip_grad_and_step()
        if step % 50 == 0:
            model.save('./output')


if __name__ == '__main__':
    train()
