import numpy as np
from peft import LoraConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

import twinkle
from twinkle import get_device_placement, get_logger, DeviceMesh, DeviceGroup, Platform
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CrossEntropyLoss
from twinkle.model import MultiLoraTransformersModel
from twinkle.processor import InputProcessor

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
    mesh=np.array([0,1,2,3]),
    mesh_dim_names=('dp',)
)


twinkle.initialize(mode='local', nproc_per_node=4, groups=device_group, global_device_mesh=device_mesh, lazy_collect=False)


def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=16)
    model.set_template('Qwen3Template', adapter_name='default')
    model.set_processor(InputProcessor, padding_side='right', adapter_name='default')
    model.set_loss(CrossEntropyLoss, adapter_name='default')
    model.set_optimizer(AdamW, lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(LinearLR, adapter_name='default')
    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name='default'))
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch, adapter_name='default')
        if step % 16 == 0:
            logger.info(f'Current is step {step // 16}, loss: {output}')
        model.clip_grad_norm(1.0, adapter_name='default')
        model.step(adapter_name='default')
        model.zero_grad(adapter_name='default')
        model.lr_step(adapter_name='default')
        if step % 50 == 0:
            model.save('./output', adapter_name='default')


if __name__ == '__main__':
    train()
