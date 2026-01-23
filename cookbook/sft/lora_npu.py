import os
import numpy as np
from peft import LoraConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

import twinkle
from twinkle import get_device_placement, get_logger, DeviceMesh, DeviceGroup, Platform
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CrossEntropyLoss
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor

logger = get_logger()

# Runtime defaults for local NPU training.
os.environ.setdefault('TRUST_REMOTE_CODE', '1')
os.environ.setdefault('TWINKLE_SEED', '42')
os.environ.setdefault('TWINKLE_FULL_DETERMINISM', '1')

MODEL_PATH = 'ms://Qwen/Qwen2.5-7B-Instruct'
NPROC_PER_NODE = 4
BATCH_SIZE = 8
device_group = [
    DeviceGroup(
        name='model',
        ranks=[0,1,2,3],
        device_type=Platform.get_platform().device_prefix(),
    )
]


device_mesh = DeviceMesh(
    device_type='npu',
    mesh=np.array([[0,1], [2,3]]),
    mesh_dim_names=('dp', 'fsdp')
)

twinkle.initialize(
    mode='ray',
    nproc_per_node=NPROC_PER_NODE,
    groups=device_group,
    global_device_mesh=device_mesh,
    lazy_collect=False,
)


def create_dataset():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', model_id=MODEL_PATH)
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    return dataset


def train():
    dataset = create_dataset()
    # Use the model group so the DataLoader is sharded across NPU ranks.
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, remote_group='model')

    model = TransformersModel(model_id=MODEL_PATH, remote_group='model')

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=16)
    model.set_template('Qwen3Template')
    model.set_processor(InputProcessor, padding_side='right')
    model.set_loss(CrossEntropyLoss)
    model.set_optimizer(AdamW, lr=1e-4)
    model.set_lr_scheduler(LinearLR)
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch)
        if step % 16 == 0:
            logger.info(f'Current is step {step // 16}, loss: {output}')
        model.clip_grad_norm(1.0)
        model.step()
        model.zero_grad()
        model.lr_step()
        if step % 50 == 0:
            model.save('./output')


if __name__ == '__main__':
    train()
