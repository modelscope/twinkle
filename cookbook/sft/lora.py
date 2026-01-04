from torch.optim.lr_scheduler import LinearLR
import numpy as np
from twinkle import get_device_placement, get_logger, DeviceMesh
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CrossEntropyLoss
from twinkle.model import TransformersModel
from peft import LoraConfig
from torch.optim import AdamW
from torch.distributed.fsdp import ShardingStrategy
from twinkle.processor import InputProcessor

logger = get_logger()


#device_mesh = DeviceMesh(
#    device_type='cuda',
#    mesh=np.array([[0,1], [2,3]]),
#    mesh_dim_names=('dp', 'fsdp')
#)

device_mesh = DeviceMesh(
    device_type='cuda',
    mesh=np.array([0,1,2,3]),
    mesh_dim_names=('dp',)
)


def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset, batch_size=8, device_mesh=device_mesh)

    fsdp_config = {
        'sharding_strategy': ShardingStrategy.FULL_SHARD,
    }
    model = TransformersModel(pretrained_model_name_or_path='ms://Qwen/Qwen2.5-7B-Instruct', device_mesh=device_mesh, fsdp_config=fsdp_config)

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=16)
    model.set_template('Qwen3Template', adapter_name='default')
    model.set_processor(InputProcessor, adapter_name='default', padding_side='right')
    model.set_loss(CrossEntropyLoss, adapter_name='default')
    model.set_optimizer(AdamW, lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(LinearLR, adapter_name='default')
    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name='default'))
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch, adapter_name='default')
        if step % 16 == 0:
            logger.info(f'Current is step {step // 16}, loss: {output["loss"]}')
        model.clip_grad_norm(1.0, adapter_name='default')
        model.step(adapter_name='default')
        model.zero_grad(adapter_name='default')
        model.lr_step(adapter_name='default')
        if step % 50 == 0:
            model.save('./output', adapter_name='default')


if __name__ == '__main__':
    train()
