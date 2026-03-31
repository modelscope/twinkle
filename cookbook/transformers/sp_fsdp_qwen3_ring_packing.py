import os
from functools import partial

import numpy as np
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import DatasetMeta, PackingDataset
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.processor import InputProcessor

logger = get_logger()

# This cookbook is for packed-input sequence-parallel / ring-attention experiments.
# It uses Qwen3-0.6B with FlashAttention2 and a PackingDataset so each local SP rank
# sees one packed sample instead of a padded dense batch.
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3-0.6B')
DATASETS = os.environ.get('DATASETS', 'ms://swift/self-cognition')
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', '2048'))
TRAIN_SLICE = int(os.environ.get('TRAIN_SLICE', '500'))
EVAL_SLICE = int(os.environ.get('EVAL_SLICE', '100'))
PACKING_NUM_PROC = int(os.environ.get('PACKING_NUM_PROC', '1'))

# Global batch size is sharded by dp/fsdp and then consumed by ulysses groups.
# With dp=2, fsdp=2, ulysses_size=2, a global batch of 2 becomes one packed sample
# per data rank, which satisfies derived-ring's local batch_size == 1 requirement.
TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE', '2'))
EVAL_BATCH_SIZE = int(os.environ.get('EVAL_BATCH_SIZE', '2'))

device_group = [DeviceGroup(
    name='default',
    ranks=[0, 1, 2, 3],
    device_type=Platform.get_platform().device_prefix(),
)]

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


def create_packed_dataset(data_slice=None):
    if data_slice is None:
        data_slice = range(TRAIN_SLICE)
    dataset = PackingDataset(
        dataset_meta=DatasetMeta(DATASETS, data_slice=data_slice),
        packing_num_proc=PACKING_NUM_PROC,
    )
    dataset.set_template('Template', model_id=MODEL_ID, max_length=MAX_LENGTH)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    dataset.pack_dataset()
    return dataset


def build_model():
    model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        attn_implementation='flash_attention_2',
    )
    model.set_processor(InputProcessor, padding_free=True)

    lora_config = LoraConfig(target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')
    return model


def eval(model):
    dataloader = DataLoader(
        dataset=partial(create_packed_dataset, data_slice=range(EVAL_SLICE)),
        batch_size=EVAL_BATCH_SIZE,
        device_mesh=device_mesh,
    )
    for _, batch in enumerate(dataloader):
        model.forward_only(inputs=batch, adapter_name='default')
        model.calculate_loss(adapter_name='default')
    return model.calculate_metric(is_training=False, adapter_name='default')


def train():
    dataloader = DataLoader(
        dataset=partial(create_packed_dataset, data_slice=range(TRAIN_SLICE)),
        batch_size=TRAIN_BATCH_SIZE,
        device_mesh=device_mesh,
    )

    model = build_model()
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
        adapter_name='default',
    )

    logger.info(model.get_train_configs(adapter_name='default'))
    logger.info(f'Total steps: {len(dataloader)}')
    logger.info(
        'Packed ring-attention cookbook: model=%s max_length=%s train_batch_size=%s packing_num_proc=%s',
        MODEL_ID,
        MAX_LENGTH,
        TRAIN_BATCH_SIZE,
        PACKING_NUM_PROC,
    )

    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch, adapter_name='default')
        model.clip_grad_and_step(adapter_name='default')
        if step % 20 == 0:
            metric = model.calculate_metric(is_training=True, adapter_name='default')
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
    model.save('last-checkpoint', interval=1)


if __name__ == '__main__':
    train()
