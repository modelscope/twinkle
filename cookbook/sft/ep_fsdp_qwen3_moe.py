# Copyright (c) ModelScope Contributors. All rights reserved.
import os

import numpy as np
import torch.distributed as dist
from transformers import AutoConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel

logger = get_logger()

MODEL_ID = os.environ.get(
    'QWEN3_MODEL_ID', 'ms://Qwen/Qwen3-30B-A3B-Instruct-2507')
DATASET_ID = os.environ.get(
    'QWEN3_DATASET_ID', '/path/to/alpaca/dataset')
TEMPLATE_ID = os.environ.get('QWEN3_TEMPLATE_ID', 'Template')
PROCESSOR_ID = os.environ.get('QWEN3_PROCESSOR_ID', 'AlpacaProcessor')
REMOTE_GROUP = 'model'
NUM_LAYERS = int(os.environ.get('QWEN3_NUM_LAYERS', '1'))

device_group = [
    DeviceGroup(
        name=REMOTE_GROUP,
        ranks=[0, 1, 2, 3],
        device_type=Platform.get_platform().device_prefix(),
    )
]

# 4 GPUs: dp=2, ep=2
device_mesh = DeviceMesh(
    device_type=Platform.get_platform().device_prefix(),
    mesh=np.array([[0, 1], [2, 3]]),
    mesh_dim_names=('dp', 'ep'),
)

os.environ.setdefault("RAY_DEDUP_LOGS", "0")
twinkle.initialize(
    mode='ray',
    nproc_per_node=4,
    groups=device_group,
    global_device_mesh=device_mesh,
)


def train():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(config, "num_hidden_layers"):
        original_layers = config.num_hidden_layers
        if NUM_LAYERS < original_layers:
            os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
            os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
            try:
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error()
            except Exception:
                pass
        config.num_hidden_layers = NUM_LAYERS
    if hasattr(config, "use_cache"):
        config.use_cache = False

    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID))
    try:
        dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    except ValueError:
        # Fallback to built-in Template when a plugin id is not available.
        dataset.set_template('Template', model_id=MODEL_ID)

    processor = PROCESSOR_ID
    if PROCESSOR_ID.lower() == 'alpaca':
        processor = 'AlpacaProcessor'

    dataset.map(processor)
    dataset.encode(batched=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        remote_group=REMOTE_GROUP,
        device_mesh=device_mesh,
    )

    grad_accum_steps = 4
    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        remote_group=REMOTE_GROUP,
        device_mesh=device_mesh,
        fsdp_config={
            "expert_parallel": {
                "enabled": True,
                "router_dtype": "fp32",
                "all_to_all": "torch",
                "keep_router_logits": False,
            }
        },
    )
    # Disable foreach to avoid DTensor mixed-type errors in EP runs.
    model.set_optimizer('AdamW', foreach=False)

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())

    for step, batch in enumerate(dataloader):
        if callable(batch):
            batch = batch()
        output = model.forward_backward(
            inputs=batch, gradient_accumulation_steps=grad_accum_steps)
        if step % grad_accum_steps == 0:
            logger.info(
                f'Current is step {step // grad_accum_steps}, loss: {output()}')
        model.clip_grad_and_step(gradient_accumulation_steps=grad_accum_steps)
        if step % 50 == 0:
            model.save('./output')


if __name__ == '__main__':
    train()
