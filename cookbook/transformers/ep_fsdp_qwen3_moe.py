# Copyright (c) ModelScope Contributors. All rights reserved.
import os

import numpy as np
from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()

MODEL_ID = os.environ.get(
    'QWEN3_MODEL_ID', 'ms://Qwen/Qwen3-30B-A3B-Instruct-2507')
DATASET_ID = os.environ.get(
    'DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'Template')
_num_layers_env = os.environ.get('NUM_LAYERS')
NUM_LAYERS = int(_num_layers_env) if _num_layers_env is not None else None

# 4 gpus, dp=2, ep=2
dp_size = 2
ep_size = 2

device_mesh = DeviceMesh(
    device_type=Platform.get_platform().device_prefix(),
    mesh=np.arange(dp_size * ep_size).reshape(dp_size, ep_size),
    mesh_dim_names=("dp", "ep"),
)

twinkle.initialize(
    mode='local',
    global_device_mesh=device_mesh,
)


def train():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if NUM_LAYERS is not None and hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = NUM_LAYERS
    if hasattr(config, "use_cache"):
        config.use_cache = False

    dataset = Dataset(dataset_meta=DatasetMeta(
        'ms://swift/self-cognition', data_slice=range(1000)))
    try:
        dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    except ValueError:
        dataset.set_template('Template', model_id=MODEL_ID)

    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode(batched=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        device_mesh=device_mesh,
    )

    grad_accum_steps = 4
    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
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
        model.forward_backward(
            inputs=batch, gradient_accumulation_steps=grad_accum_steps)
        model.clip_grad_and_step(gradient_accumulation_steps=grad_accum_steps)
        if step % grad_accum_steps == 0:
            metric = model.calculate_metric(is_training=True)
            if callable(metric):
                metric = metric()
            logger.info(
                f'Current is step {step // grad_accum_steps}, metric: {metric}')
        if step > 0 and step % 50 == 0:
            model.save('./output')


if __name__ == '__main__':
    train()
