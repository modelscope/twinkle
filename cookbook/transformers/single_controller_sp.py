from functools import partial
import numpy as np
import torch
from peft import LoraConfig

import twinkle
from twinkle import get_logger, DeviceGroup, Platform, DeviceMesh
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
MODEL_ID = 'ms://Qwen/Qwen2.5-7B-Instruct'

device_group = [
    DeviceGroup(
        name="default",
        ranks=[0, 1, 2, 3],
        device_type=Platform.get_platform().device_prefix(),
    )
]

# FSDP + SP validation over 4 GPUs: dp=2, fsdp=2 (SP only affects input slicing)
device_mesh = DeviceMesh(
    device_type="cuda",
    mesh=np.arange(4).reshape(2, 2),
    mesh_dim_names=("dp", "fsdp"),
    ulysses_size=2,
)

twinkle.initialize(
    mode="local",
    nproc_per_node=4,
    global_device_mesh=device_mesh,
    lazy_collect=False,
)


def create_dataset(data_slice=None):
    dataset = Dataset(
        dataset_meta=DatasetMeta("ms://swift/self-cognition", data_slice=data_slice)
    )
    dataset.set_template(
        "Template",
        model_id=MODEL_ID,
        truncation_strategy="left",
        max_length=256,
    )
    dataset.map(SelfCognitionProcessor("twinkle模型", "twinkle团队"))
    dataset.encode(batched=True)
    return dataset



def train():
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=None),
        batch_size=4,
        device_mesh=device_mesh,
    )

    model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=device_mesh,
        strategy="native_fsdp",
    )

    lora_config = LoraConfig(target_modules="all-linear")
    model.add_adapter_to_model("default", lora_config, gradient_accumulation_steps=1)
    model.set_optimizer("AdamW", lr=1e-4, adapter_name="default")

    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch, adapter_name='default')
        model.clip_grad_and_step(adapter_name='default')
        if step % 1 == 0:
            metric = model.calculate_metric(is_training=True, adapter_name='default')
            _metrics = {}
            for key, value in metric.items():
                try:
                    value = float(value)
                    _metrics[key] = value
                except:
                    pass
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
    model.save(f'last-checkpoint', interval=1)


if __name__ == "__main__":
    train()
