from functools import partial
import os

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig

import twinkle
from twinkle import get_device_placement, get_logger, DeviceGroup, Platform, DeviceMesh
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
MODEL_ID = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

# Local mode needs explicit process group init (torchrun provides env vars).
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="env://")

# FSDP + SP validation over 4 GPUs: dp=2, fsdp=2 (SP only affects input slicing)
device_mesh = DeviceMesh(
    device_type="cuda",
    mesh=np.arange(4).reshape(2, 2),
    mesh_dim_names=("dp", "fsdp"),
    ulysses_size=2,
)

twinkle.initialize(
    mode="local",
    global_device_mesh=device_mesh,
    lazy_collect=False,
)


def log_cuda_env(tag: str) -> None:
    try:
        rank = os.environ.get("RANK", "unknown")
        logger.info(
            f"[cuda] {tag} rank={rank} visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    except Exception as exc:
        logger.warning(f"[cuda] {tag} failed to log env: {exc}")


def create_dataset(data_slice=None):
    log_cuda_env("create_dataset")
    logger.info(f"[dataset] create_dataset start, data_slice={data_slice}")
    dataset = Dataset(
        dataset_meta=DatasetMeta("ms://swift/self-cognition", data_slice=data_slice)
    )
    dataset.set_template(
        "Template",
        model_id=MODEL_ID,
        truncation_strategy="left",
        max_length=64,
    )
    dataset.map(SelfCognitionProcessor("twinkle模型", "twinkle团队"))
    dataset.encode(batched=True)
    logger.info("[dataset] encode done")
    return dataset


def eval(model: TransformersModel):
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=range(20)),
        batch_size=4,
        drop_last=True,
        device_mesh=device_mesh,
        remote_group="model",
    )
    for step, batch in enumerate(dataloader):
        model.forward_only(inputs=batch, adapter_name="default")
        model.calculate_loss(adapter_name="default")
    metrics = model.calculate_metric(is_training=False, adapter_name="default")
    return metrics()


def train():
    log_cuda_env("train_start")
    logger.info("[train] init dataloader")
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=None),
        batch_size=4,
        device_mesh=device_mesh,
    )

    logger.info("[train] init model")
    log_cuda_env("before_model_init")
    model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=device_mesh,
        strategy="native_fsdp",
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(target_modules="all-linear")
    model.add_adapter_to_model("default", lora_config, gradient_accumulation_steps=1)
    model.set_optimizer("AdamW", lr=1e-4, adapter_name="default")
    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name="default"))

    loss_metric = 99.0
    logger.info("[train] start iter")
    for step, batch in enumerate(dataloader):
        if isinstance(batch, list) and len(batch) == 0:
            logger.warning(f"[train] empty batch at step {step}, skip")
            continue
        if step == 0:
            logger.info("[train] first batch received")
        logger.info(f"[train] step {step} forward_backward start")
        output = model.forward_backward(inputs=batch, adapter_name="default")
        logger.info(f"[train] step {step} forward_backward done")
        loss_value = output() if callable(output) else output
        logger.info(f"step {step}, loss: {loss_value}")
        model.clip_grad_and_step(adapter_name="default")
        if step % 50 == 0 and step > 0:
            metrics = eval(model)
            logger.info(f"Current is step {step // 16}, metrics: {metrics}")
            metrics["step"] = step
            if loss_metric > metrics["loss"]:
                model.save(f"checkpoint-{step}")
                loss_metric = metrics["loss"]


if __name__ == "__main__":
    train()
