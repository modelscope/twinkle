import os

import twinkle
from peft import LoraConfig
from transformers import AutoConfig
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
# `deepseek-ai/DeepSeek-V4-Flash` uses mixed FP4/FP8 weights.
# Convert the checkpoint before training by following:
# https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87
# Install `transformers==5.8.0` before running this cookbook.
MODEL_ID = os.environ.get('MODEL_ID', 'ms://deepseek-ai/DeepSeek-V4-Flash')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'DeepseekV4Template')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output')

NUM_LAYERS = int(os.environ.get('NUM_LAYERS', '4'))

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '2'))
LR = float(os.environ.get('LR', '1e-4'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '0'))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', '50'))
RESHARD_AFTER_FORWARD = os.environ.get('RESHARD_AFTER_FORWARD', '1') == '1'
GRADIENT_CHECKPOINTING = True
IGNORE_MISMATCHED_SIZES = False
LORA_TARGET_MODULES = [
    'q_a_proj',
    'q_b_proj',
    'kv_proj',
    'o_b_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]
ADAPTER_NAME = 'default'

device_mesh = DeviceMesh.from_sizes(
    fsdp_size=8,
    dp_size=1,
    ep_size=8,
    device_type=Platform.get_platform().device_prefix(),
)

twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def create_dataset(data_slice=None):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=data_slice or range(1000)))
    dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode(batched=True)
    return dataset


def eval(model):
    dataset = create_dataset(data_slice=range(100))
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)
    for _, batch in enumerate(dataloader):
        if callable(batch):
            batch = batch()
        model.forward_only(inputs=batch, adapter_name=ADAPTER_NAME)
        model.calculate_loss(adapter_name=ADAPTER_NAME)
    return model.calculate_metric(is_training=False, adapter_name=ADAPTER_NAME)


def train():
    dataset = create_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if NUM_LAYERS is not None and hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = NUM_LAYERS
    if hasattr(config, 'use_cache'):
        config.use_cache = False

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        memory_efficient_init=True,
        ignore_mismatched_sizes=IGNORE_MISMATCHED_SIZES,
        fsdp_config={
            'reshard_after_forward': RESHARD_AFTER_FORWARD,
             'expert_parallel': {
             'enabled': True,
             'router_dtype': 'fp32',
             'keep_router_logits': False,
    },
        },
    )

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=LORA_TARGET_MODULES)
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRAD_ACCUM_STEPS)

    if not GRADIENT_CHECKPOINTING:
        model.model.gradient_checkpointing_disable()

    model.set_template(TEMPLATE_ID, model_id=MODEL_ID, adapter_name=ADAPTER_NAME)
    model.set_optimizer('AdamW', lr=LR, foreach=False, adapter_name=ADAPTER_NAME)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
        adapter_name=ADAPTER_NAME,
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name=ADAPTER_NAME))
    logger.info(
        f'Total steps: {len(dataloader)}, batch_size={BATCH_SIZE}, '
        f'grad_accum={GRAD_ACCUM_STEPS}, lr={LR:.2e}, '
        f'num_layers={NUM_LAYERS}, ignore_mismatched_sizes={IGNORE_MISMATCHED_SIZES}, '
        f'gradient_checkpointing={GRADIENT_CHECKPOINTING}, '
        f'reshard_after_forward={RESHARD_AFTER_FORWARD}, '
        f'lora_target_modules={LORA_TARGET_MODULES}')

    best_loss = float('inf')
    for step, batch in enumerate(dataloader):
        if MAX_STEPS and step >= MAX_STEPS:
            break
        if callable(batch):
            batch = batch()
        model.forward_backward(
            inputs=batch,
            adapter_name=ADAPTER_NAME,
        )
        model.clip_grad_and_step(
            adapter_name=ADAPTER_NAME,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        )

        if step % 20 == 0:
            metric = model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')

        if step > 0 and step % SAVE_STEPS == 0:
            metrics = eval(model)
            logger.info(f'Eval metric: {metrics}')
            loss = float(metrics['loss'])
            if loss < best_loss:
                model.save(name=f'checkpoint-{step}', output_dir=OUTPUT_DIR, adapter_name=ADAPTER_NAME)
                best_loss = loss

    model.save(name='last-checkpoint', output_dir=OUTPUT_DIR, adapter_name=ADAPTER_NAME)


if __name__ == '__main__':
    train()
