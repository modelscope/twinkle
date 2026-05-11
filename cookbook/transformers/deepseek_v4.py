import os

import torch.distributed as dist
import twinkle
from peft import LoraConfig
from transformers import AutoConfig
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'ms://deepseek-ai/DeepSeek-V4-flash-bfa16')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'DeepseekV4Template')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output')

_num_layers_env = os.environ.get('NUM_LAYERS','1')
NUM_LAYERS = int(_num_layers_env) if _num_layers_env is not None else None

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '2'))
LR = float(os.environ.get('LR', '1e-4'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '0'))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', '50'))
USE_LORA = os.environ.get('USE_LORA', '1') == '1'
IGNORE_MISMATCHED_SIZES = os.environ.get('IGNORE_MISMATCHED_SIZES', '1') == '1'
GRADIENT_CHECKPOINTING = os.environ.get('GRADIENT_CHECKPOINTING', '1') == '1'
RESHARD_AFTER_FORWARD = os.environ.get('RESHARD_AFTER_FORWARD', '1') == '1'
LORA_TARGET_MODULES = os.environ.get(
    'LORA_TARGET_MODULES',
    'wq_a,wq_b,wkv,wgate,gate_proj,up_proj,down_proj',
)

device_mesh = DeviceMesh.from_sizes(
    fsdp_size=4,
    dp_size=1,
    ep_size=4,
    device_type=Platform.get_platform().device_prefix(),
)

twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def barrier_if_distributed(stage: str):
    if not (dist.is_available() and dist.is_initialized()):
        return
    if os.environ.get('TWINKLE_FSDP_DEBUG', '0') == '1':
        print(f'[twinkle-train-debug][rank{dist.get_rank()}] before barrier: {stage}', flush=True)
    if dist.get_backend() == 'nccl':
        dist.barrier(device_ids=[Platform.get_local_rank()])
    else:
        dist.barrier()
    if os.environ.get('TWINKLE_FSDP_DEBUG', '0') == '1':
        print(f'[twinkle-train-debug][rank{dist.get_rank()}] after barrier: {stage}', flush=True)


def train_debug(message: str):
    if os.environ.get('TWINKLE_FSDP_DEBUG', '0') != '1':
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else Platform.get_rank()
    local_rank = Platform.get_local_rank()
    print(f'[twinkle-train-debug][rank{rank} local_rank={local_rank}] {message}', flush=True)


def describe_batch(batch):
    if isinstance(batch, dict):
        return f'dict_keys={list(batch.keys())}'
    if isinstance(batch, (list, tuple)):
        return f'{type(batch).__name__}[len={len(batch)}]'
    return type(batch).__name__


def log_expert_parallel_status(model):
    logger.info(
        f'EP flags: enabled={getattr(model, "_enable_expert_parallel", None)}, '
        f'applied={getattr(model, "_expert_parallel_applied", None)}')
    raw_model = model.strategy.unwrap_model(model.model)
    found = False
    for name, module in raw_model.named_modules():
        if not hasattr(module, '_ep_patched'):
            continue
        found = True
        logger.info(
            'EP block %s: patched=%s rank=%s/%s local_experts=[%s, %s) experts_per_rank=%s',
            name,
            getattr(module, '_ep_patched', None),
            getattr(module, '_ep_rank', None),
            getattr(module, '_ep_world_size', None),
            getattr(module, '_ep_local_start', None),
            getattr(module, '_ep_local_end', None),
            getattr(module, '_ep_experts_per_rank', None),
        )
    if not found:
        logger.info('No EP-patched MoE blocks found on the wrapped model.')


def create_dataset(data_slice=None):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=data_slice or range(1000)))
    dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode(batched=True)
    return dataset


def eval(model):
    dataset = create_dataset(data_slice=range(100))
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    for _, batch in enumerate(dataloader):
        model.forward_only(inputs=batch, adapter_name='default')
        model.calculate_loss(adapter_name='default')
    return model.calculate_metric(is_training=False, adapter_name='default')


def train():
    dataset = create_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if NUM_LAYERS is not None and hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = NUM_LAYERS
    if hasattr(config, 'use_cache'):
        config.use_cache = False

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        strategy="accelerate",
        memory_efficient_init=True,
        ignore_mismatched_sizes=IGNORE_MISMATCHED_SIZES,
        # fsdp_config={
        #     'reshard_after_forward': RESHARD_AFTER_FORWARD,
        #     'expert_parallel': {
        #         'enabled': True,
        #         'router_dtype': 'fp32',
        #         'keep_router_logits': False,
        #     }
        # },
    )

    if USE_LORA:
        lora_target_modules = [name.strip() for name in LORA_TARGET_MODULES.split(',') if name.strip()]
        lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=lora_target_modules)
        model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=GRAD_ACCUM_STEPS)

    if not GRADIENT_CHECKPOINTING:
        model.model.gradient_checkpointing_disable()

    model.set_template(TEMPLATE_ID, model_id=MODEL_ID, adapter_name='default')
    model.set_optimizer('AdamW', lr=LR, foreach=False, adapter_name='default')
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
        adapter_name='default',
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name='default'))
    logger.info(
        f'Total steps: {len(dataloader)}, batch_size={BATCH_SIZE}, '
        f'grad_accum={GRAD_ACCUM_STEPS}, lr={LR:.2e}, use_lora={USE_LORA}, '
        f'num_layers={NUM_LAYERS}, ignore_mismatched_sizes={IGNORE_MISMATCHED_SIZES}, '
        f'gradient_checkpointing={GRADIENT_CHECKPOINTING}, '
        f'reshard_after_forward={RESHARD_AFTER_FORWARD}, '
        f'lora_target_modules={LORA_TARGET_MODULES}')

    barrier_if_distributed('before first train step')

    best_loss = float('inf')
    for step, batch in enumerate(dataloader):
        if MAX_STEPS and step >= MAX_STEPS:
            break
        if step < 2:
            train_debug(f'step={step} before forward_backward batch={describe_batch(batch)}')
        model.forward_backward(
            inputs=batch,
            adapter_name='default',
        )
        if step < 2:
            train_debug(f'step={step} after forward_backward')
        model.clip_grad_and_step(
            adapter_name='default',
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        )
        if step < 2:
            train_debug(f'step={step} after clip_grad_and_step')
        if step == 0:
            log_expert_parallel_status(model)

        if step % 20 == 0:
            metric = model.calculate_metric(is_training=True, adapter_name='default')
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')

        if step > 0 and step % SAVE_STEPS == 0:
            metrics = eval(model)
            logger.info(f'Eval metric: {metrics}')
            loss = float(metrics['loss'])
            if loss < best_loss:
                model.save(name=f'checkpoint-{step}', output_dir=OUTPUT_DIR, adapter_name='default')
                best_loss = loss

    model.save(name='last-checkpoint', output_dir=OUTPUT_DIR, adapter_name='default')


if __name__ == '__main__':
    train()
