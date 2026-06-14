import os
import twinkle
from peft import LoraConfig
from transformers import AutoConfig
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.kernel import apply_npu_patch
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor


logger = get_logger()
MODEL_ID = os.environ.get('DSV4_MODEL_ID', 'ms://deepseek-ai/DeepSeek-V4-Flash')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'DeepseekV4Template')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output')

MAX_LENGTH = int(os.environ.get('MAX_LENGTH', '4096'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '2'))
LOG_INTERVAL = GRAD_ACCUM_STEPS
LR = float(os.environ.get('LR', '1e-5'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '0'))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', '50'))
USE_LORA = os.environ.get('USE_LORA', '1') == '1'
MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
IGNORE_MISMATCHED_SIZES = os.environ.get('IGNORE_MISMATCHED_SIZES', '1') == '1'
GRADIENT_CHECKPOINTING = os.environ.get('GRADIENT_CHECKPOINTING', '1') == '1'
RESHARD_AFTER_FORWARD = os.environ.get('RESHARD_AFTER_FORWARD', '1') == '1'
LORA_TARGET_MODULES = os.environ.get(
    'LORA_TARGET_MODULES',
    'wq_a,wq_b,wkv,wgate,gate_proj,up_proj,down_proj',
)
USE_EP = os.environ.get('USE_EP', '0') == '1'
ADAPTER_NAME = os.environ.get('ADAPTER_NAME', 'default')
EP_SIZE = BATCH_SIZE if USE_EP else 1
device_mesh = DeviceMesh.from_sizes(
    fsdp_size=BATCH_SIZE,
    dp_size=1,
    ep_size=EP_SIZE,
    device_type=Platform.get_platform().device_prefix(),
)

twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def create_dataset(data_slice=None):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID))
    dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle', 'ModelScope'))
    return dataset

def save_checkpoint(model: TransformersModel, checkpoint_name: str, dataloader: DataLoader):
    return model.save(
        name=checkpoint_name,
        output_dir=OUTPUT_DIR,
        adapter_name=ADAPTER_NAME,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )

def train():
    dataset = create_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(config, 'use_cache'):
        config.use_cache = False
    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        strategy="native_fsdp",
        memory_efficient_init=True,
        ignore_mismatched_sizes=IGNORE_MISMATCHED_SIZES,
        fsdp_config={
            'reshard_after_forward': RESHARD_AFTER_FORWARD,
            'expert_parallel': {
                'enabled': USE_EP,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            }
        },
    )

    apply_npu_patch(model)

    if USE_LORA:
        lora_target_modules = [name.strip() for name in LORA_TARGET_MODULES.split(',') if name.strip()]
        lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=lora_target_modules)
        model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=GRAD_ACCUM_STEPS)

    if not GRADIENT_CHECKPOINTING:
        model.model.gradient_checkpointing_disable()

    model.set_template(TEMPLATE_ID, model_id=MODEL_ID, adapter_name=ADAPTER_NAME)
    model.set_optimizer('AdamW', lr=LR, foreach=False, adapter_name=ADAPTER_NAME)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=1,
        num_training_steps=len(dataloader),
        adapter_name=ADAPTER_NAME,
    )
    optimizer_group = model.optimizer_group[ADAPTER_NAME]
    for batch in dataloader:
        if callable(batch):
            batch = batch()
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step(max_grad_norm=MAX_GRAD_NORM, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
        cur_step = optimizer_group.cur_step
        if cur_step > 0 and cur_step % LOG_INTERVAL == 0:
            metric = model.calculate_metric(is_training=True)
            if callable(metric):
                metric = metric()
            logger.info(f'Current is step {cur_step} of {len(dataloader)}, metric: {metric}')

    final_checkpoint = save_checkpoint(model, 'checkpoint-final', dataloader)
    logger.info(f'Saved final adapter to {final_checkpoint}')

if __name__ == '__main__':
    train()

