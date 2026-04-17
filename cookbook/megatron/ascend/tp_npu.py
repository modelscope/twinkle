import twinkle
from peft import LoraConfig

from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.preprocessor import SelfCognitionProcessor

MODEL_ID = 'ms://Qwen/Qwen3-4B'
DATASET_ID = 'ms://swift/self-cognition'
DATASET_SLICE = range(256)
BATCH_SIZE = 8
MAX_STEPS = 10

# Keep the same 8-card TP/PP/DP layout as the GPU reference script, but run it
# through the NPU backend to validate Megatron + MindSpeed integration.
device_mesh = DeviceMesh.from_sizes(dp_size=2, tp_size=2, pp_size=2, device_type='npu')
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


def build_dataset():
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=DATASET_SLICE))
    # Qwen3-4B is a text-only model, so use the base template instead of the VL template.
    dataset.set_template('Template', model_id=MODEL_ID, max_length=512)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    return dataset


def build_model(total_steps: int):
    model = MegatronModel(model_id=MODEL_ID)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config)
    model.set_optimizer(optimizer_cls='default', lr=1e-4)
    model.set_lr_scheduler(scheduler_cls='default', lr_warmup_steps=2, lr_decay_steps=total_steps)
    return model


def train():
    dataset = build_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0)
    model = build_model(len(dataloader))

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}, validating {MAX_STEPS} steps')

    for step, batch in enumerate(dataloader):
        if step >= MAX_STEPS:
            break
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        metric = model.calculate_metric(is_training=True)
        logger.info(f'[NPU smoke] step {step + 1}/{MAX_STEPS}, metric: {metric}')


if __name__ == '__main__':
    train()
