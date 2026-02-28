from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.preprocessor import SelfCognitionProcessor

device_mesh = DeviceMesh.from_sizes(dp_size=4, tp_size=1, pp_size=1, ep_size=4)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()

MODEL_ID = 'Qwen/Qwen3.5-35B-A3B'

def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    dataset.set_template('Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    model = MegatronModel(model_id=MODEL_ID)
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config)
    model.set_optimizer(optimizer_cls='default', lr=1e-4)
    model.set_lr_scheduler(scheduler_cls='default', lr_warmup_steps=2, lr_decay_steps=len(dataloader))
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')

    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        if step % 5 == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Step {step}/{len(dataloader)}, metric: {metric}')

    # NOTE: you should merge lora for Qwen3.5 model when using Megatron
    model.save('last-checkpoint', merge_lora=True)
    logger.info('Training completed.')


if __name__ == '__main__':
    train()
