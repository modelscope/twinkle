import os

from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, Platform
from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.preprocessor import SelfCognitionProcessor

if Platform.get_rank() == 0 and os.environ.get('SWANLAB_API_KEY'):
    # rank0 recording
    import swanlab
    swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=True)

    run = swanlab.init(
        project="twinkle",
    )


# Construct a device_mesh, tp=pp=cp=2, dp=1
device_mesh = DeviceMesh.from_sizes(dp_size=1, tp_size=2, pp_size=2, cp_size=2)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


def eval(model):
    # 100 Samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(100)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
    metrics = model.calculate_metric(is_training=False)
    return metrics


def train():
    # 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    # Set template to prepare encoding
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 1, dp_size = 1
    dataloader = DataLoader(dataset=dataset, batch_size=16)
    # Use a MegatronModel
    model = MegatronModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules='all-linear'
    )

    # Add a lora to model, with name `default`
    # Comment this to use full-parameter training
    model.add_adapter_to_model('default', lora_config)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='default', lr=1e-4)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(scheduler_cls='default', lr_warmup_steps=5, lr_decay_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    # lora: 10G * 8
    # full: 40G * 8
    for step, batch in enumerate(dataloader):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        if step % 5 == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            if Platform.get_rank() == 0 and os.environ.get('SWANLAB_API_KEY'):
                swanlab.log(metric)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        if step > 0 and step % 20 == 0:
           metrics = eval(model)
           logger.info(f'Eval metric: {metrics}')
           metrics['step'] = step
           if loss_metric > float(metrics['loss']):
               model.save(f'checkpoint-{step}')
               loss_metric = float(metrics['loss'])
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()