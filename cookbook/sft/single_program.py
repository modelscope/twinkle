import os
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
from peft import LoraConfig
import twinkle
import time
from tqdm import tqdm
from twinkle import DeviceMesh, Platform
from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta, LazyDataset, PackingDataset, IterableDataset, IterablePackingDataset
from twinkle.model import MultiLoraMegatronModel, MegatronModel, TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
import torch
torch._dynamo.disable()
if Platform.get_rank() == 0:
    import swanlab
    swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=True)

    run = swanlab.init(
        project="megatron-swift",
    )


device_mesh = DeviceMesh.from_sizes()
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


def eval(model):
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500), max_length=256))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=256)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'), load_from_cache_file=False)
    dataset.encode(batched=True, load_from_cache_file=False)
    # dataset.pack_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics

def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=256)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    # dataset.pack_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=0)

    model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader), adapter_name='default')
    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name='default'))
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch, adapter_name='default')
        # outputs = model.forward_only(inputs=batch, adapter_name='default')
        model.clip_grad_and_step(adapter_name='default')
        if step % 5 == 0:
            metric = model.calculate_metric(is_training=True, adapter_name='default')
            _metrics = {}
            for key, value in metric.items():
                try:
                    value = float(value)
                    _metrics[key] = value
                except:
                    pass
            if Platform.get_rank() == 0:
                swanlab.log(_metrics)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        #if step > 0 and (step / 4) % 30 == 0:
        #    metrics = eval(model)
        #    logger.info(f'Eval metric: {metrics}')
        #    metrics['step'] = step
        #    if loss_metric > float(metrics['loss']):
        #        model.save(f'checkpoint-{step}')
        #        loss_metric = float(metrics['loss'])
    model.save(f'last-checkpoint', adapter_name='default')
    time.sleep(3)
    model.load(f'last-checkpoint', adapter_name='default')
    # model.remove_adapter(adapter_name='default')


if __name__ == '__main__':
    train()