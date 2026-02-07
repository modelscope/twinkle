import os
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
from peft import LoraConfig
import twinkle
from tqdm import tqdm
from twinkle import DeviceMesh, Platform, DeviceGroup
from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta, LazyDataset, PackingDataset, IterableDataset, IterablePackingDataset
from twinkle.model import MultiLoraMegatronModel, MegatronModel
from twinkle.preprocessor import SelfCognitionProcessor


device_group = [
        DeviceGroup(
            name='default',
            ranks=16,
            device_type='cuda',
        )
    ]
    
device_mesh = DeviceMesh.from_sizes(cp_size=2, pp_size=2, tp_size=2, dp_size=2, sequence_parallel=True)
twinkle.initialize(mode='ray', nproc_per_node=16, groups=device_group, global_device_mesh=device_mesh)

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
    dataset = PackingDataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(5000)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=256)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    dataset.pack_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=0)

    model = MultiLoraMegatronModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', mixed_precision='bf16', recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1, remote_group='default')

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer(optimizer_cls='default', lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(scheduler_cls='default', lr_warmup_steps=5, lr_decay_steps=len(dataloader), adapter_name='default')
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
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        #if step > 0 and (step / 4) % 30 == 0:
        #    metrics = eval(model)
        #    logger.info(f'Eval metric: {metrics}')
        #    metrics['step'] = step
        #    if loss_metric > float(metrics['loss']):
        #        model.save(f'checkpoint-{step}')
        #        loss_metric = float(metrics['loss'])
    model.save(f'last-checkpoint', adapter_name='default')
    model.load(f'last-checkpoint', adapter_name='default')
    # model.remove_adapter(adapter_name='default')


if __name__ == '__main__':
    train()