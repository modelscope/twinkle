import os
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
import torch
torch._dynamo.disable()
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, Platform
from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.preprocessor import SelfCognitionProcessor

if Platform.get_rank() == 0:
    import swanlab
    swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=True)

    run = swanlab.init(
        project="megatron-swift",
    )

device_mesh = DeviceMesh.from_sizes(vpp_size=2, cp_size=2, pp_size=2, tp_size=2, dp_size=2)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


def eval(model):
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    # dataset.pack_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics

def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    # dataset.pack_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=0)

    model = MegatronModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', mixed_precision='bf16', recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1)

    model.set_optimizer(optimizer_cls='default', lr=1e-5)
    model.set_lr_scheduler(scheduler_cls='default', lr_warmup_steps=10, lr_decay_steps=len(dataloader))
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch, micro_batch_size=1)
        # outputs = model.forward_only(inputs=batch, adapter_name='default')
        model.clip_grad_and_step()
        if step % 5 == 0:
            metric = model.calculate_metric(is_training=True)
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
    model.save(f'last-checkpoint')
    # model.load(f'last-checkpoint', adapter_name='default')


if __name__ == '__main__':
    train()
