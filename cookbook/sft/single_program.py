from peft import LoraConfig
import twinkle
from tqdm import tqdm
from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta, LazyDataset, PackingDataset, IterableDataset, IterablePackingDataset
from twinkle.model import MultiLoraTransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

twinkle.initialize(mode='local')

logger = get_logger()


def eval(model: MultiLoraTransformersModel):
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    # dataset.pack_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics

def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(100)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    # dataset.pack_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=8, num_workers=4)

    model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=4)
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')
    model.set_lr_scheduler('CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader), adapter_name='default')
    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name='default'))
    logger.info(f'Total steps: {len(dataloader)//4}')
    loss_metric = 99.0
    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch, adapter_name='default')
        model.clip_grad_and_step(adapter_name='default')
        if step % 20 == 0:
            logger.info(f'Current is step {step // 4} of {len(dataloader)//4}, metric: {model.calculate_metric(is_training=True, adapter_name='default')}')
        #if step > 0 and (step / 4) % 30 == 0:
        #    metrics = eval(model)
        #    logger.info(f'Eval metric: {metrics}')
        #    metrics['step'] = step
        #    if loss_metric > float(metrics['loss']):
        #        model.save(f'checkpoint-{step}')
        #        loss_metric = float(metrics['loss'])
    model.save(f'last-checkpoint', adapter_name='default', interval=1)


if __name__ == '__main__':
    train()
