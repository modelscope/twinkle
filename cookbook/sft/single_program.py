from peft import LoraConfig
import twinkle
from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta, LazyDataset, PackingDataset, IterableDataset, IterablePackingDataset
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

twinkle.initialize(mode='local')

logger = get_logger()


def eval(model: TransformersModel):
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(5000)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    # dataset.pack_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    for step, batch in enumerate(dataloader):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics

def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(5000)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'), load_from_cache_file=False)
    dataset.encode(batched=True, load_from_cache_file=False)
    # dataset.pack_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=16)
    model.set_optimizer('AdamW', lr=1e-4)
    model.set_lr_scheduler('CosineWarmupScheduler', num_warmup_steps=30, num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    loss_metric = 99.0
    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch)
        if step % 16 == 0:
            logger.info(f'Current is step {step // 16}, metric: {model.calculate_metric(is_training=True)}')
        model.clip_grad_and_step()
        if step > 0 and (step / 16) % 100 == 0:
            metrics = eval(model)
            logger.info(f'Current is step {step // 16}, metrics: {metrics}')
            metrics['step'] = step
            if loss_metric > metrics['loss']:
                model.save(f'checkpoint-{step}')
                loss_metric = metrics['loss']


if __name__ == '__main__':
    train()
