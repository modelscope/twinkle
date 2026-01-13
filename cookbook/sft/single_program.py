from peft import LoraConfig

from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel

logger = get_logger()


def eval(model: TransformersModel):
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math', data_slice=range(100)))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    for step, batch in enumerate(dataloader):
        model.forward_only(inputs=batch)
    metrics = model.calculate_metrics()
    return metrics

def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=16)
    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    loss_metric = 99.0
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch)
        if step % 16 == 0:
            logger.info(f'Current is step {step // 16}, loss: {output}')
        model.clip_grad_and_step()
        if step % 50 == 0:
            metrics = eval(model)
            logger.info(f'Eval finished with metrics: {metrics}')
            if loss_metric > metrics['loss']:
                model.save(f'checkpoint-{step}')
                loss_metric = metrics['loss']


if __name__ == '__main__':
    train()
