from torch.optim.lr_scheduler import LinearLR

from twinkle import get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CrossEntropyLoss
from twinkle.model import TransformersModel
from peft import LoraConfig
from torch.optim import AdamW

from twinkle.processor import InputProcessor

logger = get_logger()


def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math', data_slice=range(0, 19)))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset, batch_size=2)
    model = TransformersModel(pretrained_model_name_or_path='ms://Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=16)
    model.set_template('Qwen3Template', adapter_name='default')
    model.set_processor(InputProcessor, adapter_name='default')
    model.set_loss(CrossEntropyLoss, adapter_name='default')
    model.set_optimizer(AdamW, lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(LinearLR, adapter_name='default')
    logger.info(get_device_placement())
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch, adapter_name='default')
        if step % 16 == 0:
            logger.info(f'Current is step {step // 16}, loss: {output["loss"]}')
        model.step(adapter_name='default')
        model.zero_grad(adapter_name='default')
        model.lr_step(adapter_name='default')
        if step % 50 == 0:
            model.save('./output', adapter_name='default')


if __name__ == '__main__':
    train()
