from torch.optim.lr_scheduler import LinearLR

from twinkle import get_device_placement
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CrossEntropyLoss
from twinkle.model import TransformersModel
from peft import LoraConfig
from torch.optim import AdamW

from twinkle.processor import InputProcessor


def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset)
    model = TransformersModel(pretrained_model_name_or_path='ms://Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config)
    model.set_template('Qwen3Template', adapter_name='default')
    model.set_processor(InputProcessor, adapter_name='default')
    model.set_loss(CrossEntropyLoss, adapter_name='default')
    model.set_optimizer(AdamW, lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(LinearLR, adapter_name='default')
    print(get_device_placement())
    for step, batch in enumerate(dataloader):
        for gas in range(16):
            model.forward_backward(inputs=batch, adapter_name='default')
        model.step()
        model.zero_grad()
        model.lr_step()
        if step % 50 == 0:
            model.save('./output')


if __name__ == '__main__':
    train()
