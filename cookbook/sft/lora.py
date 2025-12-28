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
    dataset = Dataset(DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', model_id='Qwen/Qwen2.5-7B-Instruct')
    dataset.map('CompetitionMathProcessor')
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset)
    model = TransformersModel(pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )

    model.add_adapter_to_model('default', lora_config)
    model.set_processor(InputProcessor)
    model.set_loss(CrossEntropyLoss)
    model.set_optimizer(AdamW, lr=1e-4)
    model.set_lr_scheduler(LinearLR)
    print(get_device_placement())
    for step, batch in enumerate(dataloader):
        for gas in range(16):
            model.forward_backward(batch, adapter_name='default')
        model.step()
        model.zero_grad()
        model.lr_step()
        if step % 50 == 0:
            model.save('./output')


