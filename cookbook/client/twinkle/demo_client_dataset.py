"""Twinkle C/S training demo with dataset processing on the client.

The client loads, tokenizes, and batches the dataset. Only model operations are
sent to the Twinkle server. The input JSON/JSONL is expected to contain a
``messages`` column.
"""

from peft import LoraConfig

import twinkle
from twinkle import get_logger, init_twinkle_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle_client.model import MultiLoraTransformersModel

logger = get_logger()

base_url = '{{ base_url }}'
api_key = '{{ api_key }}'
served_model_name = '{{ base_model }}'
model_path = '{{ model_local_path }}'
dataset_id = '{{ dataset_path }}'
epochs = {{epochs}}
max_length = {{max_length}}
truncation_strategy = '{{ truncation_strategy }}'
batch_size = 32
lr = {{lr}}
grad_accumulation_steps = {{grad_accumulation_steps}}
train_id = '{{ train_id }}'

template = 'DeepseekV4Template'
lora_target_modules = 'all-linear'
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.0

train_config = {
    'base_model': served_model_name,
    'epochs': epochs,
    'max_length': max_length,
    'batch_size': batch_size,
    'lr': lr,
    'grad_accumulation_steps': grad_accumulation_steps,
    'lora_target_modules': lora_target_modules,
    'lora_rank': lora_rank,
    'lora_alpha': lora_alpha,
    'lora_dropout': lora_dropout,
}
logger.info(f'train config: {train_config}')


def _build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules='all-linear',
        exclude_modules=['o_a_proj'],
        target_parameters=[
            'mlp.experts.gate_up_proj',
            'mlp.experts.down_proj',
        ],
    )


def build_local_dataloader() -> DataLoader:
    import os
    if not os.path.exists(dataset_id):
        raise FileNotFoundError(f'Client cannot access dataset: {dataset_id}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Client cannot access tokenizer/model directory: {model_path}')

    dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=dataset_id))
    dataset.set_template(
        template,
        model_id=model_path,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
    )

    # The input is already in {"messages": [...]} format. For raw
    # query/response SelfCognition data, call dataset.map(...) before encode().
    dataset.encode(num_proc=8, load_from_cache_file=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=2,
    )


def train() -> None:
    # This ensures twinkle.dataset and twinkle.dataloader execute in this process.
    twinkle.initialize(mode='local')

    client = init_twinkle_client(base_url=base_url, api_key=api_key)
    supported_models = [item.model_name for item in client.get_server_capabilities().supported_models]
    if served_model_name not in supported_models:
        raise RuntimeError(f'{served_model_name!r} is not served; available models: {supported_models}')

    dataloader = build_local_dataloader()
    model = MultiLoraTransformersModel(model_id=served_model_name)
    model.add_adapter_to_model(
        'default',
        _build_lora_config(),
        gradient_accumulation_steps=grad_accumulation_steps,
        save_dir=None,
    )
    # The model server automatically supplies its tokenizer_id here.
    model.set_template(template)
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('Adam', lr=lr)

    for epoch in range(epochs):
        logger.info('Starting epoch %s', epoch)
        for step, batch in enumerate(dataloader, start=1):
            model.forward_backward(inputs=batch)
            model.clip_grad_and_step()
            if step % grad_accumulation_steps == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info('step=%s/%s metric=%s', step, len(dataloader), metric.result)

    checkpoint = model.save(name='twinkle-final', save_optimizer=True)
    logger.info('Saved checkpoint: %s', checkpoint.twinkle_path)
    client.close()


if __name__ == '__main__':
    train()
