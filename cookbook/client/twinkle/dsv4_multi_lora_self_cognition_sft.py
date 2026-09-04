# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4 self-cognition SFT with client-side dataset processing."""

import os

import twinkle
from peft import LoraConfig
from twinkle import get_logger, init_twinkle_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle_client.model import MultiLoraTransformersModel


logger = get_logger()

base_url = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000')
api_key = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')
served_model_name = os.environ.get('TWINKLE_MODEL_ID', 'DeepSeek-V4-Flash-0731')
model_path = os.environ.get('MODEL_LOCAL_PATH') or os.environ.get('DSV4_MODEL_ID', '')
dataset_id = os.environ.get('DATASET_PATH') or os.environ.get('DATASET_ID', '')
epochs = int(os.environ.get('EPOCHS') or os.environ.get('NUM_EPOCHS', '3'))
max_length = int(os.environ.get('MAX_LENGTH', '8192'))
truncation_strategy = os.environ.get('TRUNCATION_STRATEGY', 'delete')
batch_size = int(os.environ.get('BATCH_SIZE', '32'))
lr = float(os.environ.get('LR', '1e-4'))
grad_accumulation_steps = int(
    os.environ.get('GRAD_ACCUMULATION_STEPS') or os.environ.get('GRAD_ACCUM_STEPS', '1')
)
train_id = os.environ.get('TRAIN_ID') or os.environ.get('ADAPTER_NAME', 'tenant_a')

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
    if not dataset_id:
        raise ValueError('Set DATASET_PATH or DATASET_ID to the client-local dataset path.')
    if not model_path:
        raise ValueError(
            'Set MODEL_LOCAL_PATH or DSV4_MODEL_ID to the client-local tokenizer/model directory.'
        )
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
    dataset.map(
        'SelfCognitionProcessor',
        init_args={
            'model_name': 'twinkle模型',
            'model_author': 'twinkle团队',
        },
    )
    dataset.encode(num_proc=8, load_from_cache_file=True)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=2,
    )


def train() -> None:
    # Dataset loading, mapping, tokenization, and batching all run locally.
    twinkle.initialize(mode='local')

    client = init_twinkle_client(base_url=base_url, api_key=api_key)
    try:
        supported_models = [item.model_name for item in client.get_server_capabilities().supported_models]
        if served_model_name not in supported_models:
            raise RuntimeError(
                f'{served_model_name!r} is not served; available models: {supported_models}'
            )

        dataloader = build_local_dataloader()
        model = MultiLoraTransformersModel(model_id=served_model_name)
        model.add_adapter_to_model(
            train_id,
            _build_lora_config(),
            gradient_accumulation_steps=grad_accumulation_steps,
            save_dir=None,
        )
        model.set_template(template)
        model.set_processor('InputProcessor', padding_side='right')
        model.set_loss('CrossEntropyLoss')
        model.set_optimizer('Adam', lr=lr)

        for epoch in range(epochs):
            logger.info('Starting epoch %s/%s', epoch + 1, epochs)
            for step, batch in enumerate(dataloader, start=1):
                model.forward_backward(
                    inputs=batch,
                    gradient_accumulation_steps=grad_accumulation_steps,
                )
                model.clip_grad_and_step(
                    gradient_accumulation_steps=grad_accumulation_steps,
                )
                if step % grad_accumulation_steps == 0:
                    metric = model.calculate_metric(is_training=True)
                    logger.info('step=%s/%s metric=%s', step, len(dataloader), metric.result)

        checkpoint = model.save(name=f'{train_id}-final', save_optimizer=True)
        logger.info('Saved checkpoint: %s', checkpoint.twinkle_path)
    finally:
        client.close()


if __name__ == '__main__':
    train()
