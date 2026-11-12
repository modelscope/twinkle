"""Train a Hybrid LoRA adapter on the self-cognition dataset.

Start the Hybrid Transformers server before running this script.
"""

import os

import dotenv
from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle_client.model import MultiLoraTransformersModel

dotenv.load_dotenv('.env')

logger = get_logger()

BASE_MODEL = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-9B')
BASE_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000')
API_KEY = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')
SAVE_DIR = os.environ.get('TWINKLE_SAVE_DIR', '/tmp/twinkle_hybrid_sft_output')
MAX_STEPS = int(os.environ.get('TWINKLE_MAX_STEPS', '10'))


def train():
    init_twinkle_client(base_url=BASE_URL, api_key=API_KEY)

    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template('Qwen3_5Template', model_id=f'ms://{BASE_MODEL}', max_length=512)
    dataset.map('SelfCognitionProcessor', init_args={
        'model_name': 'twinkle模型',
        'model_author': 'ModelScope社区',
    })
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    model = MultiLoraTransformersModel(model_id=f'ms://{BASE_MODEL}')
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules='all-linear',
    )
    model.add_adapter_to_model(
        'default',
        lora_config,
        adapter_mode='hybrid',
        save_dir=SAVE_DIR,
    )
    model.set_template('Qwen3_5Template')
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('Adam', lr_lora=2.5e-5, lr_fft=1e-6)

    logger.info(model.get_train_configs().model_dump())
    global_step = 0
    for epoch in range(3):
        logger.info(f'Starting epoch {epoch}')
        for batch in dataloader:
            model.forward_backward(inputs=batch)
            model.clip_grad_and_step()
            global_step += 1

            if global_step % 2 == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(f'Current step: {global_step}, metric: {metric.result}')

            if global_step >= MAX_STEPS:
                break
        if global_step >= MAX_STEPS:
            break

    checkpoint = model.save(
        name=f'hybrid-checkpoint-step-{global_step}',
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )
    logger.info(f'Saved Hybrid checkpoint: {checkpoint}')


if __name__ == '__main__':
    train()
