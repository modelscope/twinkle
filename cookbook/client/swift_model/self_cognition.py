# Twinkle Client - LoRA fine-tuning of a swift-served model twinkle does not
# natively support (google/gemma-4-12B-it).
#
# This is the client half of the swift_model server-client pair. It is identical
# in shape to cookbook/client/twinkle/self_cognition.py; the only differences are
# the model id and the template class. The server builds gemma-4-12B-it through
# swift's family loader (see server_config.yaml's `model_loader`), and the client
# just addresses it by its public name.
#
# The server must be running first (see run_server.sh / server.py).

import dotenv

dotenv.load_dotenv('.env')

import os
from peft import LoraConfig

from twinkle import get_logger
from twinkle import init_twinkle_client
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle_client.model import MultiLoraTransformersModel

logger = get_logger()

# Public model name served by this pair; must match server_config.yaml's
# supported_models and the model app's route.
base_model = os.environ.get('TWINKLE_MODEL_ID', 'google/gemma-4-12B-it')
base_url = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000')
api_key = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')
save_dir = '/tmp/twinkle_gemma4_sft_output'

# gemma has no dedicated twinkle template class yet, so use the base `Template`,
# which renders through the checkpoint's own jinja chat_template. (twinkle's
# named templates currently cover only Qwen3.5 and DeepSeek-V4.)
template_cls = 'Template'

# Step 1: connect to the running server.
client = init_twinkle_client(base_url=base_url, api_key=api_key)

print('Available models:')
for item in client.get_server_capabilities().supported_models:
    print('- ' + item.model_name)


def train():
    # Step 2: prepare the self-cognition dataset.
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(500)))
    dataset.set_template(template_cls, model_id=f'ms://{base_model}', max_length=512)
    dataset.map('SelfCognitionProcessor', init_args={'model_name': 'twinkle模型', 'model_author': 'ModelScope社区'})
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    # Step 3: address the swift-served model and attach a LoRA adapter.
    model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')
    lora_config = LoraConfig(target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2, save_dir=save_dir)

    model.set_template(template_cls)
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('Adam', lr=1e-4)

    # Step 4: run a short training loop.
    max_steps = 10
    logger.info(model.get_train_configs().model_dump())

    global_step = 0
    for epoch in range(3):
        logger.info(f'Starting epoch {epoch}')
        for cur_step, batch in enumerate(dataloader, start=1):
            model.forward_backward(inputs=batch)
            model.clip_grad_and_step()
            global_step += 1

            if cur_step % 2 == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(f'Step {cur_step} of {len(dataloader)}, metric: {metric.result}')

            if global_step >= max_steps:
                logger.info(f'Reached max_steps={max_steps}, stopping training.')
                break

        if global_step >= max_steps:
            break

        twinkle_path = model.save(
            name=f'twinkle-gemma4-epoch-{epoch}',
            save_optimizer=True,
            consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
        )
        logger.info(f'Saved checkpoint: {twinkle_path}')


if __name__ == '__main__':
    train()
