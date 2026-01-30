import dotenv
dotenv.load_dotenv('.env')
import os
from peft import LoraConfig

from twinkle import get_device_placement, get_logger
from twinkle.dataset import DatasetMeta
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client import init_twinkle_client

logger = get_logger()

client = init_twinkle_client(base_url=os.environ['EAS_URL'], api_key=os.environ['EAS_TOKEN'])
# List all training runs
runs = client.list_training_runs()

resume_path = None
for run in runs:
    logger.info(run.model_dump_json(indent=2))
    # Get checkpoints for a run
    checkpoints = client.list_checkpoints(run.training_run_id)

    for checkpoint in checkpoints:
        logger.info(checkpoint.model_dump_json(indent=2))
        resume_path = checkpoint.twinkle_path
def train():
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition'))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)
    dataset.map('SelfCognitionProcessor', init_args={'model_name': 'twinkle模型', 'model_author': 'twinkle团队'})
    dataset.encode(batched=True)
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    model.set_template('Template')
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('AdamW', lr=1e-4)
    model.set_lr_scheduler('LinearLR')
    # Resume training if resume_path is provided
    if resume_path:
        model.load(resume_path, load_optimizer=True)
    logger.info(model.get_train_configs())
    for step, batch in enumerate(dataloader):
        output = model.forward_backward(inputs=batch)
        if step % 2 == 0:
            logger.info(f'Current is step {step // 2}, loss: {output}')
        model.clip_grad_norm(1.0)
        model.step()
        model.zero_grad()
        model.lr_step()
        if step > 0 and step % 8 == 0:
            logger.info(f'Saving checkpoint at step {step}')
            model.save(f'step-{step}', save_optimizer=True)


if __name__ == '__main__':
    train()
