from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import Trajectory, Message
from twinkle.dataloader import DataLoader
from twinkle.dataset import LazyDataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor, Preprocessor

# Construct a device_mesh, fsdp=4, dp=2
device_mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=2)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


class LatexOCRProcessor(Preprocessor):

    def __call__(self, row) -> Trajectory:
        return Trajectory(
            messages=[
                Message(role='user', content='<image>Using LaTeX to perform OCR on the image.', images=[row['image']]),
                Message(role='assistant', content=row['text']),
            ]
        )


def train():
    # 2000 samples
    dataset = LazyDataset(dataset_meta=DatasetMeta('ms://AI-ModelScope/LaTeX_OCR', data_slice=range(2000)))
    # Set template to prepare encoding
    dataset.set_template('Template', model_id='ms://Qwen/Qwen3.5-4B')
    # Preprocess the dataset to standard format
    dataset.map(LatexOCRProcessor)
    # Encode dataset
    dataset.encode()
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    # Use a TransformersModel
    model = TransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Add a lora to model, with name `default`
    # Comment this to use full-parameter training
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    # lora: 18G * 4
    # full: 50G * 4
    for step, batch in enumerate(dataloader):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        if step % 20 == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        if step > 0 and step % 40 == 0:
            metrics = eval(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = step
            if loss_metric > float(metrics['loss']):
                model.save(f'checkpoint-{step}')
                loss_metric = float(metrics['loss'])
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()
