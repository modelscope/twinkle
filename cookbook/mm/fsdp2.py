from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.cli import CLI
from twinkle.data_format import Trajectory, Message
from twinkle.dataloader import DataLoader
from twinkle.dataset import LazyDataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor

logger = get_logger()
args = CLI.from_args()

# Construct a device_mesh
device_mesh = DeviceMesh.from_sizes(fsdp_size=args.infra.fsdp_size, dp_size=args.infra.dp_size)
# use torchrun mode
twinkle.initialize(mode=args.infra.mode, global_device_mesh=device_mesh)


class LatexOCRProcessor(Preprocessor):

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(
            messages=[
                Message(role='user', content='<image>Using LaTeX to perform OCR on the image.', images=[row['image']]),
                Message(role='assistant', content=row['text']),
            ]
        )


def eval(model):
    # Eval samples
    eval_samples = args.training.eval_samples or 100
    dataset = LazyDataset(dataset_meta=DatasetMeta(args.dataset.dataset_id, data_slice=range(eval_samples)))
    dataset.set_template(args.template.template_cls, model_id=args.model.model_id)
    dataset.map(LatexOCRProcessor)
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics


def train():
    # Training samples
    train_samples = args.training.train_samples or 2000
    dataset = LazyDataset(dataset_meta=DatasetMeta(args.dataset.dataset_id, data_slice=range(train_samples)))
    # Set template to prepare encoding
    dataset.set_template(args.template.template_cls, model_id=args.model.model_id, max_length=args.template.max_length)
    # Preprocess the dataset to standard format
    dataset.map(LatexOCRProcessor)
    # Encode dataset
    dataset.encode()
    # Global batch size
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)
    # Use a TransformersModel
    model = TransformersModel(model_id=args.model.model_id, model_cls=args.model.model_cls)
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(**args.get_lora_args())

    # Add a lora to model
    model.add_adapter_to_model(args.lora.adapter_name, lora_config,
                               gradient_accumulation_steps=args.training.gradient_accumulation_steps)
    # Add Optimizer
    model.set_template(args.template.template_cls, model_id=args.model.model_id)
    model.set_optimizer(optimizer_cls=args.optimizer.optimizer_cls, lr=args.optimizer.learning_rate)
    # Add LRScheduler
    model.set_lr_scheduler(
        scheduler_cls=args.scheduler.scheduler_cls, num_warmup_steps=args.scheduler.num_warmup_steps,
        num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    eval_interval = args.training.eval_interval or 200
    for step, batch in enumerate(dataloader):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        if step % args.training.log_interval == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        if step > 0 and step % eval_interval == 0:
            metrics = eval(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = step
            if loss_metric > float(metrics['loss']):
                model.save(f'checkpoint-{step}')
                loss_metric = float(metrics['loss'])
    model.save('last-checkpoint')


if __name__ == '__main__':
    train()
