from pathlib import Path

from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.cli import CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
args = CLI.from_args()

device_mesh = DeviceMesh.from_sizes(dp_size=args.infra.dp_size, tp_size=args.infra.tp_size, pp_size=args.infra.pp_size, cp_size=args.infra.cp_size)
twinkle.initialize(mode=args.infra.mode, global_device_mesh=device_mesh)


def build_dataset(num_samples: int) -> Dataset:
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset.dataset_id, data_slice=range(num_samples)))
    dataset.set_template(args.template.template_cls, model_id=args.model.model_id)
    dataset.map(SelfCognitionProcessor(
        args.extra.get('model_name', 'twinkle大模型'),
        args.extra.get('model_author', 'ModelScope社区'),
    ))
    dataset.encode()
    return dataset


def save_checkpoint(model: MegatronModel, checkpoint_name: str, dataloader: DataLoader):
    model.save(
        checkpoint_name,
        output_dir=args.training.output_dir,
        adapter_name=args.lora.adapter_name,
        save_optimizer=args.checkpoint.save_optimizer,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def evaluate(model):
    eval_samples = args.training.eval_samples or 100
    dataloader = DataLoader(dataset=build_dataset(eval_samples), batch_size=args.training.batch_size)
    for batch in tqdm(dataloader):
        model.forward_only(inputs=batch)
    return model.calculate_metric(is_training=False)


def train():
    train_samples = args.training.train_samples or 1000
    dataset = build_dataset(train_samples)
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)

    model = MegatronModel(model_id=args.model.model_id)

    lora_config = LoraConfig(**args.get_lora_args())

    # Comment this to use full-parameter training
    model.add_adapter_to_model(args.lora.adapter_name, lora_config)
    model.set_optimizer(optimizer_cls='default', lr=args.optimizer.learning_rate)
    model.set_lr_scheduler(scheduler_cls='default', lr_warmup_steps=args.scheduler.num_warmup_steps,
                           lr_decay_steps=len(dataloader))

    start_step = 0
    if args.training.resume_from_checkpoint:
        checkpoint_path = Path(args.training.resume_from_checkpoint).expanduser().resolve()
        kwargs = {}
        if args.lora.adapter_name:
            kwargs['adapter_name'] = args.lora.adapter_name
        progress = model.resume_from_checkpoint(
            str(checkpoint_path), resume_only_model=args.training.resume_only_model, **kwargs)
        if not args.training.ignore_data_skip:
            dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
            start_step = progress['cur_step']

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')

    best_loss = float('inf')
    eval_interval = args.training.eval_interval or 20

    for step, batch in enumerate(dataloader, start=start_step):
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        if step % args.training.log_interval == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        if step > 0 and step % eval_interval == 0:
            metrics = evaluate(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = step
            current_loss = float(metrics['loss'])
            if current_loss < best_loss:
                save_checkpoint(model, f'checkpoint-{step}', dataloader)
                best_loss = current_loss
    save_checkpoint(model, 'last-checkpoint', dataloader)


if __name__ == '__main__':
    train()
