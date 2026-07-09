from pathlib import Path

from peft import LoraConfig
from tqdm import tqdm
from torch.optim import Muon  # PyTorch 2.9+; matrix-orthogonalized momentum optimizer.

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.cli import CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.utils.framework import Torch
from twinkle.kernel import kernelize_model

logger = get_logger()
args = CLI.from_args()

device_mesh = DeviceMesh.from_sizes(fsdp_size=args.infra.fsdp_size, dp_size=args.infra.dp_size)
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


def save_checkpoint(model: TransformersModel, checkpoint_name: str, dataloader: DataLoader):
    model.save(
        checkpoint_name,
        output_dir=args.training.output_dir,
        adapter_name=args.lora.adapter_name,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def evaluate(model):
    eval_samples = args.training.eval_samples or 100
    dataloader = DataLoader(dataset=build_dataset(eval_samples), batch_size=args.training.batch_size)
    for batch in tqdm(dataloader):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    return model.calculate_metric(is_training=False)


def train():
    train_samples = args.training.train_samples or 1000
    dataset = build_dataset(train_samples)
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)
    model = TransformersModel(model_id=args.model.model_id)
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}
    # npu patch
    if Torch.is_npu_available():
        model = kernelize_model(model, mode='train', device='npu')

    lora_config = LoraConfig(**args.get_lora_args())
    model.add_adapter_to_model(
        args.lora.adapter_name, lora_config,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps)
    # Muon optimizes 2D hidden-layer weight matrices via Newton-Schulz orthogonalization.
    # In LoRA training the trainable params are exclusively lora_A / lora_B (both 2D),
    # so Muon applies cleanly without an AdamW fallback for 1D params.
    # ``adjust_lr_fn='match_rms_adamw'`` rescales the orthogonalized update so the same
    # lr / weight_decay tuned for AdamW can be reused directly (Moonshot Muon recipe).
    model.set_optimizer(
        optimizer_cls=Muon,
        lr=args.optimizer.learning_rate,
        adjust_lr_fn='match_rms_adamw',
    )

    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls=args.scheduler.scheduler_cls,
        num_warmup_steps=args.scheduler.num_warmup_steps,
        num_training_steps=len(dataloader))

    if args.training.resume_from_checkpoint:
        checkpoint_path = Path(args.training.resume_from_checkpoint).expanduser().resolve()
        progress = model.resume_from_checkpoint(
            str(checkpoint_path),
            resume_only_model=args.training.resume_only_model,
            adapter_name=args.lora.adapter_name)
        if not args.training.ignore_data_skip:
            dataloader.resume_from_checkpoint(progress['consumed_train_samples'])

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    optimizer_group = model.optimizer_group[args.lora.adapter_name]
    best_loss = float('inf')
    eval_interval = args.training.eval_interval or 40
    for batch in dataloader:
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        cur_step = optimizer_group.cur_step
        if cur_step % args.training.log_interval == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {cur_step} of {len(dataloader)}, metric: {metric}')
        if cur_step > 0 and cur_step % eval_interval == 0:
            metrics = evaluate(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = cur_step
            current_loss = float(metrics['loss'])
            if current_loss < best_loss:
                save_checkpoint(model, f'checkpoint-{cur_step}', dataloader)
                best_loss = current_loss
    save_checkpoint(model, 'last-checkpoint', dataloader)


if __name__ == '__main__':
    train()
