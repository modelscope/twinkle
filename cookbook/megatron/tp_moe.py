from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.cli import CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MegatronModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
args = CLI.from_args()

# Construct a device_mesh, tp=pp=ep=dp=2
device_mesh = DeviceMesh.from_sizes(
    dp_size=args.infra.dp_size, tp_size=args.infra.tp_size,
    pp_size=args.infra.pp_size, ep_size=args.infra.ep_size,
    sequence_parallel=args.infra.sequence_parallel,
)
# use torchrun mode
twinkle.initialize(mode=args.infra.mode, global_device_mesh=device_mesh)


def eval(model):
    # Eval samples
    eval_samples = args.training.eval_samples or 100
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset.dataset_id, data_slice=range(eval_samples)))
    dataset.set_template(args.template.template_cls, model_id=args.model.model_id)
    dataset.map(SelfCognitionProcessor(
        args.extra.get('model_name', 'twinkle大模型'),
        args.extra.get('model_author', 'ModelScope社区'),
    ))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
    metrics = model.calculate_metric(is_training=False)
    return metrics


def train():
    # Training samples
    train_samples = args.training.train_samples or 1000
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset.dataset_id, data_slice=range(train_samples)))
    # Set template to prepare encoding
    dataset.set_template(args.template.template_cls, model_id=args.model.model_id)
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor(
        args.extra.get('model_name', 'twinkle大模型'),
        args.extra.get('model_author', 'ModelScope社区'),
    ))
    # Encode dataset
    dataset.encode()
    # Global batch size
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)
    # Use a MegatronModel
    model = MegatronModel(model_id=args.model.model_id)

    lora_config = LoraConfig(**args.get_lora_args())

    # Add a lora to model, with name from args
    # Comment this to use full-parameter training
    model.add_adapter_to_model(args.lora.adapter_name, lora_config)
    # Add Optimizer
    model.set_optimizer(optimizer_cls='default', lr=args.optimizer.learning_rate)
    # Add LRScheduler
    model.set_lr_scheduler(scheduler_cls='default', lr_warmup_steps=args.scheduler.num_warmup_steps,
                           lr_decay_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    eval_interval = args.training.eval_interval or 20
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
    model.save('last-checkpoint', merge_lora=True)


if __name__ == '__main__':
    train()
