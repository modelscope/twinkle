from functools import partial
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, get_logger
from twinkle.cli import CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.utils.framework import Torch
from twinkle.kernel import kernelize_model

logger = get_logger()
args = CLI.from_args()

# FSDP + sequence-parallel validation over 4 GPUs: dp=2, fsdp=2.
# In Transformers route, ulysses_size is the total sequence-parallel degree.
device_mesh = DeviceMesh.from_sizes(
    dp_size=args.infra.dp_size,
    fsdp_size=args.infra.fsdp_size,
    ulysses_size=args.infra.ulysses_size,
)

twinkle.initialize(
    mode=args.infra.mode,
    global_device_mesh=device_mesh,
    lazy_collect=args.infra.lazy_collect,
)


def eval(model):
    eval_samples = args.training.eval_samples or 100
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=range(eval_samples)),
        batch_size=args.training.batch_size,
        device_mesh=device_mesh,
    )
    for _, batch in enumerate(dataloader):
        model.forward_only(inputs=batch, adapter_name=args.lora.adapter_name)
        model.calculate_loss(adapter_name=args.lora.adapter_name)
    return model.calculate_metric(is_training=False, adapter_name=args.lora.adapter_name)


def create_dataset(data_slice=None):
    train_samples = args.training.train_samples or 500
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset.dataset_id, data_slice=data_slice or range(train_samples)))
    dataset.set_template(args.template.template_cls, model_id=args.model.model_id)
    dataset.map(SelfCognitionProcessor(
        args.extra.get('model_name', 'twinkle模型'),
        args.extra.get('model_author', 'twinkle团队'),
    ))
    dataset.encode(batched=True)
    return dataset


def train():
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=None),
        batch_size=args.training.batch_size,
        device_mesh=device_mesh,
    )

    model = TransformersModel(
        model_id=args.model.model_id,
        device_mesh=device_mesh,
        strategy=args.model.strategy,
    )
    # npu patch
    if Torch.is_npu_available():
        model = kernelize_model(model, mode='train', device='npu')
    lora_config = LoraConfig(**args.get_lora_args())
    model.add_adapter_to_model(args.lora.adapter_name, lora_config,
                               gradient_accumulation_steps=args.training.gradient_accumulation_steps)
    model.set_optimizer(args.optimizer.optimizer_cls, lr=args.optimizer.learning_rate,
                        adapter_name=args.lora.adapter_name)
    model.set_lr_scheduler(
        scheduler_cls=args.scheduler.scheduler_cls,
        num_warmup_steps=args.scheduler.num_warmup_steps,
        num_training_steps=len(dataloader),
        adapter_name=args.lora.adapter_name,
    )

    logger.info(model.get_train_configs(adapter_name=args.lora.adapter_name))
    logger.info(f'Total steps: {len(dataloader)}')

    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch, adapter_name=args.lora.adapter_name)
        model.clip_grad_and_step(adapter_name=args.lora.adapter_name)
        if step % args.training.log_interval == 0:
            metric = model.calculate_metric(is_training=True, adapter_name=args.lora.adapter_name)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
    model.save('last-checkpoint', interval=1)


if __name__ == '__main__':
    train()
