from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoConfig
from transformers import (
    Gemma4UnifiedConfig,
)

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.cli import CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel

logger = get_logger()
args = CLI.from_args()

########## Construct a device_mesh ##########
device_mesh = DeviceMesh.from_sizes(
    fsdp_size=args.infra.fsdp_size,
    dp_size=args.infra.dp_size,
    ep_size=args.infra.ep_size,
    device_type=Platform.get_platform().device_prefix(),
)
# use torchrun mode
twinkle.initialize(mode=args.infra.mode, global_device_mesh=device_mesh)

########## hyperparameters ##########
IGNORE_MISMATCHED_SIZES = args.extra.get('ignore_mismatched_sizes', True)

### reduce model layers for debug
TEXT_NUM_LAYERS = args.extra.get('text_num_layers', None)

from twinkle.preprocessor import Preprocessor
from twinkle.data_format import Message, Trajectory
class LatexOCRProcessor(Preprocessor):

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        col = self.map_row_to_col(rows)
        return col

    def preprocess(self, row) -> Trajectory:
        return Trajectory(
            messages=[
                Message(role='user', content='<image>Using LaTeX to perform OCR on the image.', images=[row['image']]),
                Message(role='assistant', content=row['text']),
            ]
        )

def evaluate(model, eval_dataloader):
    for step, batch in tqdm(enumerate(eval_dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics

def train():

    # Explicitly define full feature schema to prevent Image objects nested inside messages
    # from being incorrectly serialized into JSON during Arrow persistence via Dataset.map writer
    from datasets import Features, Value, Image, List
    sub_msg_feat = Features({
        'role': Value('string'),
        'content': Value('string'),
        'images': List(Image(decode=True))
    })
    writer_features = Features({
        'image': Image(decode=True),
        'text': Value('string'),
        'messages': List(sub_msg_feat)
    })
    ### prepare dataset and dataloader
    train_samples = args.training.train_samples or 2000
    dataset = Dataset(features=writer_features, dataset_meta=DatasetMeta(
        args.dataset.dataset_id, subset_name=args.dataset.subset_name, data_slice=range(train_samples)))
    # Set template to prepare encoding
    dataset.set_template(args.template.template_cls, model_id=args.model.model_id)
    # Preprocess the dataset to standard format
    dataset.map(preprocess_func=LatexOCRProcessor)
    # Encode dataset
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)

    config, kwargs = AutoConfig.from_pretrained(
        args.model.model_id,
        trust_remote_code=True,
        return_unused_kwargs=True,
    )

    if isinstance(config, Gemma4UnifiedConfig):    # 减层
        text_config = config.text_config
        if TEXT_NUM_LAYERS is not None and hasattr(text_config, 'num_hidden_layers'):
            text_config.num_hidden_layers = TEXT_NUM_LAYERS
            logger.info(f' modify > text_config.num_hidden_layers = {text_config.num_hidden_layers}')
    if hasattr(config, 'use_cache'):
        config.use_cache = False

    # Use a TransformersModel
    from transformers import AutoModelForMultimodalLM
    model = TransformersModel(
        model_cls=AutoModelForMultimodalLM,
        model_id=args.model.model_id,
        config=config,
        device_mesh=device_mesh,
        strategy=args.model.strategy,
        ignore_mismatched_sizes=IGNORE_MISMATCHED_SIZES,
        fsdp_config={
            'reshard_after_forward': True,
            'expert_parallel': {
                'enabled': True,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            }
        },
    )

    lora_config = LoraConfig(**args.get_lora_args())

    # Add a lora to model
    model.add_adapter_to_model(args.lora.adapter_name, lora_config,
                               gradient_accumulation_steps=args.training.gradient_accumulation_steps)
    # Add Optimizer
    model.set_optimizer(optimizer_cls=args.optimizer.optimizer_cls, lr=args.optimizer.learning_rate)

    # Add LRScheduler
    model.set_lr_scheduler(
        scheduler_cls=args.scheduler.scheduler_cls, num_warmup_steps=args.scheduler.num_warmup_steps,
        num_training_steps=len(dataloader))

    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    best_eval_loss = float('inf')

    ### eval dataset and dataloader
    eval_samples = args.training.eval_samples or 100
    eval_dataset = Dataset(features=writer_features, dataset_meta=DatasetMeta(
        args.dataset.dataset_id, subset_name=args.dataset.subset_name, data_slice=range(eval_samples)))
    eval_dataset.set_template(args.template.template_cls, model_id=args.model.model_id)
    eval_dataset.map(preprocess_func=LatexOCRProcessor)
    eval_dataset.encode()
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.training.batch_size)
    save_step = args.training.save_steps
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()

        if step % args.training.log_interval == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, Train metric: {metric}')

        if step % save_step == 0:
            metrics = evaluate(model, eval_dataloader)
            metrics['step'] = step
            if float(metrics['loss']) < best_eval_loss:
                # model.save(f'checkpoint-{step}')
                best_eval_loss = float(metrics['loss'])
            metrics['best_eval_loss'] = best_eval_loss
            logger.info(f'Current is step {step} of {len(dataloader)}, Eval metric: {metrics}')

    # model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()
