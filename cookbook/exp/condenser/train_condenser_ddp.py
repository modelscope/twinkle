"""Ray LoRA SFT for the condenser model on condense_300K.

Launch:
    python cookbook/exp/train_condenser_ddp.py
"""
from pathlib import Path

from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor

logger = get_logger()

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_ID = 'ms://twinkle-kit/condense_300K'
TEMPLATE_NAME = 'Qwen3_5Template'

DP_SIZE = 8
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 8
LOG_INTERVAL = 20
EVAL_INTERVAL = 200
EVAL_SAMPLES = 100
NUM_EPOCHS = 1

OUTPUT_DIR = './output/condenser_ddp'
RESUME_FROM_CHECKPOINT = None
RESUME_ONLY_MODEL = False
IGNORE_DATA_SKIP = False
ADAPTER_NAME = 'default'

class LegacySectionRenameProcessor(Preprocessor):
    """Rewrite legacy `## Read inline` / `## Call extract_compressed for` headers to `## Summary` / `## More`."""

    _REPLACEMENTS = (
        ('## Read inline', '## Summary'),
        ('## Call extract_compressed for', '## More'),
    )

    def __call__(self, batch):
        new_messages = []
        for msgs in batch['messages']:
            patched = []
            for m in msgs:
                content = m.get('content', '') or ''
                for old, new in self._REPLACEMENTS:
                    content = content.replace(old, new)
                patched.append({**m, 'content': content})
            new_messages.append(patched)
        return {'messages': new_messages}


def build_dataset() -> Dataset:
    dataset = Dataset(dataset_meta=DatasetMeta('/mnt/workspace/yzhao/tastelikefeet/condense_300K/train.jsonl'))
    dataset.map(LegacySectionRenameProcessor(), remove_columns=[], num_proc=16)
    dataset.set_template(TEMPLATE_NAME, model_id=MODEL_ID, max_length=40000, enable_thinking=False, truncation_strategy='delete')
    dataset.encode(load_from_cache_file=True, num_proc=64)
    return dataset


def train():
    device_groups = [DeviceGroup(name='model', ranks=DP_SIZE, device_type='GPU')]
    model_mesh = DeviceMesh.from_sizes(world_size=DP_SIZE, dp_size=4, fsdp_size=2)
    twinkle.initialize(mode='ray', nproc_per_node=DP_SIZE, groups=device_groups, global_device_mesh=model_mesh)

    dataset = build_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')

    model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
    total_optim_steps = (len(dataloader) * NUM_EPOCHS) // GRADIENT_ACCUMULATION_STEPS
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=50, num_training_steps=total_optim_steps)

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total micro-steps: {len(dataloader) * NUM_EPOCHS}, optim steps: {total_optim_steps}')

    for i in range(NUM_EPOCHS):
        for cur_step, batch in enumerate(dataloader):
            model.forward_backward(inputs=batch)
            model.clip_grad_and_step(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
            if cur_step % LOG_INTERVAL == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(f'Step {cur_step}/{len(dataloader) * NUM_EPOCHS}, metric: {metric}')
            if cur_step % 4000 == 0:
                model.save(f'step_{cur_step}', output_dir=OUTPUT_DIR)
    model.save('last_checkpoint', output_dir=OUTPUT_DIR)


if __name__ == '__main__':
    train()
