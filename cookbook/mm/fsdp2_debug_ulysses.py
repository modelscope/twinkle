from typing import Any

import torch
from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import Message, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import DatasetMeta, LazyDataset
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_ID = 'ms://AI-ModelScope/LaTeX_OCR'
MAX_LENGTH = 1024
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
DEBUG_STEPS = {0, 20}

# Construct a device_mesh, fsdp=2
device_mesh = DeviceMesh.from_sizes(fsdp_size=2)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


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


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().cpu()
    summary = {
        'shape': list(cpu_tensor.shape),
        'dtype': str(cpu_tensor.dtype),
        'numel': int(cpu_tensor.numel()),
    }
    if cpu_tensor.numel() == 0:
        return summary
    if cpu_tensor.dtype.is_floating_point:
        float_tensor = cpu_tensor.float()
        summary.update({
            'sum': float(float_tensor.sum().item()),
            'mean': float(float_tensor.mean().item()),
            'std': float(float_tensor.std(unbiased=False).item()),
            'abs_max': float(float_tensor.abs().max().item()),
        })
    else:
        int_tensor = cpu_tensor.to(torch.int64)
        summary.update({
            'sum': int(int_tensor.sum().item()),
            'min': int(int_tensor.min().item()),
            'max': int(int_tensor.max().item()),
        })
    return summary


def _summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    if isinstance(batch, dict):
        summary = {'batch_type': 'dict'}
        for key, value in batch.items():
            if torch.is_tensor(value):
                summary[key] = _tensor_summary(value)
        if 'labels' in batch and torch.is_tensor(batch['labels']):
            summary['labels_non_ignore'] = int((batch['labels'] != -100).sum().item())
        if 'attention_mask' in batch and torch.is_tensor(batch['attention_mask']):
            summary['attention_mask_total'] = int(batch['attention_mask'].sum().item())
        return summary

    if isinstance(batch, list):
        summary = {
            'batch_type': 'list',
            'batch_size': len(batch),
        }
        if batch and isinstance(batch[0], dict):
            first_item = batch[0]
            summary['item_keys'] = sorted(first_item.keys())
            if 'messages' in first_item and isinstance(first_item['messages'], list):
                summary['first_item_message_count'] = len(first_item['messages'])
                user_content = first_item['messages'][0].get('content', '')
                if isinstance(user_content, str):
                    summary['first_item_user_content_preview'] = user_content[:120]
                summary['first_item_image_count'] = len(first_item['messages'][0].get('images', []) or [])
        return summary

    return {
        'batch_type': type(batch).__name__,
        'repr': repr(batch)[:200],
    }


def _summarize_model_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        item = _tensor_summary(value)
        if key == 'labels':
            item['non_ignore'] = int((value != -100).sum().item())
        if key == 'attention_mask':
            item['positive'] = int((value > 0).sum().item())
        summary[key] = item
    return summary


def _resolve_first_linear_attn_module(model: TransformersModel):
    module = model.model
    base_model = getattr(module, 'model', module)
    language_model = getattr(getattr(base_model, 'model', None), 'language_model', None)
    if language_model is None:
        return None
    layers = getattr(language_model, 'layers', None)
    if not layers:
        return None
    return getattr(layers[0], 'linear_attn', None)


def _capture_layer0_linear_attn_summary(model: TransformersModel, batch: dict[str, Any]) -> dict[str, Any] | None:
    linear_attn = _resolve_first_linear_attn_module(model)
    if linear_attn is None:
        return None

    captured: dict[str, Any] = {}

    def _hook(_module, hook_inputs, hook_output):
        if hook_inputs and torch.is_tensor(hook_inputs[0]):
            captured['input'] = _tensor_summary(hook_inputs[0])
        output_tensor = hook_output[0] if isinstance(hook_output, (tuple, list)) else hook_output
        if torch.is_tensor(output_tensor):
            captured['output'] = _tensor_summary(output_tensor)

    handle = linear_attn.register_forward_hook(_hook)
    try:
        model.forward_only(inputs=batch)
        model.calculate_loss()
    finally:
        handle.remove()
    return captured or None


def _capture_processor_summary(model: TransformersModel, batch: dict[str, Any]) -> dict[str, Any]:
    optimizer_config = model.optimizer_group[model._get_default_group()]
    processor = optimizer_config.processor
    processed = processor(batch, sp_strategy=model.sp_strategy)
    return _summarize_model_inputs(processed)


def eval(model):
    dataset = LazyDataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(100)))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID)
    dataset.map(LatexOCRProcessor)
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=EVAL_BATCH_SIZE)
    for _, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    return model.calculate_metric(is_training=False)


def train():
    dataset = LazyDataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(2000)))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=MAX_LENGTH)
    dataset.map(LatexOCRProcessor)
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=TRAIN_BATCH_SIZE)

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    model = TransformersModel(model_id=MODEL_ID, model_cls=Qwen3_5ForConditionalGeneration)
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')

    best_loss = 99.0
    for step, batch in enumerate(dataloader):
        if step in DEBUG_STEPS:
            logger.info(f'DEBUG batch_summary step={step}: {_summarize_batch(batch)}')
            logger.info(f'DEBUG processor_outputs step={step}: {_capture_processor_summary(model, batch)}')
            logger.info(f'DEBUG layer0_linear_attn step={step}: {_capture_layer0_linear_attn_summary(model, batch)}')

        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()

        if step % 20 == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')

        if step > 0 and step % 40 == 0:
            metrics = eval(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = step
            if best_loss > float(metrics['loss']):
                model.save(f'checkpoint-{step}')
                best_loss = float(metrics['loss'])
    model.save('last-checkpoint')


if __name__ == '__main__':
    train()
