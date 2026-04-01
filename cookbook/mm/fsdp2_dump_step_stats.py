import os
import random
from pathlib import Path

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_logger
from twinkle.data_format import Message, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import DatasetMeta, LazyDataset
from twinkle.model import TransformersModel
from twinkle.preprocessor import Preprocessor

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
DATASET_ID = os.environ.get('DATASETS', 'ms://AI-ModelScope/LaTeX_OCR')
SEED = int(os.environ.get('TWINKLE_COMPARE_SEED', '1234'))
TARGET_STEP = int(os.environ.get('TWINKLE_COMPARE_STEP', '0'))
BATCH_SIZE = int(os.environ.get('TWINKLE_COMPARE_BATCH_SIZE', '4'))
MAX_LENGTH = int(os.environ.get('TWINKLE_COMPARE_MAX_LENGTH', '1024'))
OUTPUT_PATH = os.environ.get('TWINKLE_COMPARE_OUTPUT', 'compare_step_stats.pt')
PARAM_FILTERS = [p.strip() for p in os.environ.get('TWINKLE_COMPARE_PARAMS', '').split(',') if p.strip()]
ULYSSES_SIZE = os.environ.get('ULYSSES_SIZE')

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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def _build_device_mesh() -> DeviceMesh:
    kwargs = {
        'fsdp_size': 2,
        'device_type': Platform.get_platform().device_prefix(),
    }
    if ULYSSES_SIZE:
        kwargs['ulysses_size'] = int(ULYSSES_SIZE)
    return DeviceMesh.from_sizes(**kwargs)


def _create_dataloader(device_mesh: DeviceMesh) -> DataLoader:
    dataset = LazyDataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(2000)))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=MAX_LENGTH)
    dataset.map(LatexOCRProcessor)
    dataset.encode()
    return DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        device_mesh=device_mesh,
        shuffle=False,
        num_workers=0,
        max_retries=1,
    )


def _build_model(device_mesh: DeviceMesh) -> TransformersModel:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    model = TransformersModel(
        model_id=MODEL_ID,
        model_cls=Qwen3_5ForConditionalGeneration,
        device_mesh=device_mesh,
        strategy='native_fsdp',
    )
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4, adapter_name='default')
    return model


def _should_keep_param(name: str) -> bool:
    if not PARAM_FILTERS:
        return True
    return any(token in name for token in PARAM_FILTERS)


def _collect_grad_snapshot(model: TransformersModel) -> dict:
    grads = {}
    for name, param in model._get_trainable_parameters('default').items():
        if param.grad is None or not _should_keep_param(name):
            continue
        grads[name] = param.grad.detach().float().cpu().clone()
    return grads


def main():
    device_mesh = _build_device_mesh()
    twinkle.initialize(
        mode='local',
        global_device_mesh=device_mesh,
        seed=SEED,
        full_determinism=True,
        lazy_collect=False,
    )
    _seed_everything(SEED)

    dataloader = _create_dataloader(device_mesh)
    model = _build_model(device_mesh)

    captured = None
    for step, batch in enumerate(dataloader):
        outputs = model.forward_backward(inputs=batch, adapter_name='default')
        if step == TARGET_STEP:
            captured = {
                'seed': SEED,
                'ulysses_size': getattr(device_mesh, 'ulysses_size', None),
                'step': step,
                'loss': float(outputs['loss']),
                'param_filters': PARAM_FILTERS,
                'gradients': _collect_grad_snapshot(model),
            }
            break
        model.clip_grad_and_step(adapter_name='default')

    if captured is None:
        raise RuntimeError(f'Failed to capture step {TARGET_STEP}; dataloader only had {len(dataloader)} steps.')

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, output_path)
    logger.info(f'Saved compare snapshot to {output_path}')
    logger.info(
        f"step={captured['step']} ulysses_size={captured['ulysses_size']} loss={captured['loss']:.8f} "
        f"grad_tensors={len(captured['gradients'])}")


if __name__ == '__main__':
    main()
