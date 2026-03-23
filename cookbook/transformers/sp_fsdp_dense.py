import json
import math
import os
from functools import partial

import numpy as np
import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.model.transformers.models import TwinkleQwen3_5ForCausalLM
from twinkle.preprocessor import SelfCognitionProcessor
# TWINKLE_PROFILE_MODULE_MEMORY=1 \
# TWINKLE_PROFILE_MODULE_MEMORY_STEP=0 \
# TWINKLE_PROFILE_MODULE_MEMORY_RANK=0 \
# python cookbook/transformers/sp_fsdp_dense.py

logger = get_logger()
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASETS = 'ms://swift/self-cognition'
TRAIN_SEED = int(os.environ.get('TRAIN_SEED', '1234'))
TRAIN_DETERMINISTIC = os.environ.get('TRAIN_DETERMINISTIC', '1') == '1'
TRAIN_NUM_WORKERS = int(os.environ.get('TRAIN_NUM_WORKERS', '0'))
TRAIN_SHUFFLE = os.environ.get('TRAIN_SHUFFLE', '0') == '1'
TRAIN_ATTENTION_DROPOUT = float(os.environ.get('TRAIN_ATTENTION_DROPOUT', '0.0'))
TRAIN_LORA_DROPOUT = float(os.environ.get('TRAIN_LORA_DROPOUT', '0.0'))

device_group = [DeviceGroup(
    name='default',
    ranks=[0, 1, 2, 3],
    device_type=Platform.get_platform().device_prefix(),
)]

# FSDP + SP validation over 4 GPUs: dp=2, fsdp=2 (SP only affects input slicing)
device_mesh = DeviceMesh(
    device_type='cuda',
    mesh=np.arange(4).reshape(2, 2),
    mesh_dim_names=('dp', 'fsdp'),
    ulysses_size=2,
)

twinkle.initialize(
    mode='local',
    nproc_per_node=4,
    global_device_mesh=device_mesh,
    lazy_collect=False,
)
twinkle.framework_util.seed_everything(TRAIN_SEED, TRAIN_DETERMINISTIC)


def _memory_api():
    device_type = Platform.get_platform().device_prefix()
    device_api = getattr(torch, device_type, None)
    if device_api is None or not hasattr(device_api, 'is_available') or not device_api.is_available():
        return None, None
    return device_type, device_api


def _format_mib(num_bytes):
    return f'{num_bytes / (1024 ** 2):.1f} MiB'


def _get_memory_stats():
    device_type, device_api = _memory_api()
    if device_api is None:
        return {}

    if hasattr(device_api, 'synchronize'):
        device_api.synchronize()

    current_device = device_api.current_device() if hasattr(device_api, 'current_device') else 0
    return {
        'rank': Platform.get_rank(),
        'local_rank': Platform.get_local_rank(),
        'device': f'{device_type}:{current_device}',
        'mem_allocated': _format_mib(device_api.memory_allocated()),
        'mem_reserved': _format_mib(device_api.memory_reserved()),
        'mem_peak_allocated': _format_mib(device_api.max_memory_allocated()),
        'mem_peak_reserved': _format_mib(device_api.max_memory_reserved()),
    }


def _reset_peak_memory_stats():
    _, device_api = _memory_api()
    if device_api is not None and hasattr(device_api, 'reset_peak_memory_stats'):
        device_api.reset_peak_memory_stats()


def _memory_mib_value(num_bytes):
    return round(float(num_bytes) / (1024 ** 2), 3)


def _shape_of(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        for item in value:
            shape = _shape_of(item)
            if shape is not None:
                return shape
    return None


class _ModuleMemoryProfiler:

    TARGET_CLASS_NAMES = {
        'TwinkleQwen3_5DecoderLayer',
        'TwinkleQwen3_5GatedDeltaNet',
        'Qwen3_5Attention',
        'Qwen3_5MLP',
    }

    def __init__(self, model: TransformersModel):
        self.model = model
        self.enabled = os.environ.get('TWINKLE_PROFILE_MODULE_MEMORY') == '1'
        self.target_step = int(os.environ.get('TWINKLE_PROFILE_MODULE_MEMORY_STEP', '0'))
        self.target_rank = int(os.environ.get('TWINKLE_PROFILE_MODULE_MEMORY_RANK', '0'))
        self.max_records = int(os.environ.get('TWINKLE_PROFILE_MODULE_MEMORY_LIMIT', '16'))
        self.active = False
        self.handles = []
        self.entries = {}
        self.records = []

    def attach(self):
        if not self.enabled or Platform.get_rank() != self.target_rank:
            return
        base_model = getattr(self.model, 'model', None)
        if base_model is None:
            return
        for name, module in base_model.named_modules():
            if module.__class__.__name__ not in self.TARGET_CLASS_NAMES:
                continue
            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(name), with_kwargs=True))
            self.handles.append(module.register_forward_hook(self._make_post_hook(name), with_kwargs=True))

    def _make_pre_hook(self, name):

        def _hook(module, args, kwargs):
            if not self.active:
                return
            _, device_api = _memory_api()
            if device_api is None:
                return
            if hasattr(device_api, 'synchronize'):
                device_api.synchronize()
            self.entries[id(module)] = {
                'name': name,
                'class_name': module.__class__.__name__,
                'input_shape': _shape_of(args) or _shape_of(kwargs),
                'pre_allocated_mib': _memory_mib_value(device_api.memory_allocated()),
                'pre_reserved_mib': _memory_mib_value(device_api.memory_reserved()),
                'pre_peak_allocated_mib': _memory_mib_value(device_api.max_memory_allocated()),
                'pre_peak_reserved_mib': _memory_mib_value(device_api.max_memory_reserved()),
            }

        return _hook

    def _make_post_hook(self, name):

        def _hook(module, args, kwargs, output):
            del args, kwargs
            if not self.active:
                return
            _, device_api = _memory_api()
            if device_api is None:
                return
            if hasattr(device_api, 'synchronize'):
                device_api.synchronize()
            entry = self.entries.pop(id(module), None)
            if entry is None:
                return
            post_allocated_mib = _memory_mib_value(device_api.memory_allocated())
            post_reserved_mib = _memory_mib_value(device_api.memory_reserved())
            post_peak_allocated_mib = _memory_mib_value(device_api.max_memory_allocated())
            post_peak_reserved_mib = _memory_mib_value(device_api.max_memory_reserved())
            entry.update({
                'output_shape': _shape_of(output),
                'post_allocated_mib': post_allocated_mib,
                'post_reserved_mib': post_reserved_mib,
                'post_peak_allocated_mib': post_peak_allocated_mib,
                'post_peak_reserved_mib': post_peak_reserved_mib,
                'delta_allocated_mib': round(post_allocated_mib - entry['pre_allocated_mib'], 3),
                'delta_reserved_mib': round(post_reserved_mib - entry['pre_reserved_mib'], 3),
                'delta_peak_allocated_mib': round(post_peak_allocated_mib - entry['pre_peak_allocated_mib'], 3),
                'delta_peak_reserved_mib': round(post_peak_reserved_mib - entry['pre_peak_reserved_mib'], 3),
            })
            self.records.append(entry)

        return _hook

    def start_step(self, step: int):
        self.active = self.enabled and Platform.get_rank() == self.target_rank and step == self.target_step
        self.entries.clear()
        self.records.clear()
        if self.active:
            _reset_peak_memory_stats()

    def finish_step(self, step: int):
        if not self.active:
            return
        step_memory = _get_memory_stats()
        sorted_records = sorted(self.records, key=lambda item: item['delta_peak_allocated_mib'], reverse=True)
        logger.info(
            'Module memory profile summary: '
            + json.dumps(
                {
                    'step': step,
                    'rank': Platform.get_rank(),
                    'total_step_peak_allocated_mib': step_memory.get('mem_peak_allocated'),
                    'total_step_peak_reserved_mib': step_memory.get('mem_peak_reserved'),
                    'top_forward_modules_by_peak_allocated': sorted_records[:self.max_records],
                },
                ensure_ascii=False,
            ))
        self.active = False

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _get_runtime_backend_info(model: TransformersModel):
    model._ensure_sp_strategy()

    underlying_model = getattr(model, 'model', None)
    llm_model = getattr(underlying_model, 'model', underlying_model)
    config = getattr(underlying_model, 'config', None)

    attn_implementation = None
    attn_implementation_internal = None
    if config is not None:
        attn_implementation = getattr(config, '_attn_implementation', None)
        attn_implementation_internal = getattr(config, '_attn_implementation_internal', None)

    return {
        'model_cls': type(underlying_model).__name__ if underlying_model is not None else None,
        'llm_model_cls': type(llm_model).__name__ if llm_model is not None else None,
        'attn_implementation': attn_implementation,
        'attn_implementation_internal': attn_implementation_internal,
        'requires_cu_seq_lens_q': bool(getattr(llm_model, 'requires_cu_seq_lens_q', False)),
        'sp_enabled': bool(getattr(model, '_enable_sp', False)),
        'ulysses_size': getattr(getattr(model, 'device_mesh', None), 'ulysses_size', None),
        'sp_strategy_enabled': bool(getattr(getattr(model, 'sp_strategy', None), 'enabled', False)),
        'sp_strategy_ulysses_size': getattr(getattr(model, 'sp_strategy', None), 'ulysses_size', None),
    }


def eval(model):
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=range(100)),
        batch_size=4,
        device_mesh=device_mesh,
        num_workers=TRAIN_NUM_WORKERS,
        shuffle=False,
    )
    for _, batch in enumerate(dataloader):
        model.forward_only(inputs=batch, adapter_name='default')
        model.calculate_loss(adapter_name='default')
    return model.calculate_metric(is_training=False, adapter_name='default')


def create_dataset(data_slice=None):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASETS, data_slice=range(500)))
    dataset.set_template('Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    return dataset


def train():
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=None),
        batch_size=8,
        device_mesh=device_mesh,
        num_workers=TRAIN_NUM_WORKERS,
        shuffle=TRAIN_SHUFFLE,
    )

    model = TransformersModel(
        model_id=MODEL_ID,
        model_cls=TwinkleQwen3_5ForCausalLM,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        attention_dropout=TRAIN_ATTENTION_DROPOUT,
    )

    lora_config = LoraConfig(target_modules='all-linear', lora_dropout=TRAIN_LORA_DROPOUT)
    model.add_adapter_to_model('default', lora_config)
    grad_accumulation_steps = model.optimizer_group['default'].gradient_accumulation_steps
    num_optimizer_steps = math.ceil(len(dataloader) / grad_accumulation_steps)
    log_every_optimizer_steps = 20
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=num_optimizer_steps,
        adapter_name='default',
    )

    logger.info(model.get_train_configs(adapter_name='default'))
    logger.info(
        f'Total micro steps: {len(dataloader)}, optimizer steps: {num_optimizer_steps}, '
        f'gradient_accumulation_steps: {grad_accumulation_steps}')
    logger.info(
        'Reproducibility config: '
        + str({
            'seed': TRAIN_SEED,
            'deterministic': TRAIN_DETERMINISTIC,
            'dataloader_shuffle': TRAIN_SHUFFLE,
            'dataloader_num_workers': TRAIN_NUM_WORKERS,
            'attention_dropout': TRAIN_ATTENTION_DROPOUT,
            'lora_dropout': TRAIN_LORA_DROPOUT,
        }))
    logger.info(f'Backend info: {_get_runtime_backend_info(model)}')
    logger.info(f'Initial memory: {_get_memory_stats()}')
    _reset_peak_memory_stats()
    module_memory_profiler = _ModuleMemoryProfiler(model)
    module_memory_profiler.attach()

    for step, batch in enumerate(dataloader):
        module_memory_profiler.start_step(step)
        model.forward_backward(inputs=batch, adapter_name='default')
        module_memory_profiler.finish_step(step)
        model.clip_grad_and_step(adapter_name='default')
        optimizer_step = step // grad_accumulation_steps
        is_optimizer_boundary = (step + 1) % grad_accumulation_steps == 0
        if is_optimizer_boundary and optimizer_step % log_every_optimizer_steps == 0:
            metric = model.calculate_metric(is_training=True, adapter_name='default')
            metric.update(_get_memory_stats())
            optimizer_step = metric.get('iters', optimizer_step)
            logger.info(
                f'Current is optimizer step {optimizer_step} of {num_optimizer_steps} '
                f'(micro step {step} of {len(dataloader)}), metric: {metric}')
    model.save('last-checkpoint', interval=1)
    module_memory_profiler.close()


if __name__ == '__main__':
    train()
