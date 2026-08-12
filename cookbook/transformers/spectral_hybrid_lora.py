import json
import os
from pathlib import Path

import torch.distributed as dist
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.cli import CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.model.base import initialize_process_group
from twinkle.model.transformers.spectral_hybrid_lora import (
    CANDIDATE_TYPES,
    allocate_spectral_modules,
    build_spectral_lora_config,
    build_spectral_param_groups,
    compute_spectral_scores,
    resolve_spectral_config_path,
    select_spectral_targets,
)

logger = get_logger()
args = CLI.from_args()

device_mesh = DeviceMesh.from_sizes(fsdp_size=args.infra.fsdp_size, dp_size=args.infra.dp_size)
twinkle.initialize(mode=args.infra.mode, global_device_mesh=device_mesh)


def build_dataset(data_slice) -> Dataset:
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset.dataset_id, data_slice=data_slice))
    dataset.set_template(
        args.template.template_cls,
        model_id=args.model.model_id,
        max_length=args.template.max_length,
        truncation_strategy=args.template.truncation_strategy,
        enable_thinking=args.template.enable_thinking,
    )
    dataset.encode(num_proc=8, load_from_cache_file=True)
    return dataset


def load_spectral_config(config_path: Path) -> LoraConfig:
    """Load an existing Spectral Hybrid LoRA allocation."""
    with config_path.open(encoding='utf-8') as handle:
        raw_config = json.load(handle)
    if not isinstance(raw_config, dict):
        raise ValueError('Spectral config JSON must contain an object.')
    if raw_config.get('method') not in (None, 'spectral_hybrid_lora'):
        raise ValueError(f'Unsupported spectral config method: {raw_config.get("method")!r}.')

    def module_list(primary_key, peft_key):
        value = raw_config.get(primary_key, raw_config.get(peft_key))
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)) or not all(isinstance(item, str) for item in value):
            raise ValueError(f'Spectral config {primary_key} must be a list of module names.')
        return sorted(set(value))

    s_fft = module_list('s_fft', 'modules_to_save')
    s_lora = module_list('s_lora', 'target_modules')
    overlap = set(s_fft) & set(s_lora)
    if overlap:
        raise ValueError(f'Spectral modules cannot be both FFT and LoRA: {sorted(overlap)}')
    r = int(raw_config.get('r', args.extra.get('spectral_r', args.lora.lora_r)))
    lora_alpha = int(raw_config.get('lora_alpha', args.extra.get('spectral_alpha', r * 2)))
    lora_dropout = float(raw_config.get('lora_dropout', 0.0))
    return build_spectral_lora_config(
        s_lora,
        s_fft,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )


def compute_allocation(config_path: Path) -> None:
    """Compute and persist a data-free spectral allocation on the master rank."""
    r = int(args.extra.get('spectral_r', args.lora.lora_r))
    lora_alpha = int(args.extra.get('spectral_alpha', r * 2))
    fft_ratio = float(args.extra.get('spectral_fft_ratio', 0.1))
    epsilon = float(args.extra.get('spectral_epsilon', 1e-12))

    logger.info(f'Spectral Hybrid LoRA allocation: loading pretrained model '
                f'(r={r}, alpha={lora_alpha}, FFT budget={fft_ratio:.1%})')
    analysis_mesh = DeviceMesh.from_sizes(world_size=1, dp_size=1)
    base_model = TransformersModel(model_id=args.model.model_id, device_mesh=analysis_mesh)
    if base_model._memory_efficient_init:
        raise ValueError('Spectral scoring requires materialized weights; '
                         'disable memory_efficient_init.')
    target_modules = args.lora.lora_target_modules or list(CANDIDATE_TYPES.values())
    target_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
    )
    targets = select_spectral_targets(base_model.model, target_config)
    param_counts = {name: module.weight.numel() for name, module in targets.items()}
    cache_dir = Path(args.extra.get(
        'spectral_cache_dir', Path(args.training.output_dir) / 'spectral-spectrum-cache')).expanduser()
    scores = compute_spectral_scores(
        base_model.model,
        target_config,
        r=r,
        cache_dir=cache_dir,
        cache_key=str(args.model.model_id),
        epsilon=epsilon,
        log_interval=args.training.log_interval,
        broadcast=False,
    )
    s_fft, s_lora = allocate_spectral_modules(scores, param_counts, fft_ratio=fft_ratio)
    fft_params = sum(param_counts[name] for name in s_fft)
    total_params = sum(param_counts.values())
    realized_fft_param_ratio = fft_params / total_params
    logger.info(f'Spectral Hybrid LoRA allocation: {len(s_fft)} FFT modules, {len(s_lora)} LoRA modules '
                f'({realized_fft_param_ratio:.1%} of candidate params to FFT; cache={cache_dir})')
    logger.info(f'Spectral FFT modules: {", ".join(s_fft) if s_fft else "(none)"}')

    raw_config = {
        'method': 'spectral_hybrid_lora',
        'model_id': args.model.model_id,
        's_fft': s_fft,
        's_lora': s_lora,
        'r': r,
        'lora_alpha': lora_alpha,
        'lora_dropout': 0.0,
        'fft_ratio': fft_ratio,
        'realized_fft_param_ratio': realized_fft_param_ratio,
        'spectral_epsilon': epsilon,
        'metrics': {
            name: {
                **scores.metrics[name],
                'score': scores[name],
            }
            for name in sorted(scores)
        },
    }
    if Platform.is_master():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = config_path.with_suffix(f'{config_path.suffix}.tmp')
        with temporary_path.open('w', encoding='utf-8') as handle:
            json.dump(raw_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary_path, config_path)
        logger.info(f'Spectral config written to {config_path}')

    del base_model


def resolve_spectral_config() -> LoraConfig:
    config_value = args.extra.get('spectral_config')
    config_path, should_load = resolve_spectral_config_path(config_value, Path(args.training.output_dir))
    if should_load:
        spectral_config = load_spectral_config(config_path)
        logger.info(f'Using existing spectral config {config_path}; skipping spectral scoring '
                    f'({len(spectral_config.modules_to_save or [])} FFT modules, '
                    f'{len(spectral_config.target_modules or [])} LoRA modules)')
        return spectral_config
    if config_value:
        logger.info(f'Spectral config {config_path} does not exist; computing it from model weights')
    else:
        logger.info(f'No spectral config supplied; computing allocation and writing {config_path}')

    initialize_process_group()
    if Platform.is_master():
        compute_allocation(config_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return load_spectral_config(config_path)


def train() -> None:
    train_samples = args.training.train_samples or 1000
    spectral_config = resolve_spectral_config()

    dataset = build_dataset(range(train_samples))
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)
    model = TransformersModel(model_id=args.model.model_id)
    model.add_adapter_to_model(
        args.lora.adapter_name,
        spectral_config,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
    )
    param_groups = build_spectral_param_groups(
        model.strategy.unwrap_model(model.model),
        lr_lora=float(args.extra.get('spectral_lr_lora', args.optimizer.learning_rate)),
        lr_fft=float(args.extra.get('spectral_lr_fft', 1e-6)),
        weight_decay=args.optimizer.weight_decay,
        adapter_name=args.lora.adapter_name,
    )
    model.set_optimizer(optimizer_cls=args.optimizer.optimizer_cls, params=param_groups)
    model.set_lr_scheduler(
        scheduler_cls=args.scheduler.scheduler_cls,
        num_warmup_steps=args.scheduler.num_warmup_steps,
        num_training_steps=len(dataloader),
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    optimizer_group = model.optimizer_group[args.lora.adapter_name]
    for batch in dataloader:
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        cur_step = optimizer_group.cur_step
        if cur_step % args.training.log_interval == 0:
            logger.info(f'step {cur_step}/{len(dataloader)}, '
                        f'metric: {model.calculate_metric(is_training=True)}')
    model.save(
        'last-checkpoint',
        output_dir=args.training.output_dir,
        adapter_name=args.lora.adapter_name,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


if __name__ == '__main__':
    train()
