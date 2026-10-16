"""Generate a spectral allocation or train one Hybrid LoRA adapter.

Use ``--generate-allocation-only`` to analyze the base model and write the
allocation JSON. Without that flag, the script requires an existing allocation
JSON and starts training directly.
"""

import json
import os
from pathlib import Path

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.cli import Args, CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.model.transformers.hybrid.spectral_allocation import (
    CANDIDATE_TYPES,
    allocate_spectral_modules,
    build_spectral_lora_config,
    build_spectral_param_groups,
    compute_spectral_scores,
    resolve_spectral_config_path,
    select_spectral_targets,
)
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()


def allocation_path(args: Args) -> Path:
    """Return the configured allocation path, or the default output path."""
    path, _ = resolve_spectral_config_path(
        args.extra.get('spectral_config'),
        Path(args.training.output_dir),
    )
    return path


def load_allocation(path: Path, args: Args) -> LoraConfig:
    """Load an existing allocation as a PEFT Hybrid LoRA config."""
    if not path.is_file():
        raise FileNotFoundError(f'Hybrid LoRA allocation does not exist: {path}. '
                                'Run this script with --generate-allocation-only first.')

    with path.open(encoding='utf-8') as handle:
        raw_config = json.load(handle)
    if not isinstance(raw_config, dict):
        raise ValueError('Hybrid LoRA allocation JSON must contain an object.')
    if raw_config.get('method') not in (None, 'spectral_hybrid_lora'):
        raise ValueError(f'Unsupported allocation method: {raw_config.get("method")!r}.')

    def module_list(primary_key: str, peft_key: str):
        value = raw_config.get(primary_key, raw_config.get(peft_key))
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)) or not all(isinstance(item, str) for item in value):
            raise ValueError(f'Hybrid LoRA allocation {primary_key} must be a list of module names.')
        return sorted(set(value))

    s_fft = module_list('s_fft', 'modules_to_save')
    s_lora = module_list('s_lora', 'target_modules')
    overlap = set(s_fft) & set(s_lora)
    if overlap:
        raise ValueError(f'Hybrid LoRA modules cannot be both FFT and LoRA: {sorted(overlap)}.')

    r = int(raw_config.get('r', args.lora.lora_r))
    lora_alpha = int(raw_config.get('lora_alpha', args.lora.lora_alpha))
    lora_dropout = float(raw_config.get('lora_dropout', args.lora.lora_dropout))
    return build_spectral_lora_config(
        s_lora=s_lora,
        s_fft=s_fft,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )


def generate_allocation(path: Path, args: Args) -> None:
    """Analyze the base model and write the spectral allocation JSON."""
    if Platform.get_world_size() != 1:
        raise ValueError('Allocation generation only supports one process. Use --nproc_per_node=1.')

    r = int(args.extra.get('spectral_r', args.lora.lora_r))
    lora_alpha = int(args.extra.get('spectral_alpha', args.lora.lora_alpha))
    fft_ratio = float(args.extra.get('spectral_fft_ratio', 0.1))
    epsilon = float(args.extra.get('spectral_epsilon', 1e-12))
    target_modules = args.lora.lora_target_modules or list(CANDIDATE_TYPES.values())
    if args.model.memory_efficient_init:
        raise ValueError('Allocation generation requires materialized weights; disable memory_efficient_init.')

    logger.info(f'Loading the base model to generate Hybrid LoRA allocation '
                f'(rank={r}, FFT budget={fft_ratio:.1%})')
    model = TransformersModel(model_id=args.model.model_id)

    target_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
    )
    targets = select_spectral_targets(model.model, target_config)
    param_counts = {name: module.weight.numel() for name, module in targets.items()}
    cache_dir = Path(
        args.extra.get(
            'spectral_cache_dir',
            Path(args.training.output_dir) / 'spectral-spectrum-cache',
        )).expanduser()
    scores = compute_spectral_scores(
        model.model,
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
    realized_fft_ratio = fft_params / total_params
    config = {
        'method': 'spectral_hybrid_lora',
        'model_id': args.model.model_id,
        's_fft': s_fft,
        's_lora': s_lora,
        'r': r,
        'lora_alpha': lora_alpha,
        'lora_dropout': 0.0,
        'fft_ratio': fft_ratio,
        'realized_fft_param_ratio': realized_fft_ratio,
        'spectral_epsilon': epsilon,
        'metrics': {
            name: {
                **scores.metrics[name],
                'score': scores[name],
            }
            for name in sorted(scores)
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f'{path.suffix}.tmp')
    with temporary_path.open('w', encoding='utf-8') as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary_path, path)
    logger.info(f'Hybrid LoRA allocation written to {path}: '
                f'{len(s_fft)} FFT modules, {len(s_lora)} LoRA modules '
                f'({realized_fft_ratio:.1%} of candidate parameters use FFT)')


def build_dataset(args: Args, train_samples: int) -> Dataset:
    dataset_id = args.dataset.dataset_id or 'ms://swift/self-cognition'
    dataset = Dataset(dataset_meta=DatasetMeta(dataset_id, data_slice=range(train_samples)))
    dataset.set_template(
        args.template.template_cls,
        model_id=args.model.model_id,
        max_length=args.template.max_length,
        truncation_strategy=args.template.truncation_strategy,
        enable_thinking=args.template.enable_thinking,
    )
    if dataset_id == 'ms://swift/self-cognition':
        dataset.map(SelfCognitionProcessor(
            args.extra.get('model_name', 'twinkle模型'),
            args.extra.get('model_author', 'ModelScope社区'),
        ))
    dataset.encode(num_proc=8, load_from_cache_file=True)
    return dataset


def train(path: Path, args: Args) -> None:
    """Load an existing allocation and train one Hybrid LoRA adapter."""
    hybrid_config = load_allocation(path, args)
    logger.info(f'Using Hybrid LoRA allocation {path} '
                f'({len(hybrid_config.modules_to_save)} FFT modules, '
                f'{len(hybrid_config.target_modules)} LoRA modules)')

    train_samples = args.training.train_samples or 1000
    dataset = build_dataset(args, train_samples)
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)
    model = TransformersModel(model_id=args.model.model_id)
    model.add_adapter_to_model(
        args.lora.adapter_name,
        hybrid_config,
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


def main() -> None:
    args = CLI.from_args()
    generate_only = bool(args.extra.get('generate_allocation_only', False))
    if generate_only:
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('Allocation generation only supports one process. Use --nproc_per_node=1.')
        device_mesh = DeviceMesh.from_sizes(world_size=1, dp_size=1)
    else:
        device_mesh = DeviceMesh.from_sizes(fsdp_size=args.infra.fsdp_size, dp_size=args.infra.dp_size)
    twinkle.initialize(mode=args.infra.mode, global_device_mesh=device_mesh)

    path = allocation_path(args)
    if generate_only:
        generate_allocation(path, args)
    else:
        train(path, args)


if __name__ == '__main__':
    main()
