#!/usr/bin/env python3
"""Compute a data-free spectral FFT/LoRA allocation from pretrained spectra.

Example:
    python scripts/compute_spectral_hybrid_lora_config.py \
        --model-id ms://Qwen/Qwen3.5-9B \
        --spectral-r 64 \
        --spectral-fft-ratio 0.3 \
        --output-dir ./output/spectral_hybrid_lora \
        --spectral-config-output ./output/spectral_hybrid_lora/config.json
"""

import json
import os
from pathlib import Path
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_logger
from twinkle.cli import CLI
from twinkle.model import TransformersModel
from twinkle.model.transformers.spectral_hybrid_lora import (CANDIDATE_TYPES, allocate_spectral_modules,
                                                             compute_spectral_scores, select_spectral_targets)

logger = get_logger()
args = CLI.from_args()

# This utility is intentionally single-process: it only reads weights and writes one JSON config.
device_mesh = DeviceMesh.from_sizes(fsdp_size=1, dp_size=1)
twinkle.initialize(mode=args.infra.mode, global_device_mesh=device_mesh)


def main() -> None:
    if not args.model.model_id:
        raise ValueError('--model-id is required.')

    r = int(args.extra.get('spectral_r', args.lora.lora_r))
    lora_alpha = int(args.extra.get('spectral_alpha', r * 2))
    fft_ratio = float(args.extra.get('spectral_fft_ratio', 0.1))
    epsilon = float(args.extra.get('spectral_epsilon', 1e-12))
    output_path = Path(
        args.extra.get('spectral_config_output',
                       Path(args.training.output_dir) / 'spectral_hybrid_lora_config.json')).expanduser()
    cache_dir = Path(args.extra.get('spectral_cache_dir',
                                    Path(args.training.output_dir) / 'spectral-spectrum-cache')).expanduser()

    logger.info(f'Loading pretrained model for Spectral Hybrid LoRA scoring: {args.model.model_id}')
    model = TransformersModel(model_id=args.model.model_id)
    if model._memory_efficient_init:
        raise ValueError('Spectral scoring requires materialized weights; disable memory_efficient_init.')

    target_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=list(CANDIDATE_TYPES.values()),
    )
    targets = select_spectral_targets(model.model, target_config)
    param_counts = {name: module.weight.numel() for name, module in targets.items()}
    scores = compute_spectral_scores(
        model.model,
        target_config,
        r=r,
        cache_dir=cache_dir,
        cache_key=str(args.model.model_id),
        epsilon=epsilon,
        log_interval=args.training.log_interval,
    )
    counts = {name: param_counts[name] for name in scores}
    s_fft, s_lora = allocate_spectral_modules(scores, counts, fft_ratio=fft_ratio)
    fft_params = sum(counts[name] for name in s_fft)
    total_params = sum(counts.values())

    config = {
        'method': 'spectral_hybrid_lora',
        'model_id': args.model.model_id,
        's_fft': s_fft,
        's_lora': s_lora,
        'r': r,
        'lora_alpha': lora_alpha,
        'lora_dropout': 0.0,
        'fft_ratio': fft_ratio,
        'realized_fft_param_ratio': fft_params / total_params,
        'spectral_epsilon': epsilon,
        'metrics': {
            name: {
                'score': scores[name],
                **scores.metrics[name],
            }
            for name in sorted(scores)
        },
    }

    if Platform.is_master():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(f'{output_path.suffix}.tmp')
        with temporary_path.open('w', encoding='utf-8') as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary_path, output_path)
        logger.info(f'Spectral config written to {output_path}: '
                    f'{len(s_fft)} FFT modules, {len(s_lora)} LoRA modules, '
                    f'realized FFT parameter ratio={fft_params / total_params:.2%}')


if __name__ == '__main__':
    main()
