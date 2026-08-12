# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import hashlib
import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from peft import LoraConfig
from peft.tuners.tuners_utils import _maybe_include_all_linear_layers, check_target_module_exists
from torch import nn
from typing import Dict, List, Mapping, Optional, Tuple

from twinkle import get_logger

logger = get_logger()

CANDIDATE_TYPES: Dict[str, str] = {
    'q': 'q_proj',
    'k': 'k_proj',
    'v': 'v_proj',
    'o': 'o_proj',
    'gate': 'gate_proj',
    'up': 'up_proj',
    'down': 'down_proj',
}
_CANDIDATE_SUFFIXES = frozenset(CANDIDATE_TYPES.values())
_LAYER_RE = re.compile(r'\blayers\.(\d+)\.')


class ZeroShotScores(dict[str, float]):
    """Zero-shot spectral scores with per-module metric details."""

    def __init__(self, scores: Mapping[str, float], metrics: Mapping[str, Mapping[str, float]]) -> None:
        super().__init__(scores)
        self.metrics = {name: dict(values) for name, values in metrics.items()}


def compute_zero_shot_spectral_metrics(
    singular_values: torch.Tensor,
    r: int,
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    """Compute the pretrained-spectrum metrics used for zero-shot allocation."""
    if r <= 0:
        raise ValueError('Zero-shot LoRA rank r must be positive.')
    if singular_values.ndim != 1 or singular_values.numel() == 0:
        raise ValueError('Zero-shot scoring requires a non-empty singular-value vector.')
    if epsilon <= 0:
        raise ValueError('Zero-shot scoring epsilon must be positive.')

    values = singular_values.detach().to(device='cpu', dtype=torch.float64).abs()
    values = values.sort(descending=True).values
    stable_values = values.clamp_min(epsilon)
    probabilities = stable_values / stable_values.sum()
    entropy = -(probabilities * probabilities.log()).sum()
    effective_rank = entropy.exp()
    dimension = values.numel()

    rank = min(r, dimension)
    squared = values.square()
    rank_coverage = squared[:rank].sum() / squared.sum().clamp_min(epsilon)
    condition_number = values[0] / values[-1].clamp_min(epsilon)
    if dimension > 1:
        decay = torch.diff(stable_values.log()).abs().mean()
        inverse_decay = 1.0 / decay.clamp_min(epsilon)
    else:
        decay = torch.tensor(float('inf'), dtype=torch.float64)
        inverse_decay = torch.tensor(0.0, dtype=torch.float64)

    score = (0.3 * (effective_rank / dimension) + 0.3 * (1.0 - rank_coverage) + 0.2 *
             (torch.log1p(condition_number) / 10.0) + 0.2 * inverse_decay)
    return {
        'effective_rank': float(effective_rank.item()),
        'normalized_effective_rank': float((effective_rank / dimension).item()),
        'rank_coverage': float(rank_coverage.item()),
        'condition_number': float(condition_number.item()),
        'decay': float(decay.item()),
        'score': float(score.item()),
    }


def _is_candidate_module(module_name: str) -> bool:
    return _LAYER_RE.search(module_name) is not None and module_name.rsplit('.', 1)[-1] in _CANDIDATE_SUFFIXES


def select_zero_shot_targets(model: nn.Module, config: LoraConfig) -> Dict[str, nn.Module]:
    """Resolve materialized linear modules eligible for zero-shot allocation."""
    resolved = _maybe_include_all_linear_layers(copy.deepcopy(config), model)
    targets: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if not name or not isinstance(weight, nn.Parameter) or weight.ndim != 2:
            continue
        if not check_target_module_exists(resolved, name) or not _is_candidate_module(name):
            continue
        if not isinstance(module, nn.Linear):
            raise ValueError(f'Zero-shot LoRA target {name!r} is not an nn.Linear module.')
        if weight.is_meta:
            raise ValueError('Zero-shot spectral scoring requires materialized weights; '
                             'disable memory_efficient_init.')
        targets[name] = module
    if not targets:
        raise ValueError(f'Zero-shot LoRA found no candidate modules for {config.target_modules!r}.')
    return targets


def _spectrum_cache_path(
    cache_dir: Path,
    cache_key: str,
    module_name: str,
    shape: Tuple[int, int],
    dtype: torch.dtype,
) -> Path:
    identity = f'{cache_key}|{module_name}|{shape[0]}x{shape[1]}|{dtype}'
    digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
    return cache_dir / f'spectrum-{digest}.pt'


@torch.no_grad()
def compute_zero_shot_scores(
    model: nn.Module,
    config: LoraConfig,
    r: int,
    *,
    cache_dir: Optional[Path] = None,
    cache_key: str = '',
    epsilon: float = 1e-12,
    log_interval: int = 20,
) -> ZeroShotScores:
    """Score pretrained modules from singular-value spectra without training data."""
    targets = select_zero_shot_targets(model, config)
    names = sorted(targets)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    scores: Dict[str, float] = {}
    metrics_by_module: Dict[str, Dict[str, float]] = {}

    if rank == 0:
        if log_interval > 0:
            logger.info(f'Zero-shot spectral scoring: {len(names)} modules, LoRA rank={r}')
        for index, name in enumerate(names, start=1):
            weight = targets[name].weight.detach()
            shape = tuple(weight.shape)
            spectrum_path = None
            if cache_dir is not None:
                spectrum_path = _spectrum_cache_path(Path(cache_dir), cache_key, name, shape, weight.dtype)

            singular_values = None
            if spectrum_path is not None and spectrum_path.is_file():
                try:
                    cached = torch.load(spectrum_path, map_location='cpu', weights_only=True)
                    if isinstance(cached, torch.Tensor) and cached.ndim == 1 and cached.numel() == min(shape):
                        singular_values = cached
                except (OSError, RuntimeError, EOFError):
                    logger.warning(f'Zero-shot spectrum cache is unreadable: {spectrum_path}; recomputing')

            if singular_values is None:
                cpu_weight = weight.to(device='cpu')
                if cpu_weight.dtype not in (torch.float32, torch.float64):
                    cpu_weight = cpu_weight.float()
                singular_values = torch.linalg.svdvals(cpu_weight)
                if spectrum_path is not None:
                    spectrum_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path = spectrum_path.with_suffix('.tmp')
                    torch.save(singular_values, temporary_path)
                    os.replace(temporary_path, spectrum_path)

            metrics = compute_zero_shot_spectral_metrics(singular_values, r=r, epsilon=epsilon)
            metrics_by_module[name] = metrics
            scores[name] = metrics['score']
            if log_interval > 0 and (index % log_interval == 0 or index == len(names)):
                logger.info(f'Zero-shot spectral scoring: {index}/{len(names)} modules '
                            f'(last {name} -> score={metrics["score"]:.4f}, '
                            f'effective_rank={metrics["effective_rank"]:.1f}, '
                            f'rank_coverage={metrics["rank_coverage"]:.4f})')

    if distributed:
        payload = [(scores, metrics_by_module) if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        scores, metrics_by_module = payload[0]
    return ZeroShotScores(scores, metrics_by_module)


def allocate_zero_shot_modules(
    scores: Mapping[str, float],
    param_counts: Mapping[str, int],
    fft_ratio: float = 0.1,
) -> Tuple[List[str], List[str]]:
    """Allocate the highest-scoring prefix to full fine-tuning within a parameter budget."""
    if not 0.0 <= fft_ratio < 1.0:
        raise ValueError('Zero-shot FFT ratio must be in [0, 1).')
    missing = set(scores) - set(param_counts)
    if missing:
        raise ValueError(f'Zero-shot allocation is missing parameter counts for: {sorted(missing)}.')
    total = sum(param_counts[name] for name in scores)
    if total <= 0:
        raise ValueError('Zero-shot candidate parameter count must be positive.')

    budget = fft_ratio * total
    ordered = sorted(scores, key=lambda name: (-scores[name], name))
    s_fft: List[str] = []
    used = 0
    for name in ordered:
        cost = param_counts[name]
        if used + cost > budget + 1e-9:
            break
        s_fft.append(name)
        used += cost
    s_fft_set = set(s_fft)
    s_lora = [name for name in ordered if name not in s_fft_set]
    return sorted(s_fft), sorted(s_lora)


def build_zero_shot_lora_config(
    s_lora: List[str],
    s_fft: List[str],
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    **kwargs,
) -> LoraConfig:
    if not s_lora:
        raise ValueError('Zero-shot LoRA requires at least one LoRA module.')
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(s_lora),
        modules_to_save=list(s_fft),
        **kwargs,
    )


def build_zero_shot_param_groups(
    peft_model: nn.Module,
    lr_lora: float = 2.5e-5,
    lr_fft: float = 1e-6,
    weight_decay: float = 0.0,
    adapter_name: str = 'default',
) -> List[dict]:
    """Build separate optimizer groups for LoRA and fully fine-tuned modules."""
    lora_params, lora_names = [], []
    fft_params, fft_names = [], []
    adapter_token = f'.{adapter_name}.'
    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        if '.lora_' in name and adapter_token in name:
            lora_params.append(param)
            lora_names.append(name)
        elif '.modules_to_save.' in name and adapter_token in name:
            fft_params.append(param)
            fft_names.append(name)
        else:
            raise ValueError(f'Zero-shot LoRA cannot classify trainable parameter {name!r}.')

    groups: List[dict] = []
    if lora_params:
        groups.append({'params': lora_params, 'param_names': lora_names, 'lr': lr_lora, 'weight_decay': weight_decay})
    if fft_params:
        groups.append({'params': fft_params, 'param_names': fft_names, 'lr': lr_fft, 'weight_decay': weight_decay})
    if not groups:
        raise ValueError('Zero-shot LoRA found no trainable parameters to optimize.')
    return groups
