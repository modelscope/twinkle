# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed.tensor  # noqa: F401

from twinkle import Platform

MODEL_ID = os.environ.get('TWINKLE_MM_COMPARE_MODEL_ID')
DATASET_ID = os.environ.get('TWINKLE_MM_COMPARE_DATASET')
SEED = os.environ.get('TWINKLE_MM_COMPARE_SEED', '1234')
STEP = os.environ.get('TWINKLE_MM_COMPARE_STEP', '0')
BATCH_SIZE = os.environ.get('TWINKLE_MM_COMPARE_BATCH_SIZE', '4')
MAX_LENGTH = os.environ.get('TWINKLE_MM_COMPARE_MAX_LENGTH', '1024')
LOSS_ATOL = float(os.environ.get('TWINKLE_MM_COMPARE_LOSS_ATOL', '1e-2'))
GRAD_MAX_ATOL = float(os.environ.get('TWINKLE_MM_COMPARE_GRAD_MAX_ATOL', '5e-2'))
GRAD_MEAN_ATOL = float(os.environ.get('TWINKLE_MM_COMPARE_GRAD_MEAN_ATOL', '1e-2'))
LAYER0_MEAN_ATOL = float(os.environ.get('TWINKLE_MM_COMPARE_LAYER0_MEAN_ATOL', '1e-3'))
LAYER0_STD_ATOL = float(os.environ.get('TWINKLE_MM_COMPARE_LAYER0_STD_ATOL', '1e-3'))
LAYER0_ABS_MAX_ATOL = float(os.environ.get('TWINKLE_MM_COMPARE_LAYER0_ABS_MAX_ATOL', '1e-2'))
COMPARE_PARAM_FILTERS = os.environ.get('TWINKLE_MM_COMPARE_PARAMS', '')
REPO_ROOT = Path(__file__).resolve().parents[2]
DUMP_SCRIPT = REPO_ROOT / 'cookbook' / 'mm' / 'fsdp2_dump_step_stats.py'


def _device_count() -> int:
    device_type = Platform.get_platform().device_prefix()
    if device_type == 'cuda':
        return int(torch.cuda.device_count())
    if device_type == 'npu' and hasattr(torch, 'npu'):
        return int(torch.npu.device_count())
    return 0


def _load_snapshot(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def _run_dump(output_path: Path, ulysses_size: int | None) -> dict:
    env = os.environ.copy()
    env.update({
        'MODEL_ID': MODEL_ID,
        'DATASETS': DATASET_ID,
        'TWINKLE_COMPARE_SEED': SEED,
        'TWINKLE_COMPARE_STEP': STEP,
        'TWINKLE_COMPARE_BATCH_SIZE': BATCH_SIZE,
        'TWINKLE_COMPARE_MAX_LENGTH': MAX_LENGTH,
        'TWINKLE_COMPARE_OUTPUT': str(output_path),
        'PYTHONHASHSEED': SEED,
    })
    if COMPARE_PARAM_FILTERS:
        env['TWINKLE_COMPARE_PARAMS'] = COMPARE_PARAM_FILTERS
    if ulysses_size is None:
        env.pop('ULYSSES_SIZE', None)
    else:
        env['ULYSSES_SIZE'] = str(ulysses_size)
    subprocess.run(
        [sys.executable, str(DUMP_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    return _load_snapshot(output_path)


def _compare_gradients(base: dict[str, torch.Tensor], other: dict[str, torch.Tensor]) -> tuple[str, float, float]:
    if base.keys() != other.keys():
        missing = sorted(set(base.keys()) ^ set(other.keys()))
        raise AssertionError(f'Compared parameter sets differ: {missing[:10]}')
    worst_name = ''
    worst_max_diff = 0.0
    worst_mean_diff = 0.0
    for name in sorted(base.keys()):
        diff = (base[name].float() - other[name].float()).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        if max_diff > worst_max_diff:
            worst_name = name
            worst_max_diff = max_diff
            worst_mean_diff = mean_diff
    return worst_name, worst_max_diff, worst_mean_diff


def _assert_batch_summaries_match(base: dict, other: dict) -> None:
    assert base.keys() == other.keys(), f'batch summary keys differ: {base.keys()} vs {other.keys()}'
    for key in sorted(base.keys()):
        assert base[key] == other[key], f'batch summary mismatch for {key}: base={base[key]} other={other[key]}'


def _assert_layer0_summary_aligned(base: dict | None, other: dict | None) -> None:
    assert base is not None and other is not None, 'missing layer0 linear attention summary'
    assert base.keys() == other.keys(), f'layer0 summary keys differ: {base.keys()} vs {other.keys()}'
    for side in ('input', 'output'):
        assert side in base and side in other, f'missing layer0 {side} summary'
        assert base[side]['shape'] == other[side]['shape'], (
            f'layer0 {side} shape mismatch: base={base[side]["shape"]} other={other[side]["shape"]}'
        )
        mean_diff = abs(float(base[side]['mean']) - float(other[side]['mean']))
        std_diff = abs(float(base[side]['std']) - float(other[side]['std']))
        abs_max_diff = abs(float(base[side]['abs_max']) - float(other[side]['abs_max']))
        assert mean_diff <= LAYER0_MEAN_ATOL, (
            f'layer0 {side} mean mismatch: base={base[side]["mean"]:.8f} '
            f'other={other[side]["mean"]:.8f} diff={mean_diff:.8f}'
        )
        assert std_diff <= LAYER0_STD_ATOL, (
            f'layer0 {side} std mismatch: base={base[side]["std"]:.8f} '
            f'other={other[side]["std"]:.8f} diff={std_diff:.8f}'
        )
        assert abs_max_diff <= LAYER0_ABS_MAX_ATOL, (
            f'layer0 {side} abs_max mismatch: base={base[side]["abs_max"]:.8f} '
            f'other={other[side]["abs_max"]:.8f} diff={abs_max_diff:.8f}'
        )


@pytest.mark.skipif(not MODEL_ID, reason='Set TWINKLE_MM_COMPARE_MODEL_ID to a local or accessible Qwen3.5 model id')
@pytest.mark.skipif(not DATASET_ID, reason='Set TWINKLE_MM_COMPARE_DATASET to a local or accessible multimodal dataset')
@pytest.mark.skipif(_device_count() < 2, reason='requires at least 2 accelerator devices')
def test_qwen35_multimodal_ulysses_loss_and_grad_alignment():
    with tempfile.TemporaryDirectory(prefix='twinkle-mm-ulysses-') as tmpdir:
        tmpdir = Path(tmpdir)
        base_path = tmpdir / 'baseline.pt'
        ulysses_path = tmpdir / 'ulysses2.pt'

        base_snapshot = _run_dump(base_path, ulysses_size=None)
        ulysses_snapshot = _run_dump(ulysses_path, ulysses_size=2)

        assert base_snapshot['step'] == ulysses_snapshot['step'] == int(STEP)

        loss_diff = abs(float(base_snapshot['loss']) - float(ulysses_snapshot['loss']))
        assert loss_diff <= LOSS_ATOL, (
            f'ulysses loss mismatch: base={base_snapshot["loss"]:.8f} '
            f'ulysses2={ulysses_snapshot["loss"]:.8f} diff={loss_diff:.8f}'
        )

        worst_name, worst_max_diff, worst_mean_diff = _compare_gradients(
            base_snapshot['gradients'],
            ulysses_snapshot['gradients'],
        )
        assert worst_max_diff <= GRAD_MAX_ATOL, (
            f'ulysses grad max mismatch: param={worst_name} '
            f'max_diff={worst_max_diff:.8f} mean_diff={worst_mean_diff:.8f}'
        )
        assert worst_mean_diff <= GRAD_MEAN_ATOL, (
            f'ulysses grad mean mismatch: param={worst_name} '
            f'max_diff={worst_max_diff:.8f} mean_diff={worst_mean_diff:.8f}'
        )


@pytest.mark.skipif(not MODEL_ID, reason='Set TWINKLE_MM_COMPARE_MODEL_ID to a local or accessible Qwen3.5 model id')
@pytest.mark.skipif(not DATASET_ID, reason='Set TWINKLE_MM_COMPARE_DATASET to a local or accessible multimodal dataset')
@pytest.mark.skipif(_device_count() < 2, reason='requires at least 2 accelerator devices')
def test_qwen35_multimodal_ulysses_batch_and_layer0_alignment():
    with tempfile.TemporaryDirectory(prefix='twinkle-mm-ulysses-layer0-') as tmpdir:
        tmpdir = Path(tmpdir)
        base_snapshot = _run_dump(tmpdir / 'baseline.pt', ulysses_size=None)
        ulysses_snapshot = _run_dump(tmpdir / 'ulysses2.pt', ulysses_size=2)

        _assert_batch_summaries_match(
            base_snapshot['batch_summary'],
            ulysses_snapshot['batch_summary'],
        )
        _assert_layer0_summary_aligned(
            base_snapshot.get('layer0_linear_attn_summary'),
            ulysses_snapshot.get('layer0_linear_attn_summary'),
        )
