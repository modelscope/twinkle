# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in single-node GPU matrix using the real Redis state backend."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E') != '1' or not os.environ.get('TWINKLE_TEST_REDIS_URL'),
    reason='Set TWINKLE_TEST_GPU_E2E=1 and TWINKLE_TEST_REDIS_URL for the GPU Redis matrix',
)


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
def test_on_demand_training_gpu_redis_matrix(tmp_path, backend):
    model = os.environ.get('TWINKLE_TEST_MODEL') or os.environ.get('TWINKLE_DEVICE_MODEL')
    if not model:
        pytest.skip('Set TWINKLE_TEST_MODEL or TWINKLE_DEVICE_MODEL')
    root = Path(__file__).resolve().parents[3]
    run_dir = tmp_path / backend
    script = Path(__file__).with_name('on_demand_training') / 'matrix_validation.py'
    result = subprocess.run([
        sys.executable,
        str(script),
        '--backend',
        backend,
        '--model',
        model,
        '--run-dir',
        str(run_dir),
        '--persistence',
        'redis',
        '--startup-timeout-s',
        '1800',
        '--shutdown-timeout-s',
        '180',
    ],
                            cwd=root,
                            capture_output=True,
                            text=True,
                            timeout=4 * 60 * 60)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((run_dir / 'results.json').read_text())
    assert evidence['passed']
    assert evidence['persistence'] == 'redis'
    assert evidence['lazy_start'] and evidence['training_resources_released']
    assert evidence['restart_epoch'] and evidence['worker_fault_recovered']
    assert all(row['checkpoint_restored'] for row in evidence['recovery'])
    assert evidence['capacity_checked'] and evidence['idempotency_checked']
    assert all(case['returncode'] == 0 and not case['skipped'] for case in evidence['cases'])
