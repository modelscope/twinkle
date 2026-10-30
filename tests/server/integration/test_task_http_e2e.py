# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in task HTTP acceptance. Run with --noconftest on reserved devices.

Set TWINKLE_TEST_TASK_CS=1 and TWINKLE_TEST_TASK_MODEL_DIR to the prepared bases.
TWINKLE_TEST_BACKEND selects transformers or megatron; run once for each backend.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TWINKLE_TEST_TASK_CS') != '1',
                                reason='Set TWINKLE_TEST_TASK_CS=1 on reserved devices')


def test_task_cli_isolation_checkpoint_and_release(tmp_path):
    import ray
    assert not ray.is_initialized(), 'Use --noconftest; this test owns an isolated Ray cluster'
    backend = os.environ.get('TWINKLE_TEST_BACKEND', 'transformers')
    directory = Path(os.environ.get('TWINKLE_TEST_ARTIFACT_DIR', str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('task_http_validation.py')
    root = Path(__file__).resolve().parents[3]
    command = [sys.executable, '-u', str(script), '--backend', backend,
               '--task', 'embedding', 'reranker', 'seq_cls',
               '--model-dir', os.environ['TWINKLE_TEST_TASK_MODEL_DIR'], '--run-dir', str(directory),
               '--steps', '31', '--min-loss-drop', '0.1']
    env = {**os.environ, 'PYTHONPATH': str(root / 'src') + os.pathsep + str(root)}
    with (directory / 'server.log').open('w') as output:
        completed = subprocess.run(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=5400)
    assert completed.returncode == 0, (directory / 'server.log').read_text()[-6000:]
    for task in ['embedding', 'reranker', 'seq_cls']:
        result = json.loads((directory / (backend + '-' + task) / 'results.json').read_text())
        assert result['passed'] and result['slot_reuse']
        assert result['loss_drop'] >= 0.1
