# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in ST HTTP acceptance on two reserved devices (CS=0, bare reference=1)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TWINKLE_TEST_ST_CS') != '1',
                                reason='Set TWINKLE_TEST_ST_CS=1 on two reserved devices')


def test_st_http_training_isolation_and_resume(tmp_path):
    model = os.environ['TWINKLE_TEST_ST_MODEL']
    directory = Path(os.environ.get('TWINKLE_TEST_ARTIFACT_DIR', str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).with_name('validation.py')),
               '--model', model, '--run-dir', str(directory)]
    root = Path(__file__).resolve().parents[4]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, (
        str(root / 'src'), str(root), os.environ.get('PYTHONPATH'))))}
    with (directory / 'server-and-client.log').open('w') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env, timeout=1800)
    assert result.returncode == 0, (directory / 'server-and-client.log').read_text()[-6000:]
    evidence = json.loads((directory / 'results.json').read_text())
    assert evidence['passed'] and evidence['slot_reuse']
    assert evidence['bare_parameter_error'] < 2e-5 and evidence['embedding_delta'] > 1e-6
    assert evidence['cold_progress'] == evidence['progress']
