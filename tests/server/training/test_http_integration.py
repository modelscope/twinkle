import json
import subprocess
import sys
from pathlib import Path


def test_installed_native_and_tinker_sdks_use_managed_http_routes():
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / 'tools/validation/on_demand_training/http_probe.py')],
                            capture_output=True, text=True, timeout=420)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"passed":'))
    assert evidence['passed'] and evidence['native_recreation'] and evidence['tinker_shared']
    assert evidence['native_lora']
    assert evidence['runtime_lifecycle_rejected']
    assert evidence['tinker_full']
    assert evidence['restart'] and evidence['state_loss'] and evidence['cycles'] == 10
