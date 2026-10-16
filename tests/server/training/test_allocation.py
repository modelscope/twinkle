import json
import subprocess
import sys
from pathlib import Path


def test_production_allocator_uses_disjoint_devices_and_reuses_released_group():
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / 'tools/validation/on_demand_training/allocation_probe.py')],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence['passed'] and len(set(evidence['devices'])) == 8
    assert len(evidence['reused']) == 2


def test_fragmented_devices_cannot_be_combined_across_nodes():
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / 'tools/validation/on_demand_training/fragmentation_probe.py')],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence['passed'] and evidence['nodes'] == 2
    assert evidence['fragmented_rejected'] and evidence['pending_removed']
    assert len(evidence['devices']) == 2
