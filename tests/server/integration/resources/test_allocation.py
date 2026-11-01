import json
import subprocess
import sys
from pathlib import Path


def test_production_allocator_uses_disjoint_devices_and_reuses_released_group():
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('allocation_probe.py'))],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence['passed'] and len(set(evidence['devices'])) == 8
    assert len(evidence['reused']) == 2


def test_fragmented_devices_cannot_be_combined_across_nodes():
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('fragmentation_probe.py'))],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence['passed'] and evidence['nodes'] == 2
    assert evidence['fragmented_rejected'] and evidence['pending_removed']
    assert len(evidence['devices']) == 2


def test_sampler_uses_real_cpu_services_for_scaling_and_release():
    script = Path(__file__).resolve().parents[1] / 'sampler' / 'sampling_probe.py'
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"sizes":')]
    assert evidence == [{'sizes': [2, 4, 2], 'requests': 8, 'confirmed_release': True}]
