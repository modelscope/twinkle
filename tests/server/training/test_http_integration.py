import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def test_installed_native_and_tinker_sdks_use_managed_http_routes(tmp_path):
    # This probe owns a separate Ray cluster. Stop the session fixture's cluster
    # first so two local GCS/raylet pairs do not evict each other on a slow host.
    import ray
    ray.shutdown()
    try:
        # Both SDKs, recovery and release/recreate cycles start fresh
        # workers. Allow cold imports from a shared environment; each service
        # operation still enforces its own startup/execution deadline.
        evidence_path = tmp_path / 'http-result.json'
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).resolve().parents[3]
                / 'tests/server/integration/on_demand_training/http_probe.py')
        ], env={**os.environ, 'TWINKLE_TEST_HTTP_RESULT': str(evidence_path)}, timeout=2 * 60 * 60)
    finally:
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            log_to_driver=False,
            namespace='twinkle_test',
            _temp_dir='/tmp/twinkle-test-' + uuid.uuid4().hex[:8])
    assert result.returncode == 0, 'HTTP probe failed; see subprocess output above'
    evidence = json.loads(evidence_path.read_text())
    assert evidence['passed'] and evidence['native_recreation'] and evidence['tinker_shared']
    assert len(set(evidence['native_stable_routing'].values())) == 2
    assert len(set(evidence['tinker_stable_routing'].values())) == 2
    assert evidence['native_lora']
    assert evidence['runtime_lifecycle_rejected']
    assert evidence['tinker_full']
    assert evidence['restart'] and evidence['state_loss'] and evidence['cycles'] == 3
