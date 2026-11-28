# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import json
import subprocess
import sys


def _run_import_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, '-c', source],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_import_twinkle_does_not_eagerly_import_client_or_torch() -> None:
    result = _run_import_probe(
        "import json, sys, twinkle; "
        "print(json.dumps({'client': 'twinkle_client' in sys.modules, 'torch': 'torch' in sys.modules}))")
    assert result == {'client': False, 'torch': False}


def test_import_twinkle_client_does_not_load_heavy_data_dependencies() -> None:
    result = _run_import_probe(
        "import json, sys, twinkle_client; "
        "print(json.dumps({name: name in sys.modules for name in ('torch', 'datasets', 'pandas')}))")
    assert result == {'torch': False, 'datasets': False, 'pandas': False}


def test_multi_turn_import_does_not_start_ray_or_load_optional_sandbox_engine() -> None:
    # The client multi-turn surface is HTTPSampler + ClientCheckpointManager driving the local
    # MultiTurnRollout (ClientMultiTurnRollout was folded into that design); importing it must
    # stay lightweight.
    result = _run_import_probe(
        "import json, sys; from twinkle_client.rollout import HTTPSampler, ClientCheckpointManager; "
        "from twinkle_agentic.rollout import MultiTurnRollout; "
        "print(json.dumps({name: name in sys.modules for name in ('ray', 'e2b', 'vllm.engine', 'transfer_queue')}))")
    assert result == {'ray': False, 'e2b': False, 'vllm.engine': False, 'transfer_queue': False}
