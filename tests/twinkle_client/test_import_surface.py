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


def test_twinkle_client_entry_points_remain_available_without_eager_import() -> None:
    result = _run_import_probe(
        "import json, sys, twinkle; "
        "before = 'twinkle_client' in sys.modules; "
        "from twinkle import init_tinker_client, init_twinkle_client; "
        "after = 'twinkle_client' in sys.modules; "
        "print(json.dumps({'before': before, 'after': after, "
        "'tinker': callable(init_tinker_client), 'twinkle': callable(init_twinkle_client)}))")
    assert result == {'before': False, 'after': False, 'tinker': True, 'twinkle': True}


def test_twinkle_client_entry_points_delegate(monkeypatch) -> None:
    import twinkle
    import twinkle_client

    calls = []
    monkeypatch.setattr(twinkle_client, 'init_tinker_client', lambda **kwargs: calls.append(('tinker', kwargs)))
    monkeypatch.setattr(twinkle_client, 'init_twinkle_client', lambda **kwargs: ('twinkle', kwargs))

    assert twinkle.init_tinker_client(feature=True) is None
    assert calls == [('tinker', {'feature': True})]
    assert twinkle.init_twinkle_client(base_url='http://server', api_key='key') == (
        'twinkle', {
            'base_url': 'http://server',
            'api_key': 'key',
            'session_heartbeat_interval': 10,
        })


def test_import_twinkle_client_does_not_load_heavy_data_dependencies() -> None:
    result = _run_import_probe(
        "import json, sys, twinkle_client; "
        "print(json.dumps({name: name in sys.modules for name in ('torch', 'datasets', 'pandas')}))")
    assert result == {'torch': False, 'datasets': False, 'pandas': False}
