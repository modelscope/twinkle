import json
import os
import tempfile

from twinkle_agentic.preprocessor import merge_dropped_shards, truncate_dropped_logs


def test_merge_dropped_shards():
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, 'dropped.jsonl')
        with open(f'{base}.111', 'w', encoding='utf-8') as f:
            f.write(json.dumps({'step': 'A', 'id': '1'}) + '\n')
        with open(f'{base}.222', 'w', encoding='utf-8') as f:
            f.write(json.dumps({'step': 'B', 'id': '2'}) + '\n')
        merge_dropped_shards(base)
        with open(base, encoding='utf-8') as f:
            lines = [ln for ln in f if ln.strip()]
        assert len(lines) == 2
        assert not os.path.exists(f'{base}.111')
        truncate_dropped_logs(base)
        assert not os.path.exists(base)
