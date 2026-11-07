"""CPU manifest/cursor tests; no Twinkle doubles or model imports needed."""
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('manifest', ROOT / 'tts/qwen3/manifest.py')
cli = load('train_cli', ROOT / 'scripts/train_qwen3_tts_lora.py')


class ManifestContracts(unittest.TestCase):
    def test_learning_rate_decay_is_explicit_and_validated(self):
        required = ['--model_path', 'model', '--train_jsonl', 'train', '--output_dir', 'out']
        self.assertEqual(cli.parse_args(required).lr_decay, 1.0)
        self.assertEqual(cli.parse_args(required + ['--lr_decay', '0.8']).lr_decay, 0.8)
        for value in ('0', '-1', '1.1', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                cli.parse_args(required + ['--lr_decay', value])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        np.save(self.root / 'mel.npy', np.zeros((3, 128), dtype=np.float32))
        self.rows = [dict(id=str(i), text=f'text {i}', audio_codes=[[i] * 16],
                          ref_mel_path='mel.npy') for i in range(2)]

    def write(self):
        path = self.root / 'train.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in self.rows), encoding='utf-8')
        return path

    def test_fingerprint_tracks_content_not_location(self):
        first = m.Manifest(self.write())
        self.assertEqual(len(first), 2)
        self.rows[0]['ref_mel_path'] = str(self.root / 'mel.npy')
        self.assertEqual(first.fingerprint, m.Manifest(self.write()).fingerprint)
        self.rows[0]['text'] += ' changed'
        self.assertNotEqual(first.fingerprint, m.Manifest(self.write()).fingerprint)
        second = m.Manifest(self.write()).fingerprint
        np.save(self.root / 'mel.npy', np.ones((3, 128), dtype=np.float32))
        self.assertNotEqual(second, m.Manifest(self.write()).fingerprint)

    def test_rejects_empty_duplicate_and_invalid_codes(self):
        original = json.loads(json.dumps(self.rows))
        for rows in ([], [original[0], original[0]],
                     [{**original[0], 'audio_codes': [[2048] * 16]}],
                     [{**original[0], 'audio_codes': [[0.5] * 16]}],
                     [{**original[0], 'audio_codes': [[0] * 15]}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.rows = rows
                m.Manifest(self.write())

    def test_rejects_invalid_mel(self):
        for mel in (np.zeros((2, 80)), np.full((2, 128), np.nan)):
            np.save(self.root / 'mel.npy', mel)
            with self.assertRaises(ValueError):
                m.Manifest(self.write())

    def test_shuffle_is_deterministic_and_does_not_consume_rng(self):
        state = random.getstate()
        plan = m.sample_plan(5, 3, 42, True)
        self.assertEqual(state, random.getstate())
        self.assertEqual(plan, m.sample_plan(5, 3, 42, True))
        self.assertNotEqual(plan[0], plan[1])
        for order in plan:
            self.assertEqual(sorted(order), list(range(5)))
        self.assertEqual(m.sample_plan(2, 2, 42, False), [[0, 1], [0, 1]])

    def test_tail_and_resume_cursor_do_not_drop_samples(self):
        schedule = m.planned_windows(m.sample_plan(5, 2, 42, False), 2)
        self.assertEqual([len(w) for w in schedule], [2, 2, 1, 2, 2, 1])
        p = SimpleNamespace(optimizer_steps=3, micro_steps=5, consumed_samples=5, pending_micro_steps=0)
        m.validate_cursor(p, schedule)
        self.assertEqual(schedule[p.optimizer_steps:], [[0, 1], [2, 3], [4]])
        for key, value in [('micro_steps', 4), ('consumed_samples', 6),
                           ('pending_micro_steps', 1), ('optimizer_steps', 7)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_cursor(SimpleNamespace(**{**vars(p), key: value}), schedule)

    def test_cli_help_and_model_content_fingerprint(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_qwen3_tts_lora.py'),
                                 '--help'], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--resume_from', result.stdout)
        weights = self.root / 'model.safetensors'
        weights.write_bytes(b'first mock weight content')
        first = cli.model_fingerprint(self.root)
        weights.write_bytes(b'second mock weight content')
        self.assertNotEqual(first, cli.model_fingerprint(self.root))


if __name__ == '__main__':
    unittest.main(verbosity=2)
