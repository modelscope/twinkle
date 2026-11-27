"""Split preservation and provenance; codec/mel callbacks are explicit doubles."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_split', ROOT / 'scripts/prepare_split_manifest.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class Preparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for i in range(2):
            sf.write(self.root / f'{i}.wav', np.sin(np.arange(2400) * (i + 1) / 20) * .1, 24000)
        self.rows = [dict(id=str(i), text=f'Text {i}', group_id=str(i), speaker_id='speaker', audio=f'{i}.wav') for i in range(2)]
        self.write()

    def write(self):
        for name, row in zip(('train', 'dev'), self.rows):
            (self.root / f'{name}.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')

    def audit(self, reference='0'):
        return prep.audit_split(self.root / 'train.jsonl', self.root / 'dev.jsonl', reference)

    def test_preserves_text_split_and_train_reference(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        out = self.root / 'derived'
        report = prep.prepare(self.root/'train.jsonl', self.root/'dev.jsonl', out, '0',
            lambda _: np.zeros((2, 16), dtype=np.int64), lambda _: np.zeros((3, 128), dtype=np.float32))
        self.assertEqual(report['counts'], {'train': 1, 'dev': 1})
        for i, split in enumerate(('train', 'dev')):
            row = json.loads((out / f'{split}.jsonl').read_text())
            self.assertEqual(row['text'], self.rows[i]['text'])
            self.assertEqual(row['id'], str(i))
            self.assertTrue((out / row['ref_mel_path']).is_file())
            self.assertEqual(report['mapping'][i]['reference_id'], '0')
        for name, data in before.items():
            self.assertEqual((self.root/name).read_bytes(), data)
        with self.assertRaises(FileExistsError):
            prep.prepare(self.root/'train.jsonl', self.root/'dev.jsonl', out, '0', None, None)

    def test_rejects_overlap_for_each_identity(self):
        original = dict(self.rows[1])
        for key in ('id', 'group_id', 'text', 'audio'):
            self.rows[1] = {**original, key: self.rows[0][key]}
            self.write()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit()

    def test_normalized_text_overlap_is_rejected_without_rewriting(self):
        self.rows[1]['text'] = '  TEXT   0  '
        self.write()
        with self.assertRaises(ValueError):
            self.audit()

    def test_dev_reference_and_empty_dev_are_rejected(self):
        with self.assertRaises(ValueError):
            self.audit('1')
        (self.root/'dev.jsonl').write_text('')
        with self.assertRaises(ValueError):
            self.audit()

    def test_reencoded_duplicate_audio_is_rejected(self):
        wave, sr = sf.read(self.root/'0.wav', dtype='float32')
        sf.write(self.root/'1.wav', wave, sr, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'pcm'):
            self.audit()

    def test_stereo_and_multiple_speakers_are_rejected(self):
        self.rows[1]['speaker_id'] = 'other'
        self.write()
        with self.assertRaises(ValueError):
            self.audit()
        sf.write(self.root/'0.wav', np.zeros((100, 2)), 24000)
        with self.assertRaises(ValueError):
            prep.read_audio(self.root/'0.wav')


if __name__ == '__main__':
    unittest.main(verbosity=2)
