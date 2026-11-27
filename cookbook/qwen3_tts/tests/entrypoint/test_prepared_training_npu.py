"""Actual preparation CLI -> validated manifests -> one real NPU training update."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch_npu  # register the actual remote NPU runtime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tts.qwen3.manifest import Manifest


class PreparedTraining(unittest.TestCase):
    def test_public_split_preparation_and_train_only_consumption(self):
        outputs = ROOT / 'outputs'
        outputs.mkdir(exist_ok=True)
        model = os.environ.get('MODEL_PATH', '/models/Qwen3-TTS-12Hz-1.7B-Base')
        with tempfile.TemporaryDirectory(prefix='prepared-training-', dir=outputs) as folder:
            root = Path(folder)
            ids, source_hashes = {}, {}
            for split, fixture in [('train', ROOT/'tests/fixtures/librispeech'),
                                   ('dev', ROOT/'tests/fixtures/librispeech/second')]:
                meta = json.loads((fixture/'sample.json').read_text())
                audio = fixture/meta['audio']
                self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), meta['sha256'])
                source_hashes[audio] = meta['sha256']
                ids[split] = meta['id']
                # Explicit fixture-only split, not a real held-out evaluation claim.
                row = dict(id=meta['id'], text=meta['text'], audio=str(audio),
                           group_id=meta['id'], speaker_id='1272')
                (root/f'{split}.jsonl').write_text(json.dumps(row)+'\n', encoding='utf-8')
            derived = root/'derived'
            subprocess.run([sys.executable, '-u', str(ROOT/'scripts/prepare_split_manifest.py'),
                '--train_jsonl', str(root/'train.jsonl'), '--dev_jsonl', str(root/'dev.jsonl'),
                '--reference_id', ids['train'], '--model_path', model, '--output_dir', str(derived)],
                check=True, timeout=600)
            for split in ('train', 'dev'):
                data = Manifest(derived/f'{split}.jsonl')
                self.assertEqual(len(data), 1)
                self.assertEqual(data.rows[0]['id'], ids[split])
            provenance = json.loads((derived/'provenance.json').read_text())
            self.assertEqual(provenance['counts'], {'train': 1, 'dev': 1})
            self.assertTrue(all(r['reference_id'] == ids['train'] for r in provenance['mapping']))
            output = root/'training'
            subprocess.run([sys.executable, '-u', str(ROOT/'scripts/train_qwen3_tts_lora.py'),
                '--model_path', model, '--train_jsonl', str(derived/'train.jsonl'),
                '--output_dir', str(output), '--epochs', '1', '--grad_accum', '2'],
                check=True, timeout=600)
            trace = [json.loads(line) for line in (output/'trace.jsonl').read_text().splitlines()]
            self.assertEqual(len(trace), 1)
            self.assertEqual(trace[0]['sample_ids'], [ids['train']])
            result = json.loads((output/'result.json').read_text())
            self.assertEqual((result['micro_steps'], result['optimizer_steps']), (1, 1))
            self.assertTrue(result['complete'])
            self.assertTrue((Path(result['checkpoint'])/'adapter_model.safetensors').is_file())
            for audio, digest in source_hashes.items():
                self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), digest)
            print(json.dumps(dict(stage='预编码训练联通通过 / Prepared training passed',
                counts=provenance['counts'], reference_id=ids['train'], consumed_ids=trace[0]['sample_ids'],
                dev_consumed=False, loss=trace[0]['losses'], visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                scope='public regression fixtures only; no held-out quality evaluation'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
