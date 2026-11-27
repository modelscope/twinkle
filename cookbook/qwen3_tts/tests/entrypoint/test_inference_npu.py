"""Opt-in real-model adapter inference smoke test using an existing checkpoint."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
class InferenceIntegration(unittest.TestCase):
    def test_existing_adapter_produces_audio(self):
        model=os.environ['MODEL_PATH']
        checkpoint=os.environ['CHECKPOINT_PATH']
        reference=ROOT/'tests/fixtures/librispeech/1272-128104-0000.flac'
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'generated.wav'
            subprocess.run([sys.executable,str(ROOT/'scripts/generate.py'),'--model_path',model,
                '--checkpoint',checkpoint,'--reference_audio',str(reference),'--text','Today is a good day.',
                '--output',str(output),'--max_new_tokens','128'],check=True,timeout=600)
            wave,sr=sf.read(output)
            self.assertGreater(sr,0)
            self.assertTrue(wave.ndim==1 and wave.size>0 and np.isfinite(wave).all())
if __name__=='__main__':unittest.main(verbosity=2)
