"""CPU checks for the minimal inference CLI and safe waveform output."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('generation', ROOT/'scripts/generate.py')
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)

class GenerationContracts(unittest.TestCase):
    def test_base_and_adapter_arguments(self):
        base=['--model_path','model','--reference_audio','ref.wav','--text','Hello.','--output','out.wav']
        self.assertIsNone(generation.parse_args(base).checkpoint)
        self.assertEqual(generation.parse_args(base+['--checkpoint','adapter']).checkpoint,'adapter')
        for extra in [['--text',' '],['--max_new_tokens','1'],['--output','bad.mp3']]:
            with self.assertRaises(SystemExit):generation.parse_args(base+extra)
    def test_preserve_waveform_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'out.wav'
            wave=np.array([0,.2,-1.2,1.3],dtype=np.float32)
            generation.write_wave(path,wave,24000)
            actual,sr=sf.read(path,dtype='float32')
            np.testing.assert_array_equal(actual,wave)
            self.assertEqual(sr,24000)
            with self.assertRaises(FileExistsError):generation.write_wave(path,wave,24000)
    def test_invalid_waveforms_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for wave in [[],[0,0],[np.nan],[np.inf],[[1,2]]]:
                with self.assertRaises(ValueError):generation.write_wave(Path(folder)/'out.wav',wave,24000)
    def test_help_does_not_require_npu(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/generate.py'),'--help'],capture_output=True,text=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--checkpoint',r.stdout)
if __name__=='__main__':unittest.main(verbosity=2)
