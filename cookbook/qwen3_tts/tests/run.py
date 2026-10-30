"""Run each contract module in a fresh process; NPU tests are explicit opt-in."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
group = parser.add_mutually_exclusive_group()
group.add_argument('--isolated', action='store_true', help='CPU development with explicit Twinkle doubles')
group.add_argument('--npu', action='store_true', help='Also run real local-model integrations; set MODEL_PATH')
args = parser.parse_args()
if args.npu and not os.environ.get('MODEL_PATH'):
    parser.error('--npu requires MODEL_PATH pointing to an existing local model')
modules = ['batch1/test_contracts.py', 'lora/test_contracts.py', 'training/test_contracts.py',
           'entrypoint/test_contracts.py', 'entrypoint/test_preparation.py']
if args.npu:
    modules += ['lora/test_real_model.py', 'entrypoint/test_prepared_training_npu.py',
                'entrypoint/test_process_resume_npu.py']
for module in modules:
    extra = ['--isolated-twinkle'] if args.isolated and module.startswith(('batch1/', 'lora/')) else []
    print('Running: ' + module, flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT/'tests'/module), *extra], check=True, timeout=3600)
