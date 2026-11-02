import copy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
from twinkle.model.transformers.transformers import TransformersModel


def make_wrapper():
    model = torch.nn.Linear(4, 2).to('npu:0')
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.8,
    )
    wrapper = object.__new__(TransformersModel)
    wrapper.__dict__['model'] = model
    wrapper.strategy = NativeFSDPStrategy(
        device_mesh=None, enable_ep=False,
    )
    wrapper.optimizer_group = {
        '': SimpleNamespace(
            optimizer=optimizer,
            lr_scheduler=scheduler,
            scaler=None,
            cur_step=0,
            gradient_accumulation_steps=1,
        ),
    }
    return wrapper


def update(wrapper):
    config = wrapper.optimizer_group['']
    x = torch.arange(
        8, dtype=torch.float32, device='npu:0',
    ).reshape(2, 4) / 10
    target = torch.ones(2, 2, device='npu:0')

    config.optimizer.zero_grad(set_to_none=True)
    loss = (wrapper.model(x) - target).square().mean()
    loss.backward()
    config.optimizer.step()
    config.lr_scheduler.step()
    config.cur_step += 1
    return loss.detach().cpu()


def assert_optimizer_equal(left, right):
    assert left['param_groups'] == right['param_groups']
    assert left['state'].keys() == right['state'].keys()
    for key in left['state']:
        for name, value in left['state'][key].items():
            other = right['state'][key][name]
            if torch.is_tensor(value):
                torch.testing.assert_close(
                    value.cpu(), other.cpu(), rtol=1e-6, atol=1e-7,
                )
            else:
                assert value == other


def test_npu_training_continues_from_checkpoint(tmp_path):
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('NPU unavailable')
    torch.npu.set_device(0)

    baseline = make_wrapper()
    initial = copy.deepcopy(baseline.model.state_dict())
    for _ in range(4):
        baseline_loss = update(baseline)

    interrupted = make_wrapper()
    interrupted.model.load_state_dict(initial)
    for _ in range(2):
        update(interrupted)

    save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in interrupted.model.state_dict().items()
        },
        str(tmp_path / 'model.safetensors'),
    )
    interrupted._save_training_state(
        str(tmp_path), consumed_train_samples=8,
    )
    assert (tmp_path / 'optimizer.pt').is_file()
    assert (tmp_path / 'scheduler.pt').is_file()
    assert (tmp_path / 'rng_state_rank_0.pt').is_file()

    resumed = make_wrapper()
    progress = resumed.resume_from_checkpoint(str(tmp_path))
    assert progress['cur_step'] == 2
    assert progress['consumed_train_samples'] == 8

    saved_config = interrupted.optimizer_group['']
    resumed_config = resumed.optimizer_group['']
    assert resumed_config.cur_step == 2
    assert resumed_config.optimizer.param_groups[0]['lr'] == (
        saved_config.optimizer.param_groups[0]['lr']
    )
    assert resumed_config.lr_scheduler.state_dict() == (
        saved_config.lr_scheduler.state_dict()
    )
    assert_optimizer_equal(
        saved_config.optimizer.state_dict(),
        resumed_config.optimizer.state_dict(),
    )
    for name, tensor in interrupted.model.state_dict().items():
        torch.testing.assert_close(
            tensor.cpu(), resumed.model.state_dict()[name].cpu(),
            rtol=0, atol=0,
        )

    for _ in range(2):
        resumed_loss = update(resumed)

    assert resumed_config.cur_step == 4
    torch.testing.assert_close(
        baseline_loss, resumed_loss, rtol=1e-6, atol=1e-7,
    )
    for name, tensor in baseline.model.state_dict().items():
        torch.testing.assert_close(
            tensor.cpu(), resumed.model.state_dict()[name].cpu(),
            rtol=1e-6, atol=1e-7,
        )
    assert baseline.optimizer_group[''].lr_scheduler.state_dict() == (
        resumed_config.lr_scheduler.state_dict()
    )
    assert_optimizer_equal(
        baseline.optimizer_group[''].optimizer.state_dict(),
        resumed_config.optimizer.state_dict(),
    )


def test_npu_resume_in_fresh_process(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('NPU unavailable')

    worker = r"""
import os
import runpy
import sys
from pathlib import Path

import torch
import torch_npu
from safetensors.torch import save_file

source, mode, folder = sys.argv[1:]
folder = Path(folder)
folder.mkdir(parents=True, exist_ok=True)
torch.npu.set_device(0)

definitions = runpy.run_path(source)
make_wrapper = definitions['make_wrapper']
update = definitions['update']

torch.manual_seed(2026)
torch.npu.manual_seed(2026)
wrapper = make_wrapper()

if mode == 'resume':
    progress = wrapper.resume_from_checkpoint(str(folder / 'checkpoint'))
    assert progress['cur_step'] == 2
    assert progress['consumed_train_samples'] == 8

for _ in range(4 if mode == 'baseline' else 2):
    loss = update(wrapper)

if mode == 'save':
    checkpoint_dir = folder / 'checkpoint'
    checkpoint_dir.mkdir()
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in wrapper.model.state_dict().items()
        },
        str(checkpoint_dir / 'model.safetensors'),
    )
    wrapper._save_training_state(
        str(checkpoint_dir), consumed_train_samples=8,
    )
else:
    def cpu_tree(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_tree(item) for item in value)
        return value

    config = wrapper.optimizer_group['']
    torch.save(
        {
            'model': cpu_tree(wrapper.model.state_dict()),
            'optimizer': cpu_tree(config.optimizer.state_dict()),
            'scheduler': config.lr_scheduler.state_dict(),
            'step': config.cur_step,
            'loss': loss,
        },
        folder / f'{mode}.pt',
    )
print(f'PASS: mode={mode}, pid={os.getpid()}, step={wrapper.optimizer_group[""].cur_step}')
"""

    source = str(Path(__file__).resolve())
    root = Path(__file__).resolve().parents[2]
    for mode in ('baseline', 'save', 'resume'):
        subprocess.run(
            [sys.executable, '-c', worker, source, mode, str(tmp_path)],
            cwd=root,
            check=True,
            timeout=180,
        )

    baseline = torch.load(
        tmp_path / 'baseline.pt', map_location='cpu', weights_only=True,
    )
    resumed = torch.load(
        tmp_path / 'resume.pt', map_location='cpu', weights_only=True,
    )
    assert baseline['step'] == resumed['step'] == 4
    assert baseline['scheduler'] == resumed['scheduler']
    torch.testing.assert_close(
        baseline['loss'], resumed['loss'], rtol=1e-6, atol=1e-7,
    )
    for name, value in baseline['model'].items():
        torch.testing.assert_close(
            value, resumed['model'][name], rtol=1e-6, atol=1e-7,
        )
    assert_optimizer_equal(baseline['optimizer'], resumed['optimizer'])


def test_two_rank_npu_rng_restart(tmp_path):
    import subprocess
    import sys

    pytest.importorskip('torch_npu')
    if torch.npu.device_count() < 2:
        pytest.skip('Two NPUs required')
    if not torch.distributed.is_gloo_available():
        pytest.skip('Gloo unavailable')

    worker = tmp_path / 'rank_rng_worker.py'
    worker.write_text(r"""
import json
import os
import random
import sys
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from twinkle.model.transformers.transformers import TransformersModel

mode, folder = sys.argv[1:]
folder = Path(folder)
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.npu.set_device(local_rank)
dist.init_process_group('gloo', timeout=timedelta(seconds=90))

def draw():
    return {
        'python': random.random(),
        'numpy': float(np.random.random()),
        'cpu': torch.rand(8).tolist(),
        'npu': torch.rand(8, device=f'npu:{local_rank}').cpu().tolist(),
    }

config = SimpleNamespace(
    cur_step=2, gradient_accumulation_steps=1, scaler=None,
)
wrapper = SimpleNamespace(
    optimizer_group={'': config},
    _save_optimizer=lambda *a, **k: None,
    _load_optimizer=lambda *a, **k: None,
    _get_default_group=lambda: '',
)
wrapper._get_training_rng_state = partial(
    TransformersModel._get_training_rng_state, wrapper,
)
wrapper._load_rng_state = partial(
    TransformersModel._load_rng_state, wrapper,
)

try:
    if mode == 'save':
        random.seed(100 + rank)
        np.random.seed(100 + rank)
        torch.manual_seed(100 + rank)
        torch.npu.manual_seed(100 + rank)

        TransformersModel._save_training_state(
            wrapper, str(folder), consumed_train_samples=8,
        )
        expected = draw()
        (folder / f'expected_{rank}.json').write_text(json.dumps(expected))
    else:
        # Deliberately perturb every stream before restoring.
        draw()
        TransformersModel._restore_training_state(wrapper, str(folder))
        expected = json.loads(
            (folder / f'expected_{rank}.json').read_text()
        )
        assert draw() == expected, f'RNG mismatch on rank {rank}'

    dist.barrier()
    print(f'PASS: mode={mode}, rank={rank}, npu={local_rank}', flush=True)
finally:
    dist.destroy_process_group()
""")

    root = __import__('pathlib').Path(__file__).resolve().parents[2]
    for mode in ('save', 'resume'):
        subprocess.run(
            [
                sys.executable, '-m', 'torch.distributed.run',
                '--standalone', '--nnodes=1', '--nproc-per-node=2',
                str(worker), mode, str(tmp_path),
            ],
            cwd=root,
            check=True,
            timeout=240,
        )

    import json
    for rank in (0, 1):
        assert (tmp_path / f'rng_state_rank_{rank}.pt').is_file()

    first = json.loads((tmp_path / 'expected_0.json').read_text())
    second = json.loads((tmp_path / 'expected_1.json').read_text())
    assert first['npu'] != second['npu']
