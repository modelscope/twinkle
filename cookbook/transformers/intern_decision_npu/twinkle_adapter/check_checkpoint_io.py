"""Four-rank NPU check: sharded Adam moments and step survive DCP save/reload."""
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from .checkpoint import install_sharded_optimizer_io, reshard_for_checkpoint

rank = int(os.environ['RANK'])
torch.npu.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('hccl')
mesh = init_device_mesh('npu', (dist.get_world_size(),))
cpu_group = dist.new_group(backend='gloo', timeout=timedelta(minutes=5))
def fence():
    dist.monitored_barrier(group=cpu_group, timeout=timedelta(minutes=5))
torch.manual_seed(42)
model = torch.nn.Linear(16, 16).to('npu')
fully_shard(model, mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
x = torch.randn(4, 16, device='npu', dtype=torch.bfloat16)
model(x).float().square().mean().backward()
optimizer.step()
optimizer.zero_grad()
def local(t):
    return t.to_local() if hasattr(t, 'to_local') else t
expected = [{k: local(v).detach().cpu().clone() for k, v in s.items()}
            for s in optimizer.state.values()]
model.eval()
with torch.no_grad():
    model(x)
optimizer_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
before_reshard_match = optimizer_ids.issubset({id(p) for p in model.parameters()})
reshard_for_checkpoint(model)
assert optimizer_ids.issubset({id(p) for p in model.parameters()})
strategy = SimpleNamespace()
install_sharded_optimizer_io(strategy, cpu_group, fence)
path = '/workspace/results/twinkle-dcp-eval-io-check'
strategy.save_optimizer_checkpoint(model, optimizer, path)
new_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
strategy.load_optimizer_checkpoint(model, new_optimizer, path)
assert len(expected) == len(new_optimizer.state)
for reference, actual in zip(expected, new_optimizer.state.values()):
    assert reference.keys() == actual.keys()
    for key in reference:
        torch.testing.assert_close(local(actual[key]).cpu(), reference[key], rtol=0, atol=0)
fence()
if rank == 0:
    result = {'status': 'passed', 'ranks': dist.get_world_size(),
              'optimizer_parameter_identity_match_before_reshard': before_reshard_match,
              'optimizer_parameter_identity_match_after_reshard': True,
              'optimizer_moments_and_step_exactly_restored': True}
    Path('/workspace/results/twinkle-dcp-eval-io-check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
dist.destroy_process_group()
