"""Re-check the cross-rank QK-Clip agreement after splitting step() into two update rules."""
import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    from twinkle.module.optimizer import MaxLogitsTracker, MuonClip, MuonConfig, create_muon_param_groups

    torch.manual_seed(0)
    model = nn.Module()
    model.q_proj = nn.Linear(8, 8, bias=True)
    model.mlp = nn.Linear(8, 8, bias=False)
    groups = [{
        'params': [model.q_proj.weight, model.q_proj.bias, model.mlp.weight],
        'param_names': ['q_proj.weight', 'q_proj.bias', 'mlp.weight'],
        'lr': 0.0,
        'weight_decay': 0.0,
    }]
    opt = MuonClip(create_muon_param_groups(groups, MuonConfig(qk_clip_tau=100.0)), lr=0.0, weight_decay=0.0)
    for g in opt.param_groups:
        g['lr'], g['weight_decay'] = 0.0, 0.0
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    before = model.q_proj.weight.detach().clone()
    opt.step(max_logits=400.0 if rank == 1 else 120.0)
    scale = float((model.q_proj.weight.detach() / before).flatten()[0])
    got = [None, None]
    dist.all_gather_object(got, scale)
    if rank == 0:
        ok = abs(got[0] - got[1]) < 1e-9 and abs(got[0] - 0.5) < 1e-6
        print(f'B1 rank-local 120/400 -> 缩放比 {[round(s, 6) for s in got]} 一致且=0.5: {ok}')

    w = [None, None]
    dist.all_gather_object(w, model.q_proj.weight.detach().tolist())
    if rank == 0:
        print(f'B2 权重跨 rank 完全相同: {w[0] == w[1]}')

    # An AdamW group in the mix must not be clipped, and must not disturb the collective.
    MaxLogitsTracker.consume()
    bias_before = model.q_proj.bias.detach().clone()
    opt.step(max_logits=400.0)
    if rank == 0:
        print(f'B3 AdamW 组(bias)未被裁剪: {torch.allclose(model.q_proj.bias, bias_before)}')

    MaxLogitsTracker.consume()
    if rank == 1:
        MaxLogitsTracker._update(torch.tensor(900.0))
    before = model.q_proj.weight.detach().clone()
    opt.step()
    scale = float((model.q_proj.weight.detach() / before).flatten()[0])
    got = [None, None]
    dist.all_gather_object(got, scale)
    if rank == 0:
        print(f'B4 仅 rank1 有记录 -> {[round(s, 6) for s in got]} 一致: {abs(got[0] - got[1]) < 1e-9}')

    MaxLogitsTracker.consume()
    before = model.q_proj.weight.detach().clone()
    opt.step()
    if rank == 0:
        print(f'B5 两 rank 均无记录 -> 不裁剪不挂死: {torch.allclose(model.q_proj.weight, before)}')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
