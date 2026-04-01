import argparse
from pathlib import Path

import torch


def _load(path: str):
    return torch.load(path, map_location='cpu')


def _grad_norm(grad_dict: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for grad in grad_dict.values():
        total += grad.float().pow(2).sum().item()
    return total**0.5


def _compare_grads(base_grads: dict[str, torch.Tensor], other_grads: dict[str, torch.Tensor]):
    base_names = set(base_grads)
    other_names = set(other_grads)
    missing_in_other = sorted(base_names - other_names)
    missing_in_base = sorted(other_names - base_names)

    worst_name = None
    worst_max_diff = 0.0
    worst_mean_diff = 0.0
    compared = 0

    for name in sorted(base_names & other_names):
        base_grad = base_grads[name].float()
        other_grad = other_grads[name].float()
        if base_grad.shape != other_grad.shape:
            raise ValueError(f'Gradient shape mismatch for {name}: {tuple(base_grad.shape)} vs {tuple(other_grad.shape)}')
        diff = (base_grad - other_grad).abs()
        max_diff = diff.max().item()
        if max_diff > worst_max_diff:
            worst_name = name
            worst_max_diff = max_diff
            worst_mean_diff = diff.mean().item()
        compared += 1

    return {
        'compared_params': compared,
        'missing_in_other': missing_in_other,
        'missing_in_base': missing_in_base,
        'worst_name': worst_name,
        'worst_max_diff': worst_max_diff,
        'worst_mean_diff': worst_mean_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base')
    parser.add_argument('other')
    args = parser.parse_args()

    base_path = Path(args.base)
    other_path = Path(args.other)
    base = _load(str(base_path))
    other = _load(str(other_path))

    base_loss = float(base['loss'])
    other_loss = float(other['loss'])
    base_grads = base.get('gradients', {})
    other_grads = other.get('gradients', {})

    base_grad_norm = _grad_norm(base_grads)
    other_grad_norm = _grad_norm(other_grads)
    grad_cmp = _compare_grads(base_grads, other_grads)

    print(f'base: {base_path}')
    print(f'other: {other_path}')
    print(f'base ulysses_size: {base.get("ulysses_size")}')
    print(f'other ulysses_size: {other.get("ulysses_size")}')
    print(f'step: base={base.get("step")} other={other.get("step")}')
    print(f'loss: base={base_loss:.8f} other={other_loss:.8f} diff={abs(base_loss - other_loss):.8f}')
    print(
        f'grad_norm: base={base_grad_norm:.8f} other={other_grad_norm:.8f} '
        f'diff={abs(base_grad_norm - other_grad_norm):.8f}')
    print(f'compared_params: {grad_cmp["compared_params"]}')
    if grad_cmp['missing_in_other']:
        print(f'missing_in_other: {grad_cmp["missing_in_other"][:10]}')
    if grad_cmp['missing_in_base']:
        print(f'missing_in_base: {grad_cmp["missing_in_base"][:10]}')
    print(f'worst_param: {grad_cmp["worst_name"]}')
    print(f'worst_max_diff: {grad_cmp["worst_max_diff"]:.8f}')
    print(f'worst_mean_diff: {grad_cmp["worst_mean_diff"]:.8f}')


if __name__ == '__main__':
    main()
