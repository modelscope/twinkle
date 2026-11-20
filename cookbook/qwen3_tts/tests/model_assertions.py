"""Test-only parameter and gradient assertions."""
import hashlib
import torch

def trainable_snapshot(model):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not params or any('.lora_' not in n for n in params):
        raise AssertionError('Only LoRA parameters may be trainable')
    for branch in ('.talker.model.', '.talker.code_predictor.'):
        if not any(branch in n for n in params):
            raise AssertionError('Missing trainable branch: ' + branch)
    return {n: p.detach().cpu().clone() for n, p in params.items()}


def frozen_digest(model):
    """Hash all frozen weights, one tensor at a time; do not retain a base copy."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(name.encode())
        value = parameter.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_gradients(model):
    nonzero = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise AssertionError('Frozen parameter has gradient: ' + name)
            continue
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise AssertionError('Missing/nonfinite LoRA gradient: ' + name)
        if torch.count_nonzero(p.grad).item():
            nonzero.append(name)
    # Standard LoRA starts B at zero: A can legitimately have zero gradient.
    for branch in ('.talker.model.', '.talker.code_predictor.'):
        if not any(branch in name for name in nonzero):
            raise AssertionError('No nonzero gradient in ' + branch)
    return nonzero


def assert_updated(before, after):
    if before.keys() != after.keys():
        raise AssertionError('Trainable parameter set changed')
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if any(not torch.isfinite(value).all() for value in after.values()):
        raise AssertionError('Nonfinite updated weights')
    for branch in ('.talker.model.', '.talker.code_predictor.'):
        if not any(branch in name for name in changed):
            raise AssertionError('No parameter update in ' + branch)
    return changed
