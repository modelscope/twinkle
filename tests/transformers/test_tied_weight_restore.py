from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy


@pytest.mark.parametrize('backend', ['native', 'accelerate'])
@pytest.mark.parametrize('missing_independent_weight', [False, True])
def test_full_restore_expands_tied_aliases_and_rejects_missing_independent_weights(backend, missing_independent_weight):
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(4, 3)
    model.lm_head = torch.nn.Linear(3, 4, bias=False)
    model.lm_head.weight = model.embedding.weight
    model.other = torch.nn.Linear(3, 3)
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint = {name: value for name, value in saved.items() if name != 'lm_head.weight'}
    if missing_independent_weight:
        checkpoint.pop('other.weight')
    for param in model.parameters():
        param.data.fill_(7)
    if backend == 'native':
        strategy = object.__new__(NativeFSDPStrategy)
        strategy.device_mesh = object()
    else:
        strategy = object.__new__(AccelerateStrategy)
        strategy.accelerator = SimpleNamespace(
            unwrap_model=lambda value, **kwargs: value,
            state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2)))

    def strict_cpu_load(target, state, *, options):
        return target.load_state_dict(state, strict=options.strict)

    # Replace only distributed transport; the real strict PyTorch loader checks keys and writes weights.
    with patch('torch.distributed.checkpoint.state_dict.set_model_state_dict', side_effect=strict_cpu_load):
        if missing_independent_weight:
            with pytest.raises(RuntimeError, match='other.weight'):
                strategy.load_full_state_dict(model, checkpoint)
        else:
            strategy.load_full_state_dict(model, checkpoint)
            for name, actual in model.state_dict().items():
                torch.testing.assert_close(actual, saved[name])
    assert 'lm_head.weight' not in checkpoint
