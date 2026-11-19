# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

from twinkle.model.tasks import get_task_spec_obj
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.module.head import SeqClsHead


class _Classifier(torch.nn.Module):
    def __init__(self, attached):
        super().__init__()
        self.backbone = torch.nn.Linear(32, 32)
        if attached:
            self.task_head = SeqClsHead(32, 1, squeeze_single=True)
        else:
            self.score = torch.nn.Linear(32, 1, bias=False)

    def forward(self, inputs_embeds, **kwargs):
        hidden = self.backbone(inputs_embeds)
        if hasattr(self, 'task_head'):
            return SimpleNamespace(hidden_states=(hidden,))
        return {'logits': self.score(hidden)[:, -1]}


def _value_forward(model, inputs):
    # Exercise both the legacy per-token patch and the attached-head readout
    # through the same entry point that TransformersModel's value task uses.
    wrapper = SimpleNamespace(model=model)
    wrapper._strip_encoder_position_ids = TransformersModel._strip_encoder_position_ids.__get__(wrapper)
    with get_task_spec_obj('value').transformers_context(model):
        return TransformersModel._forward_with_task_readout(
            wrapper, model, 'value', {'inputs_embeds': inputs})['logits']


@pytest.mark.parametrize('attached', [False, True], ids=['score', 'attached-head'])
def test_value_readout_is_visible_to_existing_post_forward_hooks(attached):
    model = _Classifier(attached)
    observed = []
    handle = model.register_forward_hook(lambda module, args, output: observed.append(output))
    try:
        output = _value_forward(model, torch.randn(1, 8, 32))
        assert observed[0]['logits'] is output
        assert output.shape == (1, 8)
        assert len(model._forward_hooks) == 1  # Only the caller's hook remains.
    finally:
        handle.remove()


def _fsdp_value_backward(rank, directory, attached):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/pg', rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(42)
        model = _Classifier(attached)
        reference = copy.deepcopy(model)
        mesh = init_device_mesh('cpu', (2,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                      output_dtype=torch.bfloat16)
        fully_shard(model.backbone, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)

        # PPO obtains old values in eval mode before the critic's first update.
        model.eval()
        with torch.no_grad():
            assert _value_forward(model, torch.randn(1, 8, 32)).shape == (1, 8)
        model.train()
        torch.manual_seed(100 + rank)
        for _ in range(4):
            inputs = torch.randn(1, 8, 32)
            output = _value_forward(model, inputs)
            (output.float().square().mean() / 4).backward()
            with torch.autocast('cpu', dtype=torch.bfloat16):
                expected_output = _value_forward(reference, inputs)
            torch.testing.assert_close(output, expected_output)
            (expected_output.float().square().mean() / 4).backward()

        # Catch missing head/backbone reductions as well as the zero-storage crash.
        expected_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            expected_grad = expected_params[name].grad
            assert expected_grad is not None, name
            dist.all_reduce(expected_grad)
            expected_grad /= 2
            assert parameter.grad is not None, name
            actual_grad = parameter.grad.full_tensor()
            assert torch.isfinite(actual_grad).all(), name
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-3)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(Version(torch.__version__.split('+')[0]) < Version('2.11'),
                    reason='CPU FSDP2 execution requires PyTorch 2.11 or newer')
@pytest.mark.parametrize('attached', [False, True], ids=['score', 'attached-head'])
def test_fsdp_value_readout_backward_after_inference(tmp_path, attached):
    torch.multiprocessing.spawn(_fsdp_value_backward, args=(str(tmp_path), attached), nprocs=2)
