# Copyright (c) ModelScope Contributors. All rights reserved.
"""The step-delta probe's parameter digest.

Every fresh/resume and fresh/fresh conclusion in this project is read off these entries, so the
digest's own contract is what makes those readings evidence rather than assertion. The cases attack
the two ways it can silently under-report: an entry with no identity field, and an identity field
weakened by being computed over a lossy cast.

PLAN
| # | case | input | expected | failure it kills |
|---|------|-------|----------|------------------|
| 1 | entry carries an identity field | two trainable float32 params | dict with agg[3], bits[2], numel | a bare ``[sum, absmax, norm]``, which can show a difference but never establish agreement |
| 2 | change below aggregate resolution | one non-max element moved by one ulp | agg equal, bits differ | reporting "identical" over a real change -- the signature measured on a diverging conv1d weight |
| 3 | gradient absent | ``kind='grad'``, one param has no ``.grad`` | that key absent, the other present | a phantom zero entry read as "the gradient matched" |
| 4 | frozen parameter | ``requires_grad=False`` | key absent | digesting the frozen base, which is trivially unchanged, and calling that a clean update |
| 5 | empty local shard | a zero-element parameter | key present, agg zeros, bits ``[0, 0]`` | ``max()``/``norm()`` raising on an empty tensor, or dropping the key so two runs stop aligning |
| 6 | narrow-dtype parameter | a bfloat16 parameter | bits equal the native-dtype fingerprint, agg the float64 reduction | a fingerprint taken over a dtype other than the parameter's own |
| 7 | schedule independence | the same values in a different order | agg and bits both equal | a fingerprint that moves with the reduction order, which would make every comparison unusable |
"""
import torch
from torch import nn

from twinkle.model.transformers.transformers import TransformersModel
from twinkle.utils.torch_utils import bits_fingerprint


class DigestHarness:
    _digest_parameters = TransformersModel._digest_parameters
    # ``_shard`` is a ``staticmethod``; a bare assignment would rebind it as an instance method and
    # hand ``self`` to its ``tensor`` parameter.
    _shard = staticmethod(TransformersModel._shard)

    def __init__(self, model):
        self.model = model


def model_with(**shapes):
    module = nn.Module()
    for name, tensor in shapes.items():
        module.register_parameter(name, nn.Parameter(tensor, requires_grad=True))
    return module


def test_entries_carry_an_identity_field():
    digest = DigestHarness(model_with(a=torch.arange(6, dtype=torch.float32).reshape(2, 3),
                                      b=torch.ones(4, dtype=torch.float32)))._digest_parameters('param')
    assert set(digest) == {'a', 'b'}
    for name, entry in digest.items():
        assert isinstance(entry, dict), f'{name} is a bare aggregate, so it has no identity criterion'
        assert len(entry['agg']) == 3
        assert len(entry['bits']) == 2
        assert all(isinstance(value, int) for value in entry['bits'])
    assert digest['a']['numel'] == 6
    assert digest['b']['agg'] == [4.0, 1.0, 2.0]


def test_a_change_below_aggregate_resolution_still_moves_the_fingerprint():
    # One ulp on a single element of a wide tensor: the float64 sum and norm do not move at that
    # magnitude and the element is not the absmax, so all three aggregates read identically. This is
    # the exact shape of the divergence measured on ``layers.13.linear_attn.conv1d.weight``, where
    # ``|delta absmax|`` was zero while the run had demonstrably changed.
    # One element dominates the aggregates so that a one-ulp change to a small element lands below
    # their resolution: ulp(1e6) in float64 is ~1.2e-10, while one ulp of a 1e-5 float32 is ~9.5e-13.
    # Perturbing the large element instead would move every aggregate and prove nothing.
    base = torch.full((4096,), 1e-5, dtype=torch.float32)
    base[0] = 1e6
    moved = base.clone()
    moved[2048] = torch.nextafter(moved[2048], torch.tensor(float('inf'), dtype=torch.float32))
    assert moved[2048] != base[2048]

    before = DigestHarness(model_with(w=base))
    after = DigestHarness(model_with(w=moved))
    entry_before = before._digest_parameters('param')['w']
    entry_after = after._digest_parameters('param')['w']
    assert entry_before['agg'] == entry_after['agg'], 'the aggregates were expected to miss this'
    assert entry_before['bits'] != entry_after['bits']


def test_a_parameter_without_a_gradient_is_absent_from_the_grad_digest():
    module = model_with(trained=torch.ones(3), untrained=torch.ones(3))
    module.untrained.grad = None
    module.trained.grad = torch.full((3,), 2.0)
    digest = DigestHarness(module)._digest_parameters('grad')
    assert set(digest) == {'trained'}
    # The aggregate is taken over a float64 cast, so the expected norm has to be the float64 one:
    # ``torch.tensor(12.0).sqrt()`` is float32 and differs in the eighth digit.
    assert digest['trained']['agg'] == [6.0, 2.0, 12.0 ** 0.5]


def test_frozen_parameters_are_not_digested():
    module = model_with(trainable=torch.ones(3))
    module.register_parameter('frozen', nn.Parameter(torch.ones(3), requires_grad=False))
    assert set(DigestHarness(module)._digest_parameters('param')) == {'trainable'}


def test_an_empty_local_shard_keeps_its_key_and_digests_as_zeros():
    digest = DigestHarness(model_with(empty=torch.zeros(0), whole=torch.ones(2)))._digest_parameters('param')
    assert set(digest) == {'empty', 'whole'}
    assert digest['empty'] == {'agg': [0.0, 0.0, 0.0], 'bits': [0, 0], 'numel': 0, 'dtype': 'torch.float32'}


def test_a_narrow_dtype_parameter_is_fingerprinted_in_its_own_dtype():
    value = torch.tensor([1.5, -2.25, 3.0], dtype=torch.bfloat16)
    entry = DigestHarness(model_with(w=value))._digest_parameters('param')['w']
    assert entry['bits'] == bits_fingerprint(value.flatten())
    assert entry['agg'][0] == float(value.double().sum())
    assert entry['dtype'] == 'torch.bfloat16'


def test_the_fingerprint_does_not_depend_on_element_order():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    permuted = values[torch.tensor([3, 1, 0, 2])]
    first = DigestHarness(model_with(w=values))._digest_parameters('param')['w']
    second = DigestHarness(model_with(w=permuted))._digest_parameters('param')['w']
    assert first['bits'] == second['bits']
    assert first['agg'][0] == second['agg'][0]
    assert first['agg'][1] == second['agg'][1]
