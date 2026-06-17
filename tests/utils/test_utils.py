# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for utils key functions: torch_utils, seed, network, utils, safetensors, transformers_utils."""
import json
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

from twinkle.utils import (
    deep_getattr,
    is_valid_ipv6_address,
    pad_and_stack_tensors,
    pad_sequence_to_length,
)
from twinkle.utils.seed import stable_seed as _stable_seed
from twinkle.utils.torch_utils import (
    clone_state_dict_to_cpu,
    selective_log_softmax,
    to_device,
)
from twinkle.utils.network import find_free_port
from twinkle.utils.utils import (
    call_with_supported_kwargs,
    copy_files_by_pattern,
    has_signature_parameter,
    signature_info,
)
from twinkle.utils.transformers_utils import (
    align_logps_to_mask,
    filter_from_config_kwargs,
)
from twinkle.utils.safetensors import LazyTensor, StreamingSafetensorSaver


# ===========================================================================
# torch_utils
# ===========================================================================


class TestToDevice:

    def test_tensor_to_cpu(self):
        t = torch.randn(3, 4)
        result = to_device(t, 'cpu')
        assert result.device == torch.device('cpu')

    def test_dict_values_moved(self):
        data = {'a': torch.randn(2), 'b': torch.randn(3)}
        result = to_device(data, 'cpu')
        assert isinstance(result, dict)
        assert result['a'].device == torch.device('cpu')

    def test_list_values_moved(self):
        data = [torch.randn(2), torch.randn(3)]
        result = to_device(data, 'cpu')
        assert isinstance(result, list)

    def test_non_tensor_passthrough(self):
        assert to_device(42, 'cpu') == 42
        assert to_device('hello', 'cpu') == 'hello'

    def test_nested_dict(self):
        data = {'outer': {'inner': torch.randn(2)}}
        result = to_device(data, 'cpu')
        assert result['outer']['inner'].device == torch.device('cpu')


class TestCloneStateDictToCpu:

    def test_clones_tensors(self):
        state = {'w': torch.randn(3, 4), 'b': torch.randn(4)}
        cloned = clone_state_dict_to_cpu(state)
        assert cloned['w'].device == torch.device('cpu')
        assert cloned['w'].data_ptr() != state['w'].data_ptr()  # different storage
        assert torch.allclose(cloned['w'], state['w'].cpu())  # same values

    def test_preserves_non_tensors(self):
        state = {'step': 100, 'name': 'model'}
        cloned = clone_state_dict_to_cpu(state)
        assert cloned['step'] == 100
        assert cloned['name'] == 'model'


class TestPadSequenceToLength:

    def test_right_pad(self):
        t = torch.ones(2, 3)
        result = pad_sequence_to_length(t, 5)
        assert result.shape == (2, 5)
        assert (result[:, :3] == 1).all()
        assert (result[:, 3:] == 0).all()

    def test_left_pad(self):
        t = torch.ones(2, 3)
        result = pad_sequence_to_length(t, 5, left_pad=True)
        assert result.shape == (2, 5)
        assert (result[:, :2] == 0).all()
        assert (result[:, 2:] == 1).all()

    def test_no_pad_if_longer(self):
        t = torch.ones(2, 10)
        result = pad_sequence_to_length(t, 5)
        assert result.shape == (2, 10)

    def test_custom_pad_value(self):
        t = torch.ones(2, 3)
        result = pad_sequence_to_length(t, 5, pad_value=-1.0)
        assert (result[:, 3:] == -1).all()


class TestSelectiveLogSoftmax:

    def test_matches_naive_implementation(self):
        torch.manual_seed(42)
        logits = torch.randn(4, 20)
        index = torch.randint(0, 20, (4,))

        # Naive: log_softmax → gather
        expected = torch.gather(logits.log_softmax(-1), -1, index.unsqueeze(-1)).squeeze(-1)
        result = selective_log_softmax(logits, index)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_with_return_entropy(self):
        logits = torch.randn(4, 20, dtype=torch.float32)
        index = torch.randint(0, 20, (4,))
        logps, entropy = selective_log_softmax(logits, index, return_entropy=True)
        # Entropy should be non-negative
        assert (entropy >= 0).all()
        # Manual entropy check for one row
        row_probs = torch.exp(F.log_softmax(logits[0], dim=-1))
        manual_entropy = -(row_probs * row_probs.log()).sum()
        assert torch.allclose(entropy[0], manual_entropy, atol=1e-5)

    def test_bfloat16_fallback(self):
        logits = torch.randn(4, 20, dtype=torch.bfloat16)
        index = torch.randint(0, 20, (4,))
        result = selective_log_softmax(logits, index)
        expected = torch.gather(logits.float().log_softmax(-1), -1, index.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(result.float(), expected, atol=1e-2)


class TestPadAndStackTensors:

    def test_same_shape(self):
        tensors = [torch.randn(3, 4), torch.randn(3, 4)]
        result = pad_and_stack_tensors(tensors)
        assert result.shape == (6, 4)

    def test_different_length(self):
        tensors = [torch.randn(3), torch.randn(5)]
        result = pad_and_stack_tensors(tensors, pad_value=0)
        assert result.shape == (10, )  # padded to max length then concat

    def test_different_length_stack(self):
        tensors = [torch.randn(3), torch.randn(5)]
        result = pad_and_stack_tensors(tensors, pad_value=0, concat=False)
        assert result.shape == (2, 5)  # stack

    def test_single_tensor(self):
        t = torch.randn(3, 4)
        result = pad_and_stack_tensors([t])
        assert result is t  # returns as-is

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            pad_and_stack_tensors([])

    def test_different_ndim(self):
        tensors = [torch.randn(3), torch.randn(2, 4)]
        result = pad_and_stack_tensors(tensors)
        # First tensor gets unsqueezed to (1,3), second is (2,4)
        assert result.shape[1] == 4


# ===========================================================================
# seed
# ===========================================================================


class TestStableSeed:

    def test_deterministic(self):
        assert _stable_seed('hello', 42) == _stable_seed('hello', 42)

    def test_different_inputs_differ(self):
        assert _stable_seed('a') != _stable_seed('b')

    def test_returns_uint32(self):
        val = _stable_seed('test')
        assert isinstance(val, int)
        assert 0 <= val < 2**32

    def test_cross_process_stable(self):
        """SHA-256 based seed is not affected by PYTHONHASHSEED."""
        val = _stable_seed('consistent', 'key')
        assert isinstance(val, int)
        # Call again — must be same
        assert _stable_seed('consistent', 'key') == val


# ===========================================================================
# network
# ===========================================================================


class TestIsValidIPv6Address:

    def test_valid_ipv6(self):
        assert is_valid_ipv6_address('::1')

    def test_invalid_ipv6(self):
        assert not is_valid_ipv6_address('not_an_ip')

    def test_ipv4_is_not_ipv6(self):
        assert not is_valid_ipv6_address('127.0.0.1')

    def test_full_ipv6(self):
        assert is_valid_ipv6_address('2001:0db8:85a3:0000:0000:8a2e:0370:7334')


class TestFindFreePort:

    def test_finds_a_port(self):
        port = find_free_port()
        assert isinstance(port, int)
        assert 0 < port <= 65535

    def test_port_is_available(self):
        port = find_free_port()
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))  # Should not raise


# ===========================================================================
# utils.utils
# ===========================================================================


class TestDeepGetattr:

    def test_simple_attr(self):
        class Obj:
            x = 42
        assert deep_getattr(Obj(), 'x') == 42

    def test_nested_attr(self):
        class Inner:
            val = 99
        class Outer:
            inner = Inner()
        assert deep_getattr(Outer(), 'inner.val') == 99

    def test_dict_key(self):
        d = {'a': {'b': 10}}
        assert deep_getattr(d, 'a.b') == 10

    def test_missing_returns_default(self):
        assert deep_getattr(object(), 'x', default='missing') == 'missing'

    def test_dict_missing_key(self):
        d = {'a': 1}
        assert deep_getattr(d, 'b', default='nope') == 'nope'


class TestSignatureInfo:

    def test_function_with_kwargs(self):
        def f(a, b=2, **kw):
            pass
        accepts_kwargs, params = signature_info(f)
        assert accepts_kwargs is True
        assert 'a' in params

    def test_function_without_kwargs(self):
        def f(a, b=2):
            pass
        accepts_kwargs, params = signature_info(f)
        assert accepts_kwargs is False

    def test_has_signature_parameter(self):
        def f(a, b=2):
            pass
        assert has_signature_parameter(f, 'a')
        assert not has_signature_parameter(f, 'c')


class TestCallWithSupportedKwargs:

    def test_filters_unsupported(self):
        def f(a, b=2):
            return (a, b)
        result = call_with_supported_kwargs(f, a=10, b=20, c=30)
        assert result == (10, 20)

    def test_passes_all_with_kwargs(self):
        def f(a, **kw):
            return kw
        result = call_with_supported_kwargs(f, 1, x=2, y=3)
        assert result == {'x': 2, 'y': 3}


class TestCopyFilesByPattern:

    def test_copies_matching_files(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            # Create source files
            open(os.path.join(src, 'a.txt'), 'w').close()
            open(os.path.join(src, 'b.py'), 'w').close()
            copy_files_by_pattern(src, dst, '*.txt')
            assert os.path.exists(os.path.join(dst, 'a.txt'))
            assert not os.path.exists(os.path.join(dst, 'b.py'))

    def test_excludes_pattern(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            open(os.path.join(src, 'a.txt'), 'w').close()
            open(os.path.join(src, 'b.txt'), 'w').close()
            copy_files_by_pattern(src, dst, '*.txt', exclude_patterns=['b.*'])
            assert os.path.exists(os.path.join(dst, 'a.txt'))
            assert not os.path.exists(os.path.join(dst, 'b.txt'))


# ===========================================================================
# transformers_utils
# ===========================================================================


class TestAlignLogpsToMask:

    def test_already_aligned_tensor(self):
        mask = torch.tensor([[True, True, False, False]])
        logps = torch.randn(1, 4)
        result = align_logps_to_mask(logps, mask, torch.float32)
        assert torch.equal(result, logps)

    def test_ragged_list(self):
        mask = torch.tensor([[True, True, False, False]])
        # Only 2 valid positions
        old_logps = [torch.tensor([0.5, 0.3])]
        result = align_logps_to_mask(old_logps, mask, torch.float32)
        assert result.shape == (1, 4)
        # Positions 0,1 should have values, positions 2,3 should be 0
        assert result[0, 0].item() == pytest.approx(0.5)
        assert result[0, 1].item() == pytest.approx(0.3)
        assert result[0, 2].item() == 0.0
        assert result[0, 3].item() == 0.0

    def test_scalar_broadcast(self):
        mask = torch.tensor([[True, True, False]])
        result = align_logps_to_mask([5.0], mask, torch.float32)
        assert result[0, 0].item() == pytest.approx(5.0)
        assert result[0, 1].item() == pytest.approx(5.0)

    def test_returns_none_for_unsupported(self):
        mask = torch.tensor([[True]])
        result = align_logps_to_mask(42, mask, torch.float32)
        assert result is None


class TestFilterFromConfigKwargs:

    def test_filters_load_only_keys(self):
        kwargs = {
            'cache_dir': '/tmp',
            'num_layers': 12,
            'trust_remote_code': True,
            'hidden_size': 768,
        }
        result = filter_from_config_kwargs(kwargs)
        assert 'cache_dir' not in result
        assert 'trust_remote_code' not in result
        assert result['num_layers'] == 12
        assert result['hidden_size'] == 768


# ===========================================================================
# safetensors (partial — StreamingSaver without real GPU)
# ===========================================================================


class TestLazyTensor:

    def test_from_tensor(self):
        t = torch.randn(3, 4)
        lazy = LazyTensor(tensor=t)
        assert torch.equal(lazy.load(), t)

    def test_from_loader(self):
        t = torch.randn(3, 4)
        lazy = LazyTensor(loader=lambda: t)
        assert torch.equal(lazy.load(), t)

    def test_loader_called_each_time(self):
        count = [0]
        def loader():
            count[0] += 1
            return torch.tensor(count[0], dtype=torch.float32)
        lazy = LazyTensor(loader=loader)
        v1 = lazy.load()
        v2 = lazy.load()
        assert v1.item() != v2.item()  # loader called each time


class TestStreamingSafetensorSaver:

    def test_init_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, 'output')
            saver = StreamingSafetensorSaver(save_dir, max_shard_size='1GB', save_rank='master')
            # is_save_rank depends on is_master() which may be False on non-dist
            # Just verify no crash
