# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for the shared input-handling seam in ``twinkle.sampler.base.Sampler``.

Every sampler backend (transformers / vLLM / SGLang) inherits ``Sampler``'s input
classification (a not-yet-encoded ``Trajectory`` vs an already-encoded ``InputFeature``),
its trajectory encode/decode transformation, and the fail-loudly guards. That logic is
pure and backend-independent, so it is exercised here on CPU with a minimal concrete
``Sampler`` (its two abstract methods are never reached by the seam under test) and a
template collaborator that mirrors ``Template.encode`` / ``Template.decode``.

Consistent with ``test_sampler_over_existing_model.py``: these cases pin the seam -- how
the base class turns a template's output into an ``InputFeature``, which keys it keeps,
and where it refuses -- not generation itself.
"""
import pytest
import torch

from twinkle.sampler.base import Sampler


class _ConcreteSampler(Sampler):
    """Minimal concrete Sampler; the base-class seam never calls these two."""

    def sample(self, inputs, sampling_params=None, adapter_name='', *, num_samples=1):
        raise AssertionError('sample must not be called by the base-class seam under test')

    def apply_patch(self, patch_cls, **kwargs):
        raise AssertionError('apply_patch must not be called by the base-class seam under test')


class _FakeTemplate:
    """Mirrors the ``Template.encode`` / ``Template.decode`` contract that base.py relies on."""

    def __init__(self, encoded=None, decoded='decoded-text'):
        self._encoded = ({
            'input_ids': torch.tensor([1, 2, 3]),  # a tensor exercises the .tolist() branch
            'labels': torch.tensor([-100, -100, 3]),  # must be dropped from the InputFeature
            'attention_mask': [1, 1, 1],  # extra key -> must be copied through
            'images': ['pic'],  # extra key -> must be copied through
        } if encoded is None else encoded)
        self._decoded = decoded
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, trajectory, add_generation_prompt=True):
        self.encode_calls.append((trajectory, add_generation_prompt))
        return dict(self._encoded)

    def decode(self, token_ids):
        self.decode_calls.append(token_ids)
        return self._decoded


# ---------------------------------------------------------------------------
# Input classification: _not_encoded / _is_trajectory / _normalize_inputs
# ---------------------------------------------------------------------------

class TestInputClassification:

    def test_not_encoded_true_for_trajectory(self):
        assert Sampler._not_encoded({'messages': [{'role': 'user', 'content': 'hi'}]}) is True

    def test_not_encoded_false_when_input_ids_present(self):
        assert Sampler._not_encoded({'input_ids': [1, 2, 3]}) is False

    def test_not_encoded_false_when_input_embedding_present(self):
        assert Sampler._not_encoded({'input_embedding': torch.zeros(2)}) is False

    def test_not_encoded_rejects_non_dict(self):
        with pytest.raises(AssertionError, match='Expected dict'):
            Sampler._not_encoded(['not', 'a', 'dict'])

    def test_is_trajectory_single_trajectory_dict(self):
        sampler = _ConcreteSampler()
        assert sampler._is_trajectory({'messages': []}) is True

    def test_is_trajectory_single_encoded_dict(self):
        sampler = _ConcreteSampler()
        assert sampler._is_trajectory({'input_ids': [1]}) is False

    def test_is_trajectory_list_inspects_first_element(self):
        sampler = _ConcreteSampler()
        assert sampler._is_trajectory([{'messages': []}, {'messages': []}]) is True
        assert sampler._is_trajectory([{'input_ids': [1]}]) is False

    def test_is_trajectory_empty_list_is_false(self):
        sampler = _ConcreteSampler()
        assert sampler._is_trajectory([]) is False

    def test_is_trajectory_non_dict_is_false(self):
        sampler = _ConcreteSampler()
        assert sampler._is_trajectory('a raw string') is False
        assert sampler._is_trajectory(42) is False

    def test_normalize_inputs_wraps_single_dict(self):
        sampler = _ConcreteSampler()
        single = {'input_ids': [1]}
        assert sampler._normalize_inputs(single) == [single]

    def test_normalize_inputs_passes_through_iterable(self):
        sampler = _ConcreteSampler()
        batch = [{'input_ids': [1]}, {'input_ids': [2]}]
        assert sampler._normalize_inputs(batch) == batch
        # A tuple is normalized to a list.
        assert sampler._normalize_inputs(tuple(batch)) == batch


# ---------------------------------------------------------------------------
# encode_trajectory
# ---------------------------------------------------------------------------

class TestEncodeTrajectory:

    def test_raises_without_template(self):
        sampler = _ConcreteSampler()
        assert sampler.template is None
        with pytest.raises(ValueError, match='Template not set'):
            sampler.encode_trajectory({'messages': []})

    def test_builds_input_feature_and_filters_keys(self):
        sampler = _ConcreteSampler()
        template = _FakeTemplate()
        sampler.template = template
        result = sampler.encode_trajectory({'messages': []})
        # input_ids came back as a tensor and must be converted to a plain list.
        assert result['input_ids'] == [1, 2, 3]
        assert isinstance(result['input_ids'], list)
        # 'labels' is deliberately not carried into the sampler InputFeature.
        assert 'labels' not in result
        # Extra multimodal / mask keys are copied through untouched.
        assert result['attention_mask'] == [1, 1, 1]
        assert result['images'] == ['pic']

    def test_add_generation_prompt_is_forwarded(self):
        sampler = _ConcreteSampler()
        template = _FakeTemplate()
        sampler.template = template
        trajectory = {'messages': []}
        sampler.encode_trajectory(trajectory, add_generation_prompt=False)
        assert template.encode_calls == [(trajectory, False)]
        sampler.encode_trajectory(trajectory)  # default True
        assert template.encode_calls[-1] == (trajectory, True)

    def test_raises_when_encode_returns_no_input_ids(self):
        sampler = _ConcreteSampler()
        sampler.template = _FakeTemplate(encoded={'attention_mask': [1, 1]})
        with pytest.raises(ValueError, match="must return 'input_ids'"):
            sampler.encode_trajectory({'messages': []})

    def test_list_input_ids_are_kept_as_is(self):
        sampler = _ConcreteSampler()
        sampler.template = _FakeTemplate(encoded={'input_ids': [7, 8], 'extra': 'x'})
        result = sampler.encode_trajectory({'messages': []})
        assert result['input_ids'] == [7, 8]
        assert result['extra'] == 'x'


# ---------------------------------------------------------------------------
# decode_response
# ---------------------------------------------------------------------------

class TestDecodeResponse:

    def test_raises_without_template(self):
        sampler = _ConcreteSampler()
        with pytest.raises(ValueError, match='Template not set'):
            sampler.decode_response([1, 2, 3])

    def test_delegates_to_template(self):
        sampler = _ConcreteSampler()
        template = _FakeTemplate(decoded='hello world')
        sampler.template = template
        assert sampler.decode_response([1, 2, 3]) == 'hello world'
        assert template.decode_calls == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# default encode (pooling) refuses loudly
# ---------------------------------------------------------------------------

class TestDefaultEncode:

    def test_encode_not_implemented_by_default(self):
        # A backend without a pooling head must fail loudly rather than return hidden
        # states that merely look like embeddings.
        sampler = _ConcreteSampler()
        with pytest.raises(NotImplementedError, match='does not support pooling'):
            sampler.encode({'input_ids': [1, 2, 3]})
