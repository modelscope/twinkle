# Copyright (c) ModelScope Contributors. All rights reserved.
"""Headless-embedding gate for the megatron embedding patches (defect124).

mcore-bridge builds task_type='embedding' GPTModels with ``output_layer = None``
(register-then-None leaves None in ``_modules``); the passthrough patch must skip
such owners instead of dereferencing ``None.forward``, while the pooling hook
still attaches to the headless last-PP chunk.
"""
import torch
import torch.nn.functional as F

from twinkle.patch.megatron_emb import MegatronEmbeddingPatch, MegatronHiddenPassthroughPatch


class _Chunk(torch.nn.Module):
    """GPTModel-like chunk: a real output_layer registered first, then optionally None'd."""

    def __init__(self, headless: bool, post_process: bool = True):
        super().__init__()
        self.post_process = post_process
        self.output_layer = torch.nn.Linear(4, 4)
        if headless:
            self.output_layer = None

    def forward(self, hidden_states, position_ids=None, **kwargs):
        return hidden_states


class TestHeadlessPassthrough:

    def test_headless_owner_is_skipped_not_dereferenced(self):
        chunk = _Chunk(headless=True)
        assert 'output_layer' in chunk._modules and chunk._modules['output_layer'] is None
        patch = MegatronHiddenPassthroughPatch()
        assert patch(chunk) is chunk
        assert patch._patched == []
        assert patch.unpatch(chunk) is chunk

    def test_headed_owner_still_replaced_and_restored(self):
        chunk = _Chunk(headless=False)
        origin = chunk.output_layer.forward
        patch = MegatronHiddenPassthroughPatch()
        patch(chunk)
        assert chunk.output_layer.forward is not origin
        hidden = torch.randn(2, 3, 4)
        out, bias = chunk.output_layer(hidden)
        assert torch.equal(out, hidden) and bias is None
        patch.unpatch(chunk)
        assert chunk.output_layer.forward == origin

    def test_intermediate_pp_stage_untouched(self):
        chunk = _Chunk(headless=False, post_process=False)
        origin = chunk.output_layer.forward
        patch = MegatronHiddenPassthroughPatch()
        patch(chunk)
        assert patch._patched == []
        assert chunk.output_layer.forward == origin

    def test_mixed_chunks_with_headless_last(self):
        chunks = [_Chunk(headless=False, post_process=False), _Chunk(headless=True)]
        patch = MegatronHiddenPassthroughPatch()
        assert patch(chunks) is chunks
        assert patch._patched == []
        assert patch.unpatch(chunks) is chunks


class TestHeadlessEmbeddingPatch:

    def test_headless_chunk_pools_last_valid_token(self):
        chunk = _Chunk(headless=True)
        patch = MegatronEmbeddingPatch()
        patch(chunk)
        hidden = torch.randn(2, 6, 4)
        position_ids = torch.tensor([[0, 1, 2, 3, -1, -1], [0, 1, 2, 3, 4, 5]])
        out = chunk(hidden, position_ids=position_ids)
        expected = F.normalize(hidden[torch.arange(2), torch.tensor([3, 5])], p=2, dim=1)
        assert out.shape == (2, 4)
        assert torch.allclose(out, expected, atol=1e-6)
        assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-6)
        patch.unpatch(chunk)
        assert torch.equal(chunk(hidden, position_ids=position_ids), hidden)

    def test_intermediate_pp_stage_gets_no_hook(self):
        chunk = _Chunk(headless=False, post_process=False)
        patch = MegatronEmbeddingPatch()
        patch(chunk)
        assert patch._hooks == []
        hidden = torch.randn(2, 6, 4)
        assert torch.equal(chunk(hidden), hidden)
        patch.unpatch(chunk)