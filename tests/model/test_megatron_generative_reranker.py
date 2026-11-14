from types import SimpleNamespace

import torch

from twinkle.model.megatron.megatron import _generative_reranker_score


def test_generative_reranker_scores_yes_minus_no_and_backpropagates():
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: {'yes': 3, 'no': 7}[token])
    template = SimpleNamespace(processor=object(), tokenizer=tokenizer)
    logits = torch.randn(2, 5, 11, requires_grad=True)

    scores = _generative_reranker_score(logits, template)

    assert scores.shape == (2, 5, 1)
    torch.testing.assert_close(scores, logits[..., 3:4] - logits[..., 7:8])
    scores.sum().backward()
    assert torch.count_nonzero(logits.grad[..., 3]) == 10
    assert torch.count_nonzero(logits.grad[..., 7]) == 10
    assert torch.count_nonzero(logits.grad[..., [0, 1, 2, 4, 5, 6, 8, 9, 10]]) == 0


def test_full_determinism_selects_torch_gdn_kernels(monkeypatch):
    from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule
    from twinkle.model.megatron.megatron import MegatronModel

    gdn_type = type('Qwen3NextGatedDeltaNet', (torch.nn.Module,), {})
    gdn = gdn_type()
    gdn.chunk_gated_delta_rule = lambda *args: None
    gdn.causal_conv1d_fn = lambda *args: None
    wrapper = MegatronModel.__new__(MegatronModel)
    wrapper.model = [gdn]
    monkeypatch.setattr(torch, 'are_deterministic_algorithms_enabled', lambda: True)
    wrapper._apply_deterministic_model_kernels()
    assert gdn.chunk_gated_delta_rule is torch_chunk_gated_delta_rule
    assert gdn.causal_conv1d_fn is None
