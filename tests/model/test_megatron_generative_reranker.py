from types import SimpleNamespace

import torch

from twinkle.model.tasks import _generative_reranker_score


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
