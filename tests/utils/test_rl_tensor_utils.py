import pytest
import torch

from twinkle.utils.rl_tensor_utils import align_per_token_values


def test_align_per_token_values_accepts_json_rows() -> None:
    actual = align_per_token_values(
        [[-0.1, -0.2], [-0.3, -0.4]],
        (2, 2),
        device=torch.device('cpu'),
        dtype=torch.float32,
        name='ref_logps',
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
    )


def test_align_per_token_values_pads_ragged_json_rows() -> None:
    actual = align_per_token_values(
        [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
        (2, 3),
        device=torch.device('cpu'),
        dtype=torch.float32,
        name='ref_logps',
        valid_mask=torch.tensor([[True, True, True], [True, True, False]]),
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, 0.0]]),
    )


def test_align_per_token_values_rejects_missing_valid_tokens() -> None:
    with pytest.raises(ValueError, match=r'ref_logps\[1\] has 2 tokens'):
        align_per_token_values(
            [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
            (2, 3),
            device=torch.device('cpu'),
            dtype=torch.float32,
            name='ref_logps',
            valid_mask=torch.ones(2, 3, dtype=torch.bool),
        )


def test_align_per_token_values_rejects_short_batches() -> None:
    with pytest.raises(ValueError, match='batch size'):
        align_per_token_values(
            [[-0.1, -0.2]],
            (2, 2),
            device=torch.device('cpu'),
            dtype=torch.float32,
            name='ref_logps',
        )
