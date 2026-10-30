"""PPO actor and critic match the full local batch under fixed and dynamic splitting."""
import pytest
import torch

from twinkle.loss import PPOLoss, PPOValueLoss
from .test_grpo_micro_batch import data
from .test_recipe_micro_batch import Harness


class PPOHarness(Harness):
    def forward(self, *, inputs, **kwargs):
        outputs = super().forward(inputs=inputs, **kwargs)
        status = self.optimizer_group['default'].train_status
        width = status.inputs['labels'].shape[1]
        if any('completion_mask' in row for row in inputs):
            status.inputs['completion_mask'] = torch.tensor([
                row['completion_mask'] + [0] * (width - len(row['completion_mask'])) for row in inputs
            ])
        return outputs


@pytest.mark.parametrize('factory', [
    pytest.param(PPOLoss, id='policy-token-mean'),
    pytest.param(lambda: PPOLoss(loss_agg_mode='seq-mean-token-mean'), id='policy-sequence-mean'),
    pytest.param(PPOValueLoss, id='value-token-mean'),
])
@pytest.mark.parametrize('options', [
    pytest.param({'micro_batch_size': 2}, id='fixed'),
    pytest.param({'micro_batch_size': 4}, id='fixed-short-tail'),
    pytest.param({'micro_batch_size': 4, 'dynamic_batching': True,
                  'max_tokens_per_micro_batch': 18, 'packing_algorithm': 'ffd'}, id='dynamic-ffd'),
    pytest.param({'micro_batch_size': 4, 'dynamic_batching': True,
                  'max_tokens_per_micro_batch': 18, 'packing_algorithm': 'kk'}, id='dynamic-kk'),
])
@pytest.mark.parametrize('completion_mask', [False, True])
def test_ppo_loss_and_gradients_match_unsplit_batch(factory, options, completion_mask):
    baseline, internal = PPOHarness(factory()), PPOHarness(factory())
    rows = data()[:6]
    for index, row in enumerate(rows):
        row['labels'][0] = -100
        if completion_mask:
            row['completion_mask'] = [int((position + index) % 3 != 0) for position in range(len(row['labels']))]
    kwargs = {
        'advantages': [1., -.5, .7, -1.2, .4, -.3],
        'old_logps': [[0.] * len(row['labels']) for row in rows],
        'old_values': [[0.] * len(row['labels']) for row in rows],
        'returns': [[.2] * len(row['labels']) for row in rows],
    }

    expected = baseline.run(rows, **kwargs)
    actual = internal.run(rows, **options, **kwargs)

    assert actual['loss'] == pytest.approx(expected['loss'])
    torch.testing.assert_close(internal.normalized_gradient(), baseline.normalized_gradient())


@pytest.mark.parametrize('factory', [PPOLoss, PPOValueLoss])
def test_empty_ppo_micro_batch_keeps_loss_and_gradient_finite(factory):
    baseline, internal = PPOHarness(factory()), PPOHarness(factory())
    rows = data()[:4]
    for row in rows[:2]:
        row['labels'] = [-100] * len(row['labels'])
    kwargs = {
        'advantages': [1., -.5, .7, -1.2],
        'old_logps': [[0.] * len(row['labels']) for row in rows],
        'old_values': [[0.] * len(row['labels']) for row in rows],
        'returns': [[.2] * len(row['labels']) for row in rows],
    }

    expected = baseline.run(rows, **kwargs)
    actual = internal.run(rows, micro_batch_size=2, **kwargs)

    assert actual['loss'] == pytest.approx(expected['loss'])
    torch.testing.assert_close(internal.normalized_gradient(), baseline.normalized_gradient())
