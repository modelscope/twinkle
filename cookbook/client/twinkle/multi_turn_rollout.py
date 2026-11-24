# Copyright (c) ModelScope Contributors. All rights reserved.
"""Train on synchronous batches of Blackjack games.

From the repo root, run python -m cookbook.client.twinkle.multi_turn_rollout.
"""

from twinkle import init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle_client.resources import RolloutResources

from cookbook.client.async_rl.streaming_multi_turn_grpo import (
    BASE_SEED, BlackjackEnv, prompt, report_training, setup_training, train_batch,
)

NUM_GENERATIONS = 2
MAX_GROUPS = 16
MAX_STEPS = 3


def train():
    with RolloutResources(init_twinkle_client) as resources:
        env = resources.own_environment(BlackjackEnv())
        model, rollout, checkpoints = setup_training(resources, env)
        advantage = GRPOAdvantage(num_generations=NUM_GENERATIONS)
        groups = 0

        try:
            for index in range(MAX_GROUPS):
                # Roll out the same deal independently before training on this group.
                trajectories = [
                    dict(prompt(BASE_SEED + index), group_key=str(index), episode_id=f'{index}-{member}')
                    for member in range(NUM_GENERATIONS)
                ]
                rows = resources.operations.run(rollout, trajectories)
                for row in rows:
                    if row.get('error'):
                        raise RuntimeError(row['error'])
                if any(row['truncated'] or not row['episode_end']['game_finished'] for row in rows):
                    continue
                groups += 1
                train_batch(model, checkpoints, advantage, rows)
                if model.completed_step_calls == MAX_STEPS:
                    break
        finally:
            report_training(model, groups)


if __name__ == '__main__':
    train()
