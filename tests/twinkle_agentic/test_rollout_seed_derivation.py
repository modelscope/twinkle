# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect52b (QUEUE3195): the run seed reaches the rollout half as a per-(group, sample) request seed.

``--seed`` used to act only on the data shuffle; the rollout's ``SamplingParams.seed`` stayed None
(the sampler's advancing RNG), so the seed was a dead parameter for generation. The fix derives a
per-episode seed from the stable ``(group_key, sample_index)`` identity the driver stamps. The hard
constraint: a group's ``num_generations`` samples MUST get pairwise-distinct seeds -- a constant seed
collapses them to one identical trajectory, zeroing the group's reward variance and advantage (the
VL color fixture degeneracy signature).

PLAN (case | input | expected | failure it kills):
  distinct_within_group   | base 42, group '7', sample_index 0..3 | 4 pairwise-distinct seeds in [0,2^31) | constant-seed degeneracy (the trap)
  deterministic           | same (group,sample) twice           | identical seed                        | non-reproducible derivation
  base_seed_changes       | base 42 vs 43, same identity        | different seed                        | base seed ignored
  group_changes           | group '7' vs '8', same sample       | different seed                        | group identity ignored
  no_identity_none        | traj without group_key/sample_index | None                                  | seeding display/bare episodes
  non_int_group_none      | group_key='abc'                     | None                                  | crash on non-int identity
  episode_params_seeds    | base set + identity traj            | params.seed == _derive_seed(traj)     | seed not reaching SamplingParams
  episode_params_unseeded | base_seed=None                      | params.seed is None                   | default path altered (regression)
  override_seed_wins      | user_data sampling_params seed=99   | params.seed == 99                     | derived clobbering explicit override
  driver_stamps_index     | real driver, generations=4          | each group's rows sample_index={0,1,2,3} | fan-out not stamping positional identity
  reverse_constant        | all sample_index=0                  | seeds identical                       | proves distinctness detects degeneracy

The derivation oracles are independent properties (pairwise distinctness, determinism, base/group
sensitivity, range), not a recomputation of the crc32 formula. The driver case runs the REAL
fan-out/_start_attempt/_submit stamping over a fake single-turn rollout (only the engine step is
faked; the identity stamping under test is product code).
"""
from __future__ import annotations

import concurrent.futures as cf

from twinkle.data_format import attach_user_data
from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.async_rl.streaming_driver import StreamingDriver
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout, RolloutStatus


def _rollout(seed):
    return MultiTurnRollout(response_callback=lambda *a: None, template=None,
                            sampling_params=SamplingParams(), seed=seed)


def _traj(group_key, sample_index):
    return {'group_key': group_key, 'sample_index': sample_index}


def test_distinct_within_group():
    r = _rollout(42)
    seeds = [r._derive_seed(_traj('7', i)) for i in range(4)]
    assert len(set(seeds)) == 4
    assert all(0 <= s < 2**31 for s in seeds)


def test_deterministic():
    r = _rollout(42)
    assert r._derive_seed(_traj('7', 2)) == r._derive_seed(_traj('7', 2))


def test_base_seed_changes():
    assert _rollout(42)._derive_seed(_traj('7', 2)) != _rollout(43)._derive_seed(_traj('7', 2))


def test_group_changes():
    r = _rollout(42)
    assert r._derive_seed(_traj('7', 2)) != r._derive_seed(_traj('8', 2))


def test_no_identity_none():
    r = _rollout(42)
    assert r._derive_seed({'user_data': {}}) is None
    assert r._derive_seed({'group_key': '7'}) is None
    assert r._derive_seed({'sample_index': 1}) is None


def test_non_int_group_none():
    assert _rollout(42)._derive_seed(_traj('abc', 1)) is None


def test_episode_params_seeds():
    r = _rollout(42)
    traj = _traj('7', 3)
    params = r._episode_params(traj)
    assert params.seed == r._derive_seed(traj)
    assert params.seed is not None


def test_episode_params_unseeded_by_default():
    r = _rollout(None)
    assert r._episode_params(_traj('7', 3)).seed is None


def test_override_seed_wins():
    r = _rollout(42)
    traj = attach_user_data(_traj('7', 3), sampling_params={'seed': 99})
    assert r._episode_params(traj).seed == 99


def test_reverse_constant_sample_index_collapses():
    """Reverse verification: if every sample carries the same index (a constant seed), the seeds
    collapse to one -- exactly the degeneracy the distinctness assertion above detects."""
    r = _rollout(42)
    seeds = [r._derive_seed(_traj('7', 0)) for _ in range(4)]
    assert len(set(seeds)) == 1


def _completed(value):
    future = cf.Future()
    future.set_result(value)
    return future


class _Rollout:
    def step(self, traj, episode_id=None, session_wait=None):
        sample = RolloutStatus(episode_id=episode_id, done=True, turns=1, token_lengths=(4, 6))
        return _completed(([dict(traj)], sample))

    def stop(self, traj, episode_id=None):
        pass


def test_driver_stamps_sample_index():
    dataloader = [[{'prompt': i} for i in range(4)]]
    driver = StreamingDriver(dataloader, _Rollout(), batch_size=8, num_generations=4,
                             orms=[lambda trajs: _completed([1.0] * len(trajs))], max_staleness=0)
    try:
        rows = [row for batch, _facts in driver for row in batch]
    finally:
        driver.close()
    by_group = {}
    for row in rows:
        by_group.setdefault(row['group_key'], []).append(row['sample_index'])
    assert by_group, 'no rows delivered'
    for indices in by_group.values():
        assert sorted(indices) == [0, 1, 2, 3]