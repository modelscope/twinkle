# Copyright (c) ModelScope Contributors. All rights reserved.
"""Prepare scored client streaming samples for group-relative training."""
import math
from collections.abc import Sequence

from twinkle import get_logger
from twinkle_agentic.rollout.multi_turn import RolloutSample

logger = get_logger()


def training_groups(
    samples: Sequence[RolloutSample], *, num_generations: int, drop_truncated: bool = False,
) -> list[list[RolloutSample]]:
    """Return complete, single-policy groups from one scored streaming batch.

    This prepares a batch already collected by StreamingDriver; it never schedules,
    waits for or regenerates samples. Group identity and member indices must be the
    original driver/admission metadata. Incomplete or mixed-policy groups are skipped.
    Truncated groups are kept by default; tasks requiring terminal episodes opt out.

    Validate the entire batch before returning any groups, so a failed rollout cannot
    be hidden by an earlier skipped group or a caller's update budget. Accepted samples
    require aligned response tokens, masks and finite old logprobs, plus a nonempty row
    of finite outcome rewards. Multiple reward channels are supported; aggregating them,
    computing advantages and deciding whether to update remain the consumer's job.
    """
    if isinstance(num_generations, bool) or not isinstance(num_generations, int) or num_generations < 1:
        raise ValueError('num_generations must be a positive integer')
    groups = {}
    for sample in samples:
        if (sample.rollout_infos.get('error') is not None
                or sample.rollout_infos.get('stop_reason') in {'error', 'abort', 'generation_error'}):
            raise RuntimeError(f'rollout failed: {sample.rollout_infos}')
        if sample.group_key is None:
            raise RuntimeError('training groups require driver-stamped group_key')
        groups.setdefault(sample.group_key, []).append(sample)

    accepted = []
    for key, group in groups.items():
        if len(group) < num_generations:
            logger.info(f'skip group={key}: incomplete')
            continue
        if (len(group) != num_generations
                or {sample.extra.get('trajectory_idx') for sample in group} != set(range(num_generations))):
            raise RuntimeError(f'invalid group members: {key}')
        if len({(sample.policy_version, sample.extra.get('adapter_uri')) for sample in group}) != 1:
            logger.info(f'skip group={key}: mixed_policy')
            continue
        if drop_truncated and any(sample.truncated for sample in group):
            logger.info(f'skip group={key}: truncated')
            continue
        for sample in group:
            tokens, logprobs, mask = sample.response_token_ids, sample.rollout_logprobs, sample.response_loss_mask
            if (not tokens or len(tokens) != len(logprobs) or len(tokens) != len(mask)
                    or not all(math.isfinite(value) for value in logprobs)
                    or any(value not in (0, 1) for value in mask)):
                raise RuntimeError(f'missing or invalid rollout probabilities/mask: {key}')
            if not sample.orm_reward or not all(math.isfinite(value) for value in sample.orm_reward):
                raise RuntimeError(f'missing or invalid environment reward: {key}')
        accepted.append(group)
    return accepted
