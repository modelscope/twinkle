# GroupAdmissionPolicy

GRPO groups with identical rewards produce zero centered advantages. With
dense or composite rewards, a non-zero difference can also be smaller than the
reward's meaningful resolution. Normalizing that difference may amplify noise
into a strong relative signal. `GroupAdmissionPolicy` separates this quality
decision from sampling infrastructure and always admits or rejects a complete
prompt group. Bounded replacement sampling can then recover useful groups
without an unbounded retry loop.

This component is intentionally imported from its dedicated submodule rather
than re-exported by `twinkle.advantage`: it applies to group-relative methods
such as GRPO and DAPO-style training, not to every advantage estimator.

## Signals

The policy evaluates two optional conditions:

- The reward condition checks population reward standard deviation, reward
  range, and the number of non-trivial centered advantages.
  `min_reward_range` is the application-defined reward resolution.
- The optional near-duplicate condition checks whether string representations
  are excessively similar. Its token n-gram Jaccard default is a
  dependency-free lexical baseline, not a semantic-equivalence test. Callers
  can inject a custom scorer when needed.

All thresholds are disabled by default, so the default policy admits every
valid group and does not change existing training behavior.

```python
from twinkle.advantage.group_admission import (
    GroupAdmissionConfig,
    GroupAdmissionPolicy,
)

policy = GroupAdmissionPolicy(
    GroupAdmissionConfig(
        min_reward_std=1e-6,
        min_reward_range=0.02,
        min_nontrivial_advantages=2,
        max_mean_pairwise_similarity=0.95,
    )
)

decision = policy.evaluate(
    rewards=[0.0, 1.0, 0.0, 1.0],
    completions=["answer a", "answer b", "answer c", "answer d"],
)
if decision.admitted:
    train_complete_group()
```

Rejected groups expose one mutually exclusive `primary_rejection_reason`:

- `exact_dead`: the reward gate rejected an exactly equal-reward group;
- `near_tie`: rewards differ, but fail the configured dispersion or resolution
  requirement;
- `redundant`: the reward signal passed, but the near-duplicate gate rejected
  the group.

External precondition failures take precedence and use `external`. Low-level
gate reasons remain available in `decision.reasons`.

## Budget-aware resampling

`SamplingBudgetController` tracks the effective-group rate with an exponential
moving average and estimates how many replacement groups are needed. Hard caps
can be set for extra groups, resampling rounds, generated samples, and generated
tokens.

```python
from twinkle.advantage.group_admission import (
    SamplingBudgetConfig,
    SamplingBudgetController,
    SamplingBudgetState,
)

controller = SamplingBudgetController(
    SamplingBudgetConfig(
        max_extra_groups=16,
        max_extra_groups_per_round=4,
        max_resample_rounds=2,
        max_total_samples=128,
    )
)
state = SamplingBudgetState()

# `decisions` contains one decision per complete group from the current round.
controller.observe_round(
    state,
    decisions,
    num_generations=4,
    generated_tokens=round_output_tokens,
)
plan = controller.plan_resampling(
    state,
    target_groups=8,
    num_generations=4,
)
sample_more_complete_groups(plan.extra_groups)
```

Resampling must use the same rollout policy snapshot as the initial candidates.
The controller only plans work; the caller remains responsible for policy
version and staleness checks.

For a complete GSM8K rollout, admission, bounded-resampling, GRPO advantage,
and optimizer-step loop, run the end-to-end
[`cookbook/rl/grpo/group_admission.py`](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/grpo/group_admission.py)
example. It follows the official `short_math_grpo.py` setup and keeps one
rollout policy snapshot across initial and replacement groups:

```bash
sh cookbook/rl/grpo/group_admission.sh --max-steps 2
```

Exact-dead and near-tie outcomes are stochastic in a real rollout; their
deterministic branch coverage remains in `tests/advantage/test_group_admission.py`.

Use `group_admission_metrics(decisions)` to report effective-group rate, the
orthogonal reward/near-duplicate rejection counts, and the mutually exclusive
`exact_dead`, `near_tie`, and `redundant` counts. Derive ratios at the caller
from these raw counts so merged/distributed metrics retain a single
denominator.

## Challenger integration

`twinkle_agentic.challenger.Challenger` already refills its batch continuously
when a completed unit contributes no trainable group. Pass the same policy to
`AgenticChallenger` to apply resolution-aware admission at that existing
group-atomic boundary:

```python
challenger = AgenticChallenger(
    backend,
    envs=envs,
    group_admission_policy=policy,
)
```

The original equal-reward filter remains active when no policy is supplied.
When the optional near-duplicate gate is enabled, Challenger uses each
trajectory's final assistant text as its representation. The existing
continuous refill and `max_empty_rounds` behavior are unchanged. The separate
`SamplingBudgetController` remains available to sampling loops that need hard
sample or token budgets; Challenger does not silently relax a gate.

## Non-goals

The policy does not relax a configured gate when a budget is exhausted. The
caller should skip or fail the task according to its batch semantics. It also
does not change GRPO loss or normalized-advantage computation.
