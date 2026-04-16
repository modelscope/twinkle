# OlympiadBench Reward

A family of reward functions for evaluating OlympiadBench math and physics competition problems.

## OlympiadBenchAccuracyReward

Evaluates answer correctness with support for LaTeX normalization, numeric tolerance, and partial matching.

```python
from twinkle.reward import OlympiadBenchAccuracyReward

reward_fn = OlympiadBenchAccuracyReward()
rewards = reward_fn(generated_trajectories, ground_truth_trajectories)
# rewards: List[float], 1.0 for correct, 0.0 for incorrect
```

The reward function:
1. Extracts boxed answers from `\boxed{...}` with nested brace handling
2. Normalizes both prediction and ground truth (LaTeX, units, fractions)
3. Compares via normalized string matching or numeric comparison with tolerance

## OlympiadBenchFormatReward

Validates the structural format of model outputs.

```python
from twinkle.reward import OlympiadBenchFormatReward

reward_fn = OlympiadBenchFormatReward()
rewards = reward_fn(trajectories, ground_truths)
# rewards: List[float], scores based on format quality
```

Scoring criteria:
- Presence of `\boxed{...}` answer
- Answer positioning (should appear near the end)
- Answer uniqueness and consistency

## OlympiadBenchQualityReward

A composite quality reward combining multiple aspects of response quality.

```python
from twinkle.reward import OlympiadBenchQualityReward

reward_fn = OlympiadBenchQualityReward()
rewards = reward_fn(trajectories, ground_truths)
```

Quality dimensions:
- **Reasoning structure**: Detection of step-by-step reasoning patterns
- **Length appropriateness**: Smooth penalty curve for responses that are too short or too long
- **Content uniqueness**: Penalizes repetitive content within the response

> These rewards can be used individually or combined as a composite reward for GRPO training on olympiad-level math and physics problems.
