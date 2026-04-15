# GSM8K Reward

Reward functions specifically designed for evaluating GSM8K math problem solutions.

## GSM8KAccuracyReward

Evaluates the correctness of GSM8K answers by extracting boxed or hash-formatted (`####`) answers and performing numeric/string comparison.

```python
from twinkle.reward import GSM8KAccuracyReward

reward_fn = GSM8KAccuracyReward()
rewards = reward_fn(generated_trajectories, ground_truth_trajectories)
# rewards: List[float], 1.0 for correct, 0.0 for incorrect
```

The reward function:
1. Extracts the answer from `\boxed{...}` or `#### ...` format in the model's completion
2. Extracts the ground truth answer from the reference trajectory
3. Performs numeric comparison (with tolerance) or exact string matching

## GSM8KFormatReward

Checks whether the model output contains a properly formatted answer section.

```python
from twinkle.reward import GSM8KFormatReward

reward_fn = GSM8KFormatReward()
rewards = reward_fn(trajectories, ground_truths)
# rewards: List[float], 1.0 if format is valid, 0.0 otherwise
```

> Use GSM8KAccuracyReward and GSM8KFormatReward together as a composite reward for GRPO training on math problem solving tasks.
