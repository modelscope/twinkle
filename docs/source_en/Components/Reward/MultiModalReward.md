# MultiModal Reward

Reward function for evaluating multimodal visual question answering (VQA) tasks.

## MultiModalAccuracyReward

Evaluates the correctness of multimodal VQA answers with a fallback to symbolic math verification.

```python
from twinkle.reward import MultiModalAccuracyReward

reward_fn = MultiModalAccuracyReward()
rewards = reward_fn(generated_trajectories, ground_truth_trajectories)
# rewards: List[float], 1.0 for correct, 0.0 for incorrect
```

The reward function:
1. Extracts the model's answer from the completion text
2. Compares with ground truth using exact string matching
3. Falls back to `math_verify` for symbolic expression comparison when string matching fails

> Designed for visual reasoning tasks such as CLEVR, where answers may be numeric, boolean, or short text.
