# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reward functions for OlympiadBench math/physics problems."""
import re
from typing import Any, Dict, List

from twinkle.reward.base import Reward


class OlympiadBenchAccuracyReward(Reward):
    """Accuracy reward for OlympiadBench: checks if model's answer matches ground truth.

    Extracts the answer from \\boxed{} format.
    Handles multiple answers and numeric comparison with tolerance.
    Returns 1.0 for correct, 0.0 for incorrect.
    """

    @staticmethod
    def extract_boxed_answers(text: str) -> List[str]:
        """Extract all answers from \\boxed{} in text."""
        # Look in the last 1000 chars for efficiency
        text = text[-1000:] if len(text) > 1000 else text
        # Find all boxed answers
        matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        if matches:
            # Clean and return all matches
            return [m.replace(',', '').replace(' ', '').strip() for m in matches]
        return []

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer string for comparison."""
        # Remove common units and whitespace
        answer = re.sub(r'\s+', '', answer)
        answer = re.sub(r'(cm|mm|m|kg|g|s|°|度|米|千克|克|秒)', '', answer, flags=re.IGNORECASE)
        return answer.strip()

    @staticmethod
    def is_numeric_match(pred: str, gt: str, tolerance: float = 1e-5) -> bool:
        """Check if two values match numerically."""
        try:
            pred_val = float(pred)
            gt_val = float(gt)
            return abs(pred_val - gt_val) < tolerance * max(abs(gt_val), 1.0)
        except (ValueError, OverflowError):
            return False

    def compare_answers(self, predicted: List[str], ground_truth: str, is_multiple: bool = False) -> bool:
        """Compare predicted answers with ground truth."""
        if not predicted or not ground_truth:
            return False

        # Parse ground truth (may have multiple answers separated by comma)
        gt_parts = [self.normalize_answer(g.strip()) for g in ground_truth.split(',')]

        # For single answer, just check the last predicted
        if not is_multiple:
            pred = self.normalize_answer(predicted[-1])
            gt = gt_parts[0] if gt_parts else ''
            # Try numeric comparison first
            if self.is_numeric_match(pred, gt):
                return True
            return pred == gt

        # For multiple answers, check if all gt answers are found
        pred_normalized = [self.normalize_answer(p) for p in predicted]
        for gt in gt_parts:
            found = False
            for pred in pred_normalized:
                if self.is_numeric_match(pred, gt) or pred == gt:
                    found = True
                    break
            if not found:
                return False
        return True

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            # Get model completion (last assistant message)
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            # Get ground truth and metadata from user_data
            user_data = trajectory.get('user_data', [])
            gt = ''
            is_multiple = False
            for item in user_data:
                if item[0] == 'ground_truth':
                    gt = item[1]
                elif item[0] == 'is_multiple_answer':
                    is_multiple = item[1]

            # Extract predicted answers
            predicted = self.extract_boxed_answers(completion)

            # Compare
            correct = self.compare_answers(predicted, gt, is_multiple)
            rewards.append(1.0 if correct else 0.0)

        return rewards


class OlympiadBenchFormatReward(Reward):
    """Format reward: checks if output contains \\boxed{} answer format.

    Returns 1.0 if a valid boxed answer is present, 0.0 otherwise.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_boxed = bool(re.search(r'\\boxed\{[^}]+\}', completion))
            rewards.append(1.0 if has_boxed else 0.0)
        return rewards


class OlympiadBenchReasoningReward(Reward):
    """Reasoning reward: checks if output contains step-by-step reasoning.

    Returns a score based on:
    - Presence of multiple reasoning steps
    - Use of mathematical notation
    - Logical structure (因为/所以, therefore/because, etc.)
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            score = 0.0

            # Check for step indicators
            step_patterns = [
                r'第[一二三四五六七八九十\d]+步',  # Chinese steps
                r'Step\s*\d+',  # English steps
                r'\d+\.',  # Numbered list
                r'首先|然后|最后|因此|所以',  # Chinese connectors
                r'First|Then|Finally|Therefore|Hence',  # English connectors
            ]
            for pattern in step_patterns:
                if re.search(pattern, completion, re.IGNORECASE):
                    score += 0.2
                    break

            # Check for mathematical notation
            math_patterns = [r'=', r'\+', r'-', r'\*', r'/', r'\^', r'\\frac', r'\\sqrt']
            math_count = sum(1 for p in math_patterns if re.search(p, completion))
            score += min(0.3, math_count * 0.1)

            # Check completion length (longer usually means more reasoning)
            if len(completion) > 200:
                score += 0.2
            if len(completion) > 500:
                score += 0.1

            # Check for boxed answer at the end
            if re.search(r'\\boxed\{[^}]+\}\s*$', completion):
                score += 0.2

            rewards.append(min(1.0, score))

        return rewards
