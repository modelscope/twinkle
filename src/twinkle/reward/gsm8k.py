import logging
import multiprocessing as mp
import re
import threading
from typing import Any, Dict, List

from twinkle.data_format import ground_truth, last_assistant_text
from twinkle.reward.base import Reward

logger = logging.getLogger(__name__)


def _extract_last_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...}, handling nested braces."""
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return ''
    start = idx + len('\\boxed{')
    depth = 1
    j = start
    while j < len(text) and depth > 0:
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
        j += 1
    if depth == 0:
        return text[start:j - 1].strip()
    return ''


def _has_boxed(text: str) -> bool:
    """Check whether *text* contains a valid \\boxed{...} (nested-brace aware)."""
    return bool(_extract_last_boxed(text))


def _math_verify_worker_loop(conn, precision, try_extract_without_anchor, tail_chars):
    """Child-process scoring loop: parse+verify one (completion, gold) per request.

    Runs in a dedicated process so the parent can bound each call by wall-clock and
    terminate it on timeout. math_verify's own signal.alarm timeout only fires on the
    main thread and raises ValueError in the background worker threads that rollout
    rewards run on, so it cannot protect this path; the parent's hard timeout can.
    """
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    extraction_config = (
        ExprExtractionConfig(try_extract_without_anchor=try_extract_without_anchor),
        LatexExtractionConfig(),
    )
    while True:
        try:
            completion, gold = conn.recv()
        except (EOFError, OSError):
            break
        try:
            gold_parsed = parse(gold, extraction_config=extraction_config, parsing_timeout=None)
            tail = completion[-tail_chars:] if tail_chars > 0 else completion
            answer_parsed = None
            if tail != completion:
                answer_parsed = parse(tail, extraction_config=extraction_config, parsing_timeout=None)
            if not answer_parsed:
                answer_parsed = parse(completion, extraction_config=extraction_config, parsing_timeout=None)
            if not gold_parsed or not answer_parsed:
                conn.send((True, False))
                continue
            correct = verify(gold_parsed, answer_parsed, float_rounding=precision, timeout_seconds=None)
            conn.send((True, bool(correct)))
        except Exception:
            try:
                conn.send((True, False))
            except (EOFError, OSError):
                break


class _HardTimeoutMathScorer:
    """Persistent subprocess scorer with a thread-safe hard timeout.

    math_verify's signal.alarm timeout raises in worker threads, so the reward used to
    disable it entirely (parsing_timeout=None / timeout_seconds=None), leaving sympy
    unbounded: a hard equivalence check (combsimp / heuristic polynomial GCD) could hang
    a reward worker indefinitely. This scorer runs parse+verify in a dedicated child
    process and bounds every call by wall-clock -- on timeout it terminates the child
    (killing the sympy computation), respawns a fresh worker on the next call, logs, and
    scores 0, matching the existing except->0 semantics. ``spawn`` avoids forking a
    multi-threaded parent; a lock serializes concurrent reward-worker access.
    """

    def __init__(self, timeout, precision, try_extract_without_anchor, tail_chars):
        self._timeout = timeout
        self._precision = precision
        self._try_extract = try_extract_without_anchor
        self._tail_chars = tail_chars
        self._ctx = mp.get_context('spawn')
        self._lock = threading.Lock()
        self._proc = None
        self._conn = None

    def __getstate__(self):
        return {'timeout': self._timeout, 'precision': self._precision,
                'try_extract': self._try_extract, 'tail_chars': self._tail_chars}

    def __setstate__(self, state):
        self.__init__(state['timeout'], state['precision'], state['try_extract'], state['tail_chars'])

    def _ensure_worker(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_math_verify_worker_loop,
            args=(child_conn, self._precision, self._try_extract, self._tail_chars),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()

    def _kill_worker(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
        self._proc = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def score(self, completion, gold):
        with self._lock:
            self._ensure_worker()
            try:
                self._conn.send((completion, gold))
                if self._conn.poll(self._timeout):
                    ok, correct = self._conn.recv()
                    return bool(correct) if ok else False
            except (EOFError, OSError, BrokenPipeError):
                pass
            self._kill_worker()
            logger.warning(
                'math_verify scoring exceeded %.1fs hard timeout (sympy equivalence check); '
                'reward=0 for this completion.', self._timeout)
            return False

    def close(self):
        with self._lock:
            self._kill_worker()


class GSM8KAccuracyReward(Reward):
    """Accuracy reward for GSM8K: checks if the model's answer matches ground truth.

    Extracts the answer from \\boxed{} (preferred) or #### format.
    Returns 1.0 for correct, 0.0 for incorrect.
    """

    @staticmethod
    def extract_answer(completion: str) -> str:
        """Extract the answer from model completion, preferring \\boxed{} over ####."""
        text = completion[-500:] if len(completion) > 500 else completion
        boxed = _extract_last_boxed(text)
        if boxed:
            return boxed.replace(',', '').replace(' ', '').strip()
        matches = re.findall(r'####\s*([\-\d,\.\s]+)', text)
        if matches:
            return matches[-1].replace(',', '').replace(' ', '').strip()
        return ''

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            # Get model completion (last assistant message)
            completion = last_assistant_text(trajectory)

            gt = ground_truth(trajectory)
            gt = '' if gt is None else str(gt)

            predicted = self.extract_answer(completion)

            # Numeric comparison
            correct = False
            if predicted and gt:
                try:
                    correct = abs(float(predicted) - float(gt)) < 1e-5
                except (ValueError, OverflowError):
                    correct = predicted == gt

            rewards.append(1.0 if correct else 0.0)
        return rewards


class MathVerifyAccuracyReward(Reward):
    """Use the same math-verify parsing and equivalence check as AReaL."""

    def __init__(self, *, precision: int = 6, try_extract_without_anchor: bool = True,
                 ground_truth_key: str = 'ground_truth', tail_chars: int = 600,
                 timeout: float = 10.0):
        self.precision = precision
        self.try_extract_without_anchor = try_extract_without_anchor
        self.ground_truth_key = ground_truth_key
        # Final answers live at the END of a natural-language completion.
        # Anchor-less extraction returns the FIRST expression it finds, which
        # on outputs without \boxed{} grabs an intermediate quantity from the
        # reasoning body. Parsing a tail window first is model-agnostic: any
        # model that states its answer last is extracted correctly, while
        # anchored (\boxed{}/$...$) completions are unaffected.
        self.tail_chars = tail_chars
        self.timeout = timeout
        self._scorer = _HardTimeoutMathScorer(timeout, precision, try_extract_without_anchor, tail_chars)

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion = last_assistant_text(trajectory)
            gold = ground_truth(trajectory, self.ground_truth_key)
            gold = '' if gold is None else str(gold)
            try:
                # parse+verify run in a dedicated child process bounded by a hard
                # wall-clock timeout (thread-safe; signal.alarm cannot protect the
                # background reward workers). Timeout/exception -> reward 0.
                correct = self._scorer.score(completion, gold)
                rewards.append(1.0 if correct else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    def metric_payload(
        self,
        trajectories: List[Dict[str, Any]],
        *,
        rewards: List[float],
        **kwargs,
    ) -> Dict[str, float]:
        return {'accuracy_reward': sum(rewards) / len(rewards)}


class GSM8KBrevityReward(Reward):
    """Reward concise completions that contain a parseable final answer."""

    def __init__(self, full_reward_length: int = 300, decay_length: int = 3000):
        self.full_reward_length = full_reward_length
        self.decay_length = decay_length

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion = last_assistant_text(trajectory)
            has_answer = _has_boxed(completion) or bool(re.search(r'####\s*[\-\d,\.]+', completion))
            if not has_answer:
                rewards.append(0.0)
                continue
            excess_length = max(0, len(completion) - self.full_reward_length)
            rewards.append(max(0.0, 1.0 - excess_length / self.decay_length))
        return rewards


class GSM8KAccuracyBrevityReward(Reward):
    """Sum GSM8K answer accuracy and brevity rewards."""

    def __init__(self, accuracy_weight: float = 1.0, brevity_weight: float = 1.0):
        self.accuracy_weight = accuracy_weight
        self.brevity_weight = brevity_weight
        self.accuracy_reward = GSM8KAccuracyReward()
        self.brevity_reward = GSM8KBrevityReward()

    def components(self, trajectories: List[Dict[str, Any]]) -> tuple[list[float], list[float]]:
        return self.accuracy_reward(trajectories), self.brevity_reward(trajectories)

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        accuracy_rewards, brevity_rewards = self.components(trajectories)
        return [
            self.accuracy_weight * accuracy + self.brevity_weight * brevity
            for accuracy, brevity in zip(accuracy_rewards, brevity_rewards)
        ]

    def metric_payload(
        self,
        trajectories: List[Dict[str, Any]],
        *,
        rewards: List[float],
        **kwargs,
    ) -> Dict[str, float]:
        accuracy_rewards, brevity_rewards = self.components(trajectories)
        size = len(trajectories)
        return {
            'total_reward': sum(rewards) / size,
            'accuracy_reward': sum(accuracy_rewards) / size,
            'brevity_reward': sum(brevity_rewards) / size,
        }


class GSM8KFormatReward(Reward):
    """Format reward: checks if output contains \\boxed{} or #### answer format.

    Returns 1.0 if a valid answer format is present, 0.0 otherwise.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion = last_assistant_text(trajectory)
            has_answer = bool(_has_boxed(completion) or re.search(r'####\s*[\-\d,\.]+', completion))
            rewards.append(1.0 if has_answer else 0.0)
        return rewards
