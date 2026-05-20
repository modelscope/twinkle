import re
import string
from collections import Counter
from typing import Any, Dict, List, Tuple

from twinkle.reward import Reward

_BOXED_MARKER = '\\boxed{'


def _extract_final_answer(completion: str) -> str:
    if not completion:
        return ''
    out = ''
    idx = 0
    while True:
        i = completion.find(_BOXED_MARKER, idx)
        if i == -1:
            break
        j = i + len(_BOXED_MARKER)
        depth = 1
        while j < len(completion) and depth > 0:
            c = completion[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            j += 1
        if depth == 0:
            out = completion[i + len(_BOXED_MARKER):j - 1].strip()
            idx = j
        else:
            # Unbalanced trailing marker — stop, keep last good match.
            break
    return out


def _last_assistant_text(traj: Dict[str, Any]) -> str:
    for msg in reversed(traj.get('messages', [])):
        if msg.get('role') != 'assistant':
            continue
        content = msg.get('content') or ''
        if isinstance(content, str):
            return content
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text')
    return ''


def _stem(tok: str) -> str:
    from nltk.stem import PorterStemmer
    return PorterStemmer().stem(tok) if len(tok) >= 4 and tok.isalpha() else tok


def _normalize_answer(s: str) -> str:
    s = (s or '').lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(_stem(t) for t in s.split())


def _f1_score(prediction: str, gold: str) -> Tuple[float, float]:
    filler_tokens: frozenset = frozenset([
        'long',
        'tall',
        'high',
        'wide',
        'deep',
        'heavy',
        'old',
        'large',
        'small',
        'big',
        'short',
        'away',
        'ago',
        'approximately',
        'about',
        'around',
        'over',
        'under',
        'below',
        'above',
        'total',
        'roughly',
        'nearly',
        'almost',
        'exactly',
    ])
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        em = float(pred_tokens == gold_tokens)
        return em, em
    em = float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, em
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    f1 = 2 * p * r / (p + r)

    pred_set, gold_set = set(pred_tokens), set(gold_tokens)
    if gold_set < pred_set:
        extras = pred_set - gold_set
        if all(t.isdigit() or t in filler_tokens for t in extras):
            return 1.0, em
    if pred_set < gold_set:
        missing = gold_set - pred_set
        if all(t in filler_tokens for t in missing):
            return 1.0, em
    return f1, em


class F1Reward(Reward):

    def __init__(self, answer_pattern=None):
        if isinstance(answer_pattern, str):
            answer_pattern = re.compile(answer_pattern)
        self._answer_pattern = answer_pattern

    def _extract(self, completion: str) -> str:
        balanced = _extract_final_answer(completion)
        if balanced:
            return balanced
        if self._answer_pattern is None:
            return ''
        matches = self._answer_pattern.findall(completion or '')
        if not matches:
            return ''
        last = matches[-1]
        if isinstance(last, tuple):
            last = last[0] if last else ''
        return (last or '').strip()

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            golds = [val for key, val in traj.get('user_data', []) or [] if key == 'ground_truth' and val]
            pred = self._extract(_last_assistant_text(traj))
            if golds:
                f1 = max(_f1_score(pred, g)[0] for g in golds)
            else:
                f1, _ = _f1_score(pred, '')
            rewards.append(f1)
        return rewards


class CoTReward(Reward):
    _STEP_LINE_RE = re.compile(r'(?im)^\s*step\s*(\d+)\s*[.:]')
    _HAS_BOXED_RE = re.compile(r'\\boxed\{')

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards: List[float] = []
        for t in trajectories:
            msgs = t.get('messages', []) or []

            # Newline-joined so ``^`` line anchors work even when
            # multiple assistant turns exist.
            assistant_text = '\n'.join(
                m.get('content', '') or '' for m in msgs
                if m.get('role') == 'assistant' and isinstance(m.get('content'), str))

            if not self._HAS_BOXED_RE.search(assistant_text):
                rewards.append(0.0)
                continue

            steps: set = set()
            for match in self._STEP_LINE_RE.finditer(assistant_text):
                try:
                    steps.add(int(match.group(1)))
                except ValueError:
                    continue

            n = len(steps)
            if 3 <= n <= 6:
                rewards.append(0.3)
            elif n >= 7:
                rewards.append(0.1)
            else:
                rewards.append(0.0)

        return rewards


class ToolExploreReward(Reward):

    def __init__(self, f1_threshold: float = 0.5, answer_pattern=None):
        if isinstance(answer_pattern, str):
            answer_pattern = re.compile(answer_pattern)
        self._answer_pattern = answer_pattern
        self._f1_threshold = float(f1_threshold)

    def _extract(self, completion: str) -> str:
        balanced = _extract_final_answer(completion)
        if balanced:
            return balanced
        if self._answer_pattern is None:
            return ''
        matches = self._answer_pattern.findall(completion or '')
        if not matches:
            return ''
        last = matches[-1]
        if isinstance(last, tuple):
            last = last[0] if last else ''
        return (last or '').strip()

    def _trajectory_f1(self, traj: Dict[str, Any]) -> float:
        golds = [val for key, val in traj.get('user_data', []) or [] if key == 'ground_truth' and val]
        pred = self._extract(_last_assistant_text(traj))
        if golds:
            return max(_f1_score(pred, g)[0] for g in golds)
        f1, _ = _f1_score(pred, '')
        return f1

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards: List[float] = []
        for t in trajectories:
            if self._trajectory_f1(t) < self._f1_threshold:
                rewards.append(0.0)
                continue

            msgs = t.get('messages', []) or []
            n_msgs = len(msgs)
            n_success = 0
            for i, m in enumerate(msgs):
                if m.get('role') != 'assistant' or not m.get('tool_calls'):
                    continue
                # Scan subsequent consecutive ``tool`` messages; this
                # assistant turn counts as successful if any of them
                # came back non-empty and non-ERROR.
                j = i + 1
                turn_ok = False
                while j < n_msgs and msgs[j].get('role') == 'tool':
                    content = msgs[j].get('content') or ''
                    text = content if isinstance(content, str) else str(content)
                    if text.strip() and not text.lstrip().startswith('ERROR'):
                        turn_ok = True
                        break
                    j += 1
                if turn_ok:
                    n_success += 1
            if n_success == 0:
                rewards.append(0.0)
            elif n_success <= 3:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        return rewards
