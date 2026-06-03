# Copyright (c) ModelScope Contributors. All rights reserved.
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from twinkle.preprocessor import Preprocessor

from .llm_backend import LLMBackend, OpenAIBackend

_DEFAULT_SYSTEM_PROMPT = (
    'You are a strict trajectory quality judge. '
    'Given a multi-turn conversation, decide whether the assistant response is high-quality. '
    'Criteria: factual accuracy, helpfulness, coherence, and completeness. '
    'Reply with EXACTLY one word: PASS or FAIL.'
)

_DEFAULT_TIMEOUT = 120.0


class JudgeSource:
    """One LLM judge backend."""

    def __init__(
        self,
        backend: LLMBackend = None,
        api_endpoint: str = '',
        model: str = 'default',
        api_key: str = '',
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        if backend is not None:
            self.backend = backend
        else:
            self.backend = OpenAIBackend(
                endpoint=api_endpoint, model=model, api_key=api_key, timeout=timeout)


def _build_judge_messages(
    messages: List[Dict[str, Any]],
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Wrap the trajectory into a judge prompt."""
    conversation_text = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role', 'unknown')
        content = (m.get('content') or '').strip()
        if content:
            conversation_text.append(f'[{role}]: {content}')
    joined = '\n'.join(conversation_text)
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f'Please judge the following conversation:\n\n{joined}'},
    ]


def _vote_one(
    source: JudgeSource,
    judge_messages: List[Dict[str, Any]],
    temperature: float,
) -> Optional[bool]:
    """Send one judge request. Returns True=PASS, False=FAIL, None=error."""
    choices = source.backend.chat(judge_messages, temperature=temperature, max_tokens=16)
    if not choices:
        return None
    text = choices[0].get('content', '').strip().upper()
    if 'PASS' in text:
        return True
    if 'FAIL' in text:
        return False
    return None


class MajorityVoteFilter(Preprocessor):
    """Multi-judge majority vote filter.

    Sends each trajectory to N independent OpenAI-compatible judges.
    Keeps the row only if the majority votes PASS.
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        pass_threshold: float = 0.5,
        temperature: float = 0.0,
        max_workers: int = 8,
        skip_on_error: bool = True,
    ):
        """
        Args:
            sources: List of judge source configs, each dict has keys:
                     api_endpoint (required), model, api_key, timeout.
            system_prompt: Evaluation prompt sent to each judge.
            pass_threshold: Fraction of votes needed to pass (> threshold keeps).
            temperature: Sampling temperature for judges.
            max_workers: Thread pool size for concurrent API calls.
            skip_on_error: If True, keep rows where all judges failed.
        """
        if not sources:
            raise ValueError('At least one judge source is required')
        self._sources = [JudgeSource(**s) for s in sources]
        self._system_prompt = system_prompt
        self._pass_threshold = pass_threshold
        self._temperature = temperature
        self._max_workers = max_workers
        self._skip_on_error = skip_on_error

    def _judge_row(self, messages: List[Dict[str, Any]]) -> Optional[bool]:
        """Collect votes from all sources for one row. Returns pass/fail/None."""
        judge_msgs = _build_judge_messages(messages, self._system_prompt)

        votes: List[bool] = []
        with ThreadPoolExecutor(max_workers=len(self._sources)) as pool:
            futures = [
                pool.submit(_vote_one, src, judge_msgs, self._temperature)
                for src in self._sources
            ]
            for f in as_completed(futures):
                result = f.result()
                if result is not None:
                    votes.append(result)

        if not votes:
            return None
        return sum(votes) / len(votes) > self._pass_threshold

    def __call__(self, rows) -> List[Dict[str, Any]]:
        """Filter rows by majority vote across configured judge sources."""
        if not rows:
            return rows

        results: Dict[int, Optional[bool]] = {}
        n_workers = min(self._max_workers, len(rows))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(self._judge_row, row.get('messages') or []): i
                for i, row in enumerate(rows)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None

        out = []
        for i, row in enumerate(rows):
            verdict = results.get(i)
            if verdict is None:
                if self._skip_on_error:
                    out.append(row)
                continue
            if verdict:
                out.append(row)
        return out
