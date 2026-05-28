# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger

from .llm_backend import LLMBackend, OpenAIBackend

logger = get_logger(only_local_master=False)

_REFINE_SYSTEM_PROMPT = """\
You are an expert response quality optimizer. You will be given a conversation context \
and must produce the ideal assistant response.

Requirements:
1. Correctness: The answer must be logically sound with no factual errors.
2. Conciseness: Remove redundant reasoning, filler phrases, and unnecessary repetition. \
Every sentence should carry new information.
3. Completeness: Cover all aspects of the user's question without omitting key points.
4. Structure: Use clear organization (numbered steps, code blocks, formulas) when appropriate.
5. Length: Response length should be proportional to question complexity — \
short questions get short answers, complex ones get detailed answers.

Output format:
- Return ONLY the assistant's response content. Do not include any meta-commentary.\
"""

_INTENT_PROMPT_SUFFIX = {
    'code': (
        '\nFocus: This round is about CODE. '
        'Ensure the code is correct, complete, runnable, and well-commented. '
        'Fix any bugs in the original. Use proper formatting with language-tagged fenced blocks.'
    ),
    'math': (
        '\nFocus: This round is about MATH. '
        'Show derivation steps clearly with proper LaTeX notation. '
        'Verify the final answer by substitution or sanity check.'
    ),
    'complex_logic': (
        '\nFocus: This round requires COMPLEX REASONING. '
        'Present a clean logical chain without backtracking. '
        'Number each reasoning step. State assumptions explicitly.'
    ),
    'user_dissatisfaction': (
        '\nFocus: The user was DISSATISFIED with the previous response. '
        'Address the root cause of dissatisfaction directly. '
        'Acknowledge the issue and provide a substantially improved answer.'
    ),
    'tool_call': (
        '\nFocus: This round involves TOOL CALLS. '
        'Ensure tool call arguments are correct and the synthesis of tool results is accurate. '
        'Present the final answer clearly based on tool outputs.'
    ),
}


def _call_model(
    backend: LLMBackend,
    context_messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    intent: str = '',
) -> Optional[Dict[str, str]]:
    """Call the model and return {'content': ..., 'reasoning_content': ...}."""
    system_prompt = _REFINE_SYSTEM_PROMPT + _INTENT_PROMPT_SUFFIX.get(intent, '')
    messages = [{'role': 'system', 'content': system_prompt}] + context_messages

    choices = backend.chat(messages, temperature=temperature, max_tokens=max_tokens)
    if not choices:
        return None

    content = choices[0].get('content') or ''
    reasoning = choices[0].get('reasoning_content') or ''

    if not content.strip():
        return None

    return {'content': content, 'reasoning_content': reasoning}


def _refine_round(
    backend: LLMBackend,
    messages: List[Dict[str, Any]],
    assistant_idx: int,
    temperature: float,
    max_tokens: int,
    intent: str = '',
) -> Optional[Dict[str, str]]:
    """Refine a single key round's assistant response."""
    if assistant_idx >= len(messages) or assistant_idx < 1:
        return None

    asst_msg = messages[assistant_idx]
    if not isinstance(asst_msg, dict) or asst_msg.get('role') != 'assistant':
        return None

    context = messages[:assistant_idx]
    if not context:
        return None

    return _call_model(backend, context, temperature, max_tokens, intent)


class ResponseRefiner(Preprocessor):
    """Re-annotate key rounds with a strong model for highest quality responses.

    For each key round (from IntentClassifier/IFDFilter), sends the context
    to an OpenAI-compatible API and replaces the assistant response with a
    refined version containing both reasoning_content and content.

    Rows without key_rounds are discarded.
    If refinement fails for a round, the original response is kept.
    """

    def __init__(
        self,
        backend: LLMBackend = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        max_workers: int = 8,
        # Legacy params (used to create OpenAIBackend if backend is None)
        api_endpoint: str = '',
        model: str = 'default',
        api_key: str = '',
    ):
        super().__init__()
        if backend is not None:
            self._backend = backend
        else:
            self._backend = OpenAIBackend(
                endpoint=api_endpoint, model=model, api_key=api_key, timeout=180.0)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_workers = max_workers

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.refine(rows)
        return self.map_row_to_col(rows)

    def refine(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Refine key round responses in parallel."""
        if not rows:
            return rows

        # Collect tasks: (row_idx, round_idx, assistant_idx, messages, intent)
        tasks: List[Tuple[int, int, int, List[Dict[str, Any]], str]] = []
        for ri, row in enumerate(rows):
            user_data = row.get('user_data')
            if not isinstance(user_data, dict):
                continue
            key_rounds = user_data.get('key_rounds')
            if not isinstance(key_rounds, list) or not key_rounds:
                continue
            messages = row.get('messages') or []
            intents = user_data.get('intents') or {}
            for rnd_idx, asst_idx in enumerate(key_rounds):
                tasks.append((ri, rnd_idx, asst_idx, messages, intents.get(asst_idx, '')))

        if not tasks:
            # No key rounds anywhere → drop all
            logger.info('[ResponseRefiner] no key rounds found, dropping all rows')
            return []

        # Parallel refinement
        results: Dict[Tuple[int, int], Optional[Dict[str, str]]] = {}
        n_workers = min(self._max_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_key = {
                pool.submit(
                    _refine_round, self._backend,
                    msgs, asst_idx, self._temperature, self._max_tokens, intent,
                ): (ri, rnd_idx)
                for ri, rnd_idx, asst_idx, msgs, intent in tasks
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.warning(f'[ResponseRefiner] round {key} failed: {e}')
                    results[key] = None

        # Apply refinements
        out = []
        n_refined = 0
        n_dropped = 0

        for ri, row in enumerate(rows):
            user_data = row.get('user_data')
            if not isinstance(user_data, dict):
                n_dropped += 1
                continue
            key_rounds = user_data.get('key_rounds')
            if not isinstance(key_rounds, list) or not key_rounds:
                n_dropped += 1
                continue

            messages = list(row.get('messages') or [])
            modified = False

            for rnd_idx, asst_idx in enumerate(key_rounds):
                result = results.get((ri, rnd_idx))
                if result is None:
                    continue

                if asst_idx >= len(messages):
                    continue

                # Replace assistant content
                old_msg = messages[asst_idx]
                new_msg = dict(old_msg)
                new_msg['content'] = result['content']
                if result['reasoning_content']:
                    new_msg['reasoning_content'] = result['reasoning_content']
                elif 'reasoning_content' in new_msg:
                    del new_msg['reasoning_content']
                messages[asst_idx] = new_msg
                modified = True
                n_refined += 1

            row = dict(row, messages=messages)
            if modified:
                row['user_data'] = dict(user_data, refined=True)
            out.append(row)

        logger.info(
            f'[ResponseRefiner] refined {n_refined} rounds, '
            f'dropped {n_dropped} rows without key_rounds, '
            f'output {len(out)} rows')
        return out
