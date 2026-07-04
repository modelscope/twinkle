# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Sequence

from twinkle_agentic.utils.llm_backup import llm_backup

if TYPE_CHECKING:
    from twinkle.data_format import SamplingParams, Trajectory  # noqa: F401
    from twinkle.sampler.base import Sampler  # noqa: F401


DEFAULT_USER_PROMPT_TEMPLATE = """\
Compress the following text as much as possible while preserving all key information.

## Target length
HARD CEILING: {budget} chars. If core facts fit in far fewer chars, output fewer.

## Text
{text}"""


class Condenser:
    """Base condenser with progressive distillation via llm_backup.

    Subclasses customize compression behavior by providing their own
    ``system_prompt``, ``user_prompt_template``, and ``lora_path``.
    The shared ``_sample`` method (decorated with ``@llm_backup``) handles
    the student-teacher routing transparently.

    Teacher is a global OpenAI-compatible API configured via env vars:
        - LLM_BACKUP_MODEL: teacher model name
        - LLM_BACKUP_API_KEY: API key
        - LLM_BACKUP_BASE_URL: API endpoint

    Args:
        sampler: Student model sampler (local inference, shared across types).
        compression_ratio: Target compression factor (> 1).
        model_path: Model identifier.
        sampling_params: Default sampling params.
        system_prompt: System prompt for this condenser type.
        user_prompt_template: User prompt template. Must contain
            ``{budget}`` and ``{text}``. May contain ``{query}``.
        min_budget_chars: Floor for the character budget in the prompt.
        template: Optional :class:`Template` for special token stripping.
        lora_path: LoRA adapter path specific to this condenser type.
            Each subclass can use a different LoRA for its task.
    """

    def __init__(
        self,
        sampler: Sampler,
        compression_ratio: float = 2.0,
        *,
        model_path: str = '',
        sampling_params: SamplingParams | None = None,
        system_prompt: str = 'You are a text compression assistant.',
        user_prompt_template: str | None = None,
        min_budget_chars: int = 250,
        template: Any | None = None,
        lora_path: str | None = None,
    ):
        if sampler is None:
            raise ValueError('sampler is required')
        if compression_ratio <= 1.0:
            raise ValueError(f'compression_ratio must be > 1, got {compression_ratio}')
        if min_budget_chars < 1:
            raise ValueError(f'min_budget_chars must be >= 1, got {min_budget_chars}')

        tpl = user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
        if '{budget}' not in tpl or '{text}' not in tpl:
            raise ValueError('user_prompt_template must contain both {budget} and {text}')

        self.model_path = model_path
        self.sampler = sampler
        self.compression_ratio = float(compression_ratio)
        self.sampling_params = sampling_params
        self.system_prompt = system_prompt
        self.user_prompt_template = tpl
        self.min_budget_chars = int(min_budget_chars)
        self.template = template
        self.lora_path = lora_path if lora_path else None
        self._special_tokens_cache: tuple[str, ...] | None = None

    # ------------------------------------------------------------------
    # public entry point (pre/post processing, NOT decorated)
    # ------------------------------------------------------------------
    def __call__(self, text: str, system: str = None, query: str = None,
                 sampling_params: Any = None) -> str:
        system = system or self.system_prompt
        budget = max(self.min_budget_chars, math.ceil(len(text) / self.compression_ratio))
        if budget >= len(text):
            return text
        trajectory = self._make_trajectory(system, self.user_prompt_template, text, budget, query)
        sp = sampling_params or self.sampling_params or self._default_sampling_params(budget)

        raw = self._sample(trajectory=trajectory, sampling_params=sp, query=query)

        result = self._postprocess(raw, text, self._get_special_tokens())
        return result if result is not None else text

    # ------------------------------------------------------------------
    # student sampling (decorated with llm_backup)
    # ------------------------------------------------------------------
    @llm_backup(key_params=["query"])
    def _sample(self, trajectory, sampling_params, query: str = None) -> str:
        """Student model: trajectory + sampling_params -> raw text."""
        sample_kwargs: dict[str, Any] = {'sampling_params': sampling_params}
        if self.lora_path is None:
            sample_kwargs['use_base_model'] = True
        else:
            sample_kwargs['adapter_path'] = self.lora_path
        responses = self.sampler.sample([trajectory], **sample_kwargs)
        return self._decoded(list(responses)[0]) if responses else ''

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _get_special_tokens(self) -> tuple[str, ...]:
        if self._special_tokens_cache is not None:
            return self._special_tokens_cache
        tpl = self.template or getattr(self.sampler, 'template', None)
        tokenizer = getattr(tpl, 'tokenizer', None) if tpl is not None else None
        tokens: list[str] = []
        if tokenizer is not None:
            extras = getattr(tokenizer, 'all_special_tokens', None) or []
            if extras:
                tokens.extend(t for t in extras if isinstance(t, str) and t and not t.isspace())
            else:
                for attr in ('eos_token', 'pad_token', 'bos_token'):
                    t = getattr(tokenizer, attr, None)
                    if isinstance(t, str) and t:
                        tokens.append(t)
        self._special_tokens_cache = tuple(dict.fromkeys(tokens))
        return self._special_tokens_cache


    # ------------------------------------------------------------------
    # static helpers
    # ------------------------------------------------------------------
    _CODE_FENCE_RE = re.compile(r'^```[a-zA-Z]*\s*\n(.*?)\n```\s*$', re.DOTALL)

    @staticmethod
    def _make_trajectory(system: str, user_template: str, text: str,
                         budget: int, query: str | None = None) -> dict:
        """Build a trajectory dict for sampler / API."""
        user = user_template.replace('{budget}', str(budget))
        user = user.replace('{text}', text)
        if '{query}' in user:
            q_text = (
                query.strip() if isinstance(query, str) and query and query.strip() else
                '(no explicit query; compress by general salience)')
            user = user.replace('{query}', q_text)
        return {
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        }

    @staticmethod
    def _default_sampling_params(budget: int):
        from twinkle.data_format.sampling import SamplingParams
        max_new = max(512, budget * 3 + 128)
        return SamplingParams(temperature=0.0, max_tokens=max_new)

    @staticmethod
    def _postprocess(raw: str, original: str, special_tokens: tuple[str, ...]) -> str | None:
        text = Condenser._strip_special_tokens(
            Condenser._strip_code_fences(raw), special_tokens).strip()
        if not text or not Condenser._has_alnum(text):
            return None
        if len(text) >= len(original):
            return None
        return text

    @staticmethod
    def _decoded(response: Any) -> str:
        seqs = getattr(response, 'sequences', None) or []
        if not seqs:
            return ''
        return getattr(seqs[0], 'decoded', None) or ''

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        m = Condenser._CODE_FENCE_RE.match(stripped)
        return m.group(1) if m else text

    @staticmethod
    def _strip_special_tokens(text: str, tokens: Sequence[str]) -> str:
        for tok in tokens:
            if tok and tok in text:
                text = text.replace(tok, '')
        return text

    @staticmethod
    def _has_alnum(text: str) -> bool:
        return any(ch.isalnum() for ch in text)
