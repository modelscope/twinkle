# Copyright (c) ModelScope Contributors. All rights reserved.
"""Experimental / not-yet-wired preprocessor components (AUDIT R1).

These modules are kept out of the main :mod:`twinkle_agentic.preprocessor`
namespace because they have no active consumer in the shipped pipeline
(``QualityPreprocessor``) and are not exercised by the cookbook or tests:

- :class:`ScoreFilter` and its scorers (per-round SFT key-round selection). Active
  LLM generation in the framework goes through ``twinkle_agentic.utils.llm_backup``
  instead of the local ``LLMBackend`` abstraction.
- :class:`LLMBackend` / :class:`OpenAIBackend` / :class:`SamplerBackend`, which
  exclusively serve ``ScoreFilter``.

Import explicitly from here if you want to experiment with them, e.g.::

    from twinkle_agentic.preprocessor.experimental import ScoreFilter, SamplerBackend
"""
from .llm_backend import LLMBackend, OpenAIBackend, SamplerBackend  # noqa: F401
from .score_filter import ScoreFilter  # noqa: F401

__all__ = ['ScoreFilter', 'LLMBackend', 'OpenAIBackend', 'SamplerBackend']
