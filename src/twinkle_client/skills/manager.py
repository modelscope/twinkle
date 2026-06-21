# Copyright (c) Twinkle Contributors. All rights reserved.
"""Skill manager - orchestrates multiple skill providers and formats for LLM injection."""

from __future__ import annotations

from twinkle.utils.logger import get_logger

from twinkle_client.skills.base import Skill, SkillProvider

logger = get_logger()


class SkillManager:
    """Manages multiple SkillProviders and aggregates their skills.

    The SkillManager is the single entry point for the agent to load skills.
    It supports registering multiple providers (ModelScope, HuggingFace, local, etc.)
    and produces a combined prompt section for LLM injection.

    Usage:
        manager = SkillManager()
        manager.register(ModelScopeSkillProvider())
        manager.register(HuggingFaceSkillProvider())  # future
        await manager.load_all()
        prompt_section = manager.format_for_prompt()
    """

    def __init__(self):
        self._providers: list[SkillProvider] = []
        self._skills: list[Skill] = []

    def register(self, provider: SkillProvider) -> None:
        """Register a skill provider.

        Args:
            provider: An instance of a SkillProvider subclass.
        """
        self._providers.append(provider)
        logger.info(f'Registered skill provider: {provider.name}')

    async def load_all(self) -> list[Skill]:
        """Fetch and load skills from all registered providers.

        Returns:
            Combined list of all skills from all providers.
        """
        self._skills = []
        for provider in self._providers:
            try:
                skills = await provider.get_skills()
                self._skills.extend(skills)
                logger.info(f'Provider [{provider.name}] loaded {len(skills)} skills')
            except Exception as e:
                logger.error(f'Provider [{provider.name}] failed: {e}')
        return self._skills

    @property
    def skills(self) -> list[Skill]:
        """Return all currently loaded skills."""
        return self._skills

    def format_for_prompt(self) -> str:
        """Format all loaded skills into a single text block for LLM system prompt.

        Returns:
            A formatted string containing all skills, ready to be appended
            to the system prompt. Returns empty string if no skills loaded.
        """
        if not self._skills:
            return ''

        sections: list[str] = []
        sections.append('# Available Skills')
        sections.append('')
        sections.append(
            'The following skills provide you with specialized knowledge and capabilities. '
            'Use them to better assist the user.'
        )
        sections.append('')

        for skill in self._skills:
            sections.append(f'## Skill: {skill.name}')
            sections.append(f'(source: {skill.source})')
            sections.append('')
            sections.append(skill.content)
            sections.append('')
            sections.append('---')
            sections.append('')

        return '\n'.join(sections)

    def get_skill_names(self) -> list[str]:
        """Return names of all loaded skills (for logging/debug)."""
        return [s.name for s in self._skills]
