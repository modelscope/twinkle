# Copyright (c) Twinkle Contributors. All rights reserved.
"""Local skill provider - loads skill markdown files from user's local skills directory."""

from __future__ import annotations

from pathlib import Path

from twinkle_client.skills.base import SkillProvider

# Default: user-local skills directory (user drops .md files here)
_DEFAULT_SKILLS_DIR = Path.home() / '.cache' / 'twinkle' / 'auto' / 'skills' / 'local'


class LocalSkillProvider(SkillProvider):
    """Loads skill markdown files from a local directory.

    By default, reads from ~/.cache/twinkle/auto/skills/local/.
    Users can place custom .md skill files there to extend the agent's
    domain knowledge without modifying the codebase.
    """

    def __init__(self, skills_dir: Path | str | None = None):
        self._skills_dir = Path(skills_dir) if skills_dir else _DEFAULT_SKILLS_DIR
        super().__init__(cache_dir=self._skills_dir)

    @property
    def name(self) -> str:
        return 'local'

    async def fetch(self) -> None:
        """Ensure the local skills directory exists."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)

    def _skills_root(self) -> Path:
        return self._skills_dir
