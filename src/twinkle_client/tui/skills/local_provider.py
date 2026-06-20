# Copyright (c) Twinkle Contributors. All rights reserved.
"""Local skill provider - loads skill markdown files from the project's skills/ directory."""

from __future__ import annotations

from pathlib import Path

from twinkle_client.tui.skills.base import SkillProvider

# Default: project root's skills/ directory (auto-detected relative to this file)
_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[3] / 'skills'


class LocalSkillProvider(SkillProvider):
    """Loads skill markdown files from a local directory.

    By default, reads from the project's `skills/` folder which contains
    twinkle-training.md and autoresearch.md — the core domain knowledge
    that the agent needs to write correct training scripts.
    """

    def __init__(self, skills_dir: Path | str | None = None):
        self._skills_dir = Path(skills_dir) if skills_dir else _DEFAULT_SKILLS_DIR
        super().__init__(cache_dir=self._skills_dir)

    @property
    def name(self) -> str:
        return 'local'

    async def fetch(self) -> None:
        """No-op for local provider (files are already on disk)."""
        pass

    def _skills_root(self) -> Path:
        return self._skills_dir
