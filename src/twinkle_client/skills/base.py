# Copyright (c) Twinkle Contributors. All rights reserved.
"""Base class for skill providers.

A SkillProvider is responsible for fetching skill files (typically markdown)
from a remote source (Git repo, API, local directory, etc.) and making their
content available for injection into the agent's system prompt.

To create a new provider, subclass SkillProvider and implement:
- `name` property: human-readable provider name
- `fetch()`: download/update skill files to local cache
- `load_skills()`: read cached files and return list of Skill objects
"""

from __future__ import annotations

import dataclasses
from twinkle.utils.logger import get_logger
from abc import ABC, abstractmethod
from pathlib import Path

logger = get_logger()

# File stems to skip when scanning for skill markdown files
_SKIP_STEMS = frozenset({'license', 'readme', 'contributing', 'changelog'})


@dataclasses.dataclass
class Skill:
    """A single skill loaded from a provider.

    Attributes:
        name: Short identifier for the skill (typically filename without extension).
        content: Full markdown content of the skill.
        source: Provider name + relative path for traceability.
    """

    name: str
    content: str
    source: str


class SkillProvider(ABC):
    """Abstract base class for skill providers.

    Subclass this to add new skill sources (e.g., HuggingFace, local files).
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the provider.

        Args:
            cache_dir: Local directory to cache fetched skill files.
                       If None, uses ~/.cache/twinkle/auto/skills/<provider_name>
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'twinkle' / 'auto' / 'skills' / self.name
        self.cache_dir = cache_dir

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this provider (used as cache subdirectory)."""
        ...

    @abstractmethod
    async def fetch(self) -> None:
        """Fetch or update skill files from the remote source.

        Should clone/pull a git repo, download files, etc.
        Results should be stored in self.cache_dir.
        """
        ...

    async def load_skills(self) -> list[Skill]:
        """Load all .md skill files from the provider's root directory.

        Override `_skills_root` if the scan directory differs from cache_dir.
        """
        root = self._skills_root()
        if not root.exists():
            logger.warning(f'[{self.name}] Skills directory not found: {root}')
            return []
        return self._scan_markdown_files(root)

    def _skills_root(self) -> Path:
        """Return the root directory to scan for .md files. Override as needed."""
        return self.cache_dir

    def _scan_markdown_files(self, root: Path) -> list[Skill]:
        """Scan a directory tree for markdown skill files.

        Skips hidden directories and common non-skill files (README, LICENSE, etc.).
        """
        skills: list[Skill] = []
        for md_file in sorted(root.rglob('*.md')):
            rel = md_file.relative_to(root)
            # Skip hidden directories
            if any(part.startswith('.') for part in rel.parts):
                continue
            # Skip common non-skill files
            if md_file.stem.lower() in _SKIP_STEMS:
                continue
            try:
                content = md_file.read_text(encoding='utf-8')
                skills.append(Skill(
                    name=md_file.stem,
                    content=content,
                    source=f'{self.name}/{rel}',
                ))
            except Exception as e:
                logger.warning(f'[{self.name}] Failed to read {md_file}: {e}')
        logger.info(f'[{self.name}] Loaded {len(skills)} skills from {root}')
        return skills

    async def get_skills(self) -> list[Skill]:
        """Convenience method: fetch then load.

        Can be overridden if a provider wants custom logic.
        """
        await self.fetch()
        return await self.load_skills()
