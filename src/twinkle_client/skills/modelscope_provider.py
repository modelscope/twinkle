# Copyright (c) Twinkle Contributors. All rights reserved.
"""ModelScope skill provider - fetches skills from modelscope-skills GitHub repo."""

from __future__ import annotations

import asyncio
from twinkle.utils.logger import get_logger
from pathlib import Path

from twinkle_client.skills.base import SkillProvider

logger = get_logger()

_DEFAULT_REPO_URL = 'https://github.com/modelscope/modelscope-skills.git'
_DEFAULT_BRANCH = 'main'


class ModelScopeSkillProvider(SkillProvider):
    """Fetches skill markdown files from the modelscope-skills GitHub repository.

    Skills are cloned to a local cache directory. On subsequent calls,
    the repo is pulled to get updates.
    """

    def __init__(
        self,
        repo_url: str = _DEFAULT_REPO_URL,
        branch: str = _DEFAULT_BRANCH,
        cache_dir: Path | None = None,
    ):
        self._repo_url = repo_url
        self._branch = branch
        super().__init__(cache_dir=cache_dir)

    @property
    def name(self) -> str:
        return 'modelscope'

    async def fetch(self) -> None:
        """Clone or pull the modelscope-skills repository."""
        repo_dir = self.cache_dir / 'repo'

        if (repo_dir / '.git').exists():
            proc = await asyncio.create_subprocess_exec(
                'git', '-C', str(repo_dir), 'pull', '--ff-only',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(f'git pull failed: {stderr.decode().strip()}')
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                'git', 'clone', '--depth', '1', '--branch', self._branch,
                self._repo_url, str(repo_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error(f'git clone failed: {stderr.decode().strip()}')

    def _skills_root(self) -> Path:
        return self.cache_dir / 'repo'
