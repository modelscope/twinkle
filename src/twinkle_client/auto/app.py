# Copyright (c) Twinkle Contributors. All rights reserved.
"""Twinkle Auto — minimal chat-only interface for terminal use.

No complex UI framework (Textual/curses). Just an async input loop with
streaming LLM output. The TrainingMonitor still runs in background for
automatic error detection and script repair.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

from twinkle.utils.logger import get_logger

logger = get_logger()

_SKILLS_FETCH_TIMEOUT = 10.0

# ANSI color helpers
_CYAN = '\033[1;36m'
_GREEN = '\033[1;32m'
_YELLOW = '\033[1;33m'
_DIM = '\033[2m'
_RESET = '\033[0m'


class TwinkleAuto:
    """Minimal chat-only interface for terminal (SSH-friendly).

    Features retained:
    - Agent with tool-calling (start/stop/resume training, search, etc.)
    - TrainingMonitor: auto-detects crashes and fixes scripts
    - Skills loading (bundled + local + community)

    Removed (will be in WebUI):
    - Metrics chart rendering
    - Log panel
    - Status bar
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        llm_base_url: str = 'http://localhost:11434/v1',
        llm_model: str = 'qwen3.5',
        llm_api_key: str = 'not-needed',
    ):
        self.run_id = run_id
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self._connection = None
        self._agent = None
        self._monitor = None

    def run(self) -> None:
        """Entry point — blocks until user exits."""
        try:
            asyncio.run(self._main())
        except KeyboardInterrupt:
            pass

    async def _main(self) -> None:
        from twinkle_client.auto.agent.core import AgentLoop
        from twinkle_client.auto.agent.monitor import TrainingMonitor
        from twinkle_client.auto.connection import LocalConnection

        from openai import AsyncOpenAI

        # Connection
        self._connection = LocalConnection()
        if self.run_id:
            self._connection.current_run_id = self.run_id

        # Shared LLM client
        client = AsyncOpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)

        # Agent
        self._agent = AgentLoop(
            connection=self._connection,
            llm_client=client,
            llm_model=self.llm_model,
            skills_prompt='',
        )

        # Monitor (auto-fix mechanism — runs silently in background)
        self._monitor = TrainingMonitor(
            connection=self._connection,
            on_message=self._on_monitor_message,
            llm_client=client,
            llm_model=self.llm_model,
        )

        # Background tasks
        bg_tasks = [
            asyncio.create_task(self._monitor.run()),
            asyncio.create_task(self._load_skills()),
        ]

        # Welcome
        self._print_welcome()

        # Chat loop
        try:
            await self._chat_loop()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            for t in bg_tasks:
                t.cancel()
            print(f'\n{_DIM}Bye.{_RESET}')

    def _print_welcome(self) -> None:
        print(f'{_CYAN}Twinkle Auto{_RESET} — ML Training Control via Natural Language')
        if self._connection and self._connection.current_run_id:
            print(f'Monitoring run: {_CYAN}{self._connection.current_run_id}{_RESET}')
        else:
            print(f'{_DIM}Try: "list my training runs" or "start a new GRPO training"{_RESET}')
        print()

    async def _chat_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            try:
                user_input = await loop.run_in_executor(
                    None, lambda: input(f'{_GREEN}You:{_RESET} ')
                )
            except (KeyboardInterrupt, EOFError):
                break

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in ('exit', 'quit', 'q'):
                break

            # Stream response
            print(f'{_CYAN}Agent:{_RESET} ', end='', flush=True)
            try:
                await self._agent.send(
                    text,
                    on_token=self._on_token,
                    on_stream_reset=self._on_stream_reset,
                )
            except Exception as e:
                print(f'\n{_YELLOW}[Error]{_RESET} {e}')
            print()  # newline after response

    @staticmethod
    def _on_token(token: str) -> None:
        print(token, end='', flush=True)

    @staticmethod
    def _on_stream_reset(tool_names: list[str]) -> None:
        names = ', '.join(tool_names)
        print(f'\n  {_DIM}↳ calling {names}{_RESET}')
        print(f'{_CYAN}Agent:{_RESET} ', end='', flush=True)

    def _on_monitor_message(self, message: str) -> None:
        print(f'\n{_YELLOW}[Monitor]{_RESET} {message}')
        print(f'{_GREEN}You:{_RESET} ', end='', flush=True)

    async def _load_skills(self) -> None:
        from twinkle_client.skills import LocalSkillProvider, ModelScopeSkillProvider, SkillManager

        manager = SkillManager()
        bundled = Path(__file__).resolve().parent.parent / 'skills' / 'bundled'
        if bundled.is_dir():
            manager.register(LocalSkillProvider(skills_dir=bundled))
        manager.register(LocalSkillProvider())
        manager.register(ModelScopeSkillProvider())
        try:
            await asyncio.wait_for(manager.load_all(), timeout=_SKILLS_FETCH_TIMEOUT)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f'Skills loading incomplete: {e}')
        skills_prompt = manager.format_for_prompt()
        if skills_prompt and self._agent:
            self._agent.inject_skills(skills_prompt)
            logger.info(f'Skills injected: {manager.get_skill_names()}')
