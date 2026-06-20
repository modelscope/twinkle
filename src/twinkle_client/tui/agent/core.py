# Copyright (c) Twinkle Contributors. All rights reserved.
"""Agent core - async LLM tool-calling loop using OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import Any, Callable

from openai import AsyncOpenAI

from twinkle_client.tui.agent.prompts import SYSTEM_PROMPT
from twinkle_client.tui.agent.tools import TOOL_SCHEMAS, ToolExecutor
from twinkle_client.tui.connection import LocalConnection


class AgentLoop:
    """Async tool-calling agent loop using OpenAI-compatible API.

    Manages conversation history, LLM calls, and tool execution.
    Includes automatic history pruning to prevent context overflow.
    """

    MAX_TOOL_ROUNDS = 10  # prevent infinite tool loops
    MAX_HISTORY_MESSAGES = 50  # keep last N messages (excluding system prompt)

    def __init__(
        self,
        connection: LocalConnection,
        on_message: Callable[[str], None],
        llm_base_url: str,
        llm_model: str,
        llm_api_key: str,
        skills_prompt: str = '',
    ):
        self.connection = connection
        self.on_message = on_message
        self.llm_model = llm_model
        self._client = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self._tool_executor = ToolExecutor(connection)
        # Build system prompt with optional skills section
        full_prompt = SYSTEM_PROMPT
        if skills_prompt:
            full_prompt = f'{SYSTEM_PROMPT}\n\n{skills_prompt}'
        self.history: list[dict[str, Any]] = [
            {'role': 'system', 'content': full_prompt},
        ]

    async def send(self, user_input: str) -> str:
        """Process user input through LLM with tool calling.

        Returns the final assistant text response.
        """
        self.history.append({'role': 'user', 'content': user_input})
        self._prune_history()

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = await self._call_llm()

            message = response.choices[0].message

            # If no tool calls, return the text response
            if not message.tool_calls:
                content = message.content or ''
                self.history.append({'role': 'assistant', 'content': content})
                return content

            # Process tool calls
            self.history.append({
                'role': 'assistant',
                'content': message.content or '',
                'tool_calls': [
                    {
                        'id': tc.id,
                        'type': 'function',
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = await self._tool_executor.execute(tc.function.name, args)
                self.history.append({
                    'role': 'tool',
                    'tool_call_id': tc.id,
                    'content': result,
                })

        # Exceeded max rounds
        fallback = 'I reached the maximum number of tool calls. Please try a simpler request.'
        self.history.append({'role': 'assistant', 'content': fallback})
        return fallback

    async def _call_llm(self):
        """Make a single LLM API call with tools."""
        return await self._client.chat.completions.create(
            model=self.llm_model,
            messages=self.history,
            tools=TOOL_SCHEMAS,
            tool_choice='auto',
        )

    def set_metrics_callback(self, callback: Callable) -> None:
        """Set the callback for metrics zoom control."""
        self._tool_executor.metrics_callback = callback

    def set_run_selected_callback(self, callback: Callable[[str], None]) -> None:
        """Set the callback invoked when the agent switches to a different run."""
        self._tool_executor.on_run_selected = callback

    def inject_skills(self, skills_prompt: str) -> None:
        """Inject skills into the system prompt after initial load.

        Called asynchronously once skills finish loading, so the agent
        is usable immediately even before skills are ready.
        """
        if not skills_prompt:
            return
        self.history[0]['content'] = f"{self.history[0]['content']}\n\n{skills_prompt}"

    def _prune_history(self) -> None:
        """Prune conversation history to prevent context overflow.

        Keeps the system prompt (index 0) and the most recent messages.
        Cuts at a 'user' message boundary to avoid splitting tool_call sequences.
        """
        if len(self.history) <= self.MAX_HISTORY_MESSAGES + 1:
            return
        # Find the nearest 'user' message at or after the ideal cut point
        cut_idx = len(self.history) - self.MAX_HISTORY_MESSAGES
        while cut_idx < len(self.history) and self.history[cut_idx]['role'] != 'user':
            cut_idx += 1
        if cut_idx >= len(self.history):
            return  # No safe cut point found; skip pruning this round
        self.history = [self.history[0]] + self.history[cut_idx:]
