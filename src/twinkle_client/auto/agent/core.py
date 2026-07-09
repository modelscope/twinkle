# Copyright (c) Twinkle Contributors. All rights reserved.
"""Agent core - async LLM tool-calling loop using OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
from twinkle.utils.logger import get_logger
from typing import Any, Callable

from twinkle_client.auto.agent.prompts import SYSTEM_PROMPT
from twinkle_client.auto.agent.tools import TOOL_SCHEMAS, ToolExecutor
from twinkle_client.auto.connection import LocalConnection

logger = get_logger()


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
        llm_client: 'AsyncOpenAI',
        llm_model: str,
        skills_prompt: str = '',
    ):
        self.connection = connection
        self.llm_model = llm_model
        self._client = llm_client
        self._tool_executor = ToolExecutor(connection)
        # Build system prompt with optional skills section
        full_prompt = SYSTEM_PROMPT
        if skills_prompt:
            full_prompt = f'{SYSTEM_PROMPT}\n\n{skills_prompt}'
        self.history: list[dict[str, Any]] = [
            {'role': 'system', 'content': full_prompt},
        ]

    async def send(
        self,
        user_input: str,
        on_token: Callable[[str], None] | None = None,
        on_stream_reset: Callable[[list[str]], None] | None = None,
    ) -> str:
        """Process user input through LLM with tool calling.

        Args:
            user_input: The user's message text.
            on_token: Callback for each streamed text chunk.
            on_stream_reset: Called when tool-calls are detected mid-stream,
                receives a list of tool names being invoked.

        Returns the final assistant text response.
        """
        logger.info(f'User message: {user_input[:200]}')
        self.history.append({'role': 'user', 'content': user_input})
        self._prune_history()

        for round_idx in range(self.MAX_TOOL_ROUNDS):
            logger.debug(f'LLM call round {round_idx + 1}/{self.MAX_TOOL_ROUNDS}')
            content, tool_calls = await self._call_llm_stream(on_token=on_token)

            # If no tool calls, we're done — content was correctly streamed
            if not tool_calls:
                self.history.append({'role': 'assistant', 'content': content})
                logger.info(f'Assistant response ({len(content)} chars): {content[:150]}...')
                return content

            # Tool calls detected — discard any leaked intermediate tokens
            tool_names = [tc['function']['name'] for tc in tool_calls]
            if on_stream_reset:
                on_stream_reset(tool_names)

            logger.info(f'Tool calls detected: {tool_names}')
            self.history.append({
                'role': 'assistant',
                'content': content,
                'tool_calls': tool_calls,
            })

            for tc in tool_calls:
                func_name = tc['function']['name']
                raw_args = tc['function']['arguments']
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError as e:
                    logger.error(f'Tool {func_name}: invalid JSON args: {e}\n  raw={raw_args[:500]}')
                    args = {}
                logger.info(f'Executing tool: {func_name}({", ".join(f"{k}={v!r}" for k, v in list(args.items())[:5])})')
                result = await self._tool_executor.execute(func_name, args)
                logger.debug(f'Tool {func_name} result ({len(result)} chars): {result[:300]}')
                self.history.append({
                    'role': 'tool',
                    'tool_call_id': tc['id'],
                    'content': result,
                })

        # Exceeded max rounds
        logger.warning(f'Exceeded max tool rounds ({self.MAX_TOOL_ROUNDS})')
        fallback = 'I reached the maximum number of tool calls. Please try a simpler request.'
        self.history.append({'role': 'assistant', 'content': fallback})
        return fallback

    async def _call_llm_stream(
        self,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Make a streaming LLM API call. Accumulates content and tool_calls.

        Streams text tokens via on_token only if the response has no tool calls.
        Returns (full_content, tool_calls_list).
        """
        try:
            stream = await self._client.chat.completions.create(
                model=self.llm_model,
                messages=self.history,
                tools=TOOL_SCHEMAS,
                tool_choice='auto',
                stream=True,
            )
        except Exception as e:
            logger.error(f'LLM API call failed: {type(e).__name__}: {e}')
            raise

        content_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, Any]] = {}
        has_tool_calls = False
        chunk_count = 0

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Accumulate content
            if delta.content:
                content_parts.append(delta.content)
                # Only stream tokens if no tool calls detected yet
                if not has_tool_calls and on_token:
                    on_token(delta.content)

            # Accumulate tool calls
            if delta.tool_calls:
                has_tool_calls = True
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            'id': '',
                            'type': 'function',
                            'function': {'name': '', 'arguments': ''},
                        }
                    tc = tool_calls_map[idx]
                    if tc_delta.id:
                        tc['id'] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc['function']['name'] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc['function']['arguments'] += tc_delta.function.arguments

            # Yield to event loop periodically to allow UI rendering
            chunk_count += 1
            if chunk_count % 5 == 0:
                await asyncio.sleep(0)

        content = ''.join(content_parts)
        tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)] if tool_calls_map else []
        return content, tool_calls

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
