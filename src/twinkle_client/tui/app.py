# Copyright (c) Twinkle Contributors. All rights reserved.
"""Twinkle TUI main application."""

from __future__ import annotations

import asyncio
from twinkle.utils.logger import get_logger
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding

from twinkle_client.tui.agent.core import AgentLoop
from twinkle_client.tui.agent.monitor import TrainingMonitor
from twinkle_client.tui.connection import LocalConnection
from twinkle_client.skills import LocalSkillProvider, ModelScopeSkillProvider, SkillManager
from twinkle_client.tui.widgets.chat import ChatPanel
from twinkle_client.tui.widgets.logs import LogPanel
from twinkle_client.tui.widgets.metrics import MetricsPanel
from twinkle_client.tui.widgets.status_bar import StatusBar

logger = get_logger()

# Timeout for remote skills fetching (seconds)
_SKILLS_FETCH_TIMEOUT = 10.0


class TwinkleTUI(App):
    """Main Textual application for Twinkle training control."""

    TITLE = 'Twinkle TUI'
    SUB_TITLE = 'ML Training Control'

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: auto 2fr 3fr;
        grid-columns: 2fr 1fr;
    }

    #status-bar {
        column-span: 2;
        height: 3;
    }

    #metrics {
        height: 100%;
    }

    #logs {
        height: 100%;
        row-span: 2;
    }

    #chat {
        height: 100%;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding('q', 'quit', 'Quit'),
        Binding('ctrl+p', 'toggle_metrics', 'Toggle Metrics'),
        Binding('ctrl+l', 'clear_logs', 'Clear Logs'),
    ]

    def __init__(
        self,
        run_id: str | None = None,
        llm_base_url: str = 'http://localhost:11434/v1',
        llm_model: str = 'qwen3.5',
        llm_api_key: str = 'not-needed',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.run_id = run_id
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self._connection: LocalConnection | None = None
        self._agent: AgentLoop | None = None
        self._monitor: TrainingMonitor | None = None
        self._bg_tasks: list[asyncio.Task] = []

    def compose(self) -> ComposeResult:
        yield StatusBar(id='status-bar')
        yield MetricsPanel(id='metrics')
        yield LogPanel(id='logs')
        yield ChatPanel(id='chat')

    async def on_mount(self) -> None:
        """Initialize connection, agent, and background tasks.

        Agent is created immediately (with empty skills) so user messages
        are never silently dropped. Skills are loaded asynchronously and
        injected once ready.
        """
        self._connection = LocalConnection()
        if self.run_id:
            self._connection.current_run_id = self.run_id

        # Create agent immediately (usable before skills are loaded)
        self._agent = AgentLoop(
            connection=self._connection,
            llm_base_url=self.llm_base_url,
            llm_model=self.llm_model,
            llm_api_key=self.llm_api_key,
            skills_prompt='',
        )
        self._agent.set_run_selected_callback(self._on_run_selected)
        self._agent.set_metrics_callback(self._handle_metrics_zoom)
        self._agent.set_select_metrics_callback(self._handle_select_metrics)

        self._monitor = TrainingMonitor(
            connection=self._connection,
            on_message=self._on_agent_message,
            llm_base_url=self.llm_base_url,
            llm_model=self.llm_model,
            llm_api_key=self.llm_api_key,
        )

        # Start background tasks
        self._bg_tasks = [
            asyncio.create_task(self._monitor.run()),
            asyncio.create_task(self._poll_logs()),
            asyncio.create_task(self._poll_metrics()),
            asyncio.create_task(self._load_skills_async()),
        ]

        # Show welcome message and initial status
        self._show_welcome()
        self._update_status_bar()

    def _show_welcome(self) -> None:
        """Show a welcome hint in the chat panel."""
        chat = self.query_one('#chat', ChatPanel)
        if self._connection and self._connection.current_run_id:
            chat.add_assistant_message(
                f'Monitoring run: [bold]{self._connection.current_run_id}[/]. '
                'Ask me anything about your training.'
            )
        else:
            chat.add_assistant_message(
                'Welcome! I can help you start, monitor, and control ML training. '
                'Try: "list my training runs" or "start a new GRPO training".'
            )

    async def _load_skills_async(self) -> None:
        """Load skills in background and inject into agent when ready."""
        manager = SkillManager()

        # 1. Bundled skills shipped inside the twinkle_client package (pip-installable)
        from pathlib import Path as _Path
        _bundled_skills = _Path(__file__).resolve().parent.parent / 'skills' / 'bundled'
        if _bundled_skills.is_dir():
            manager.register(LocalSkillProvider(skills_dir=_bundled_skills))

        # 2. User-local custom skills (~/.cache/twinkle/tui/skills/local/)
        manager.register(LocalSkillProvider())
        # 3. Community skills from ModelScope (remote, best-effort)
        manager.register(ModelScopeSkillProvider())
        try:
            await asyncio.wait_for(manager.load_all(), timeout=_SKILLS_FETCH_TIMEOUT)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f'Skills loading incomplete: {e}')
        skills_prompt = manager.format_for_prompt()
        if skills_prompt and self._agent:
            self._agent.inject_skills(skills_prompt)
            logger.info(f'Skills injected: {manager.get_skill_names()}')

    # ──────────────────────────────────────────────────────────────────────
    # Background polling
    # ──────────────────────────────────────────────────────────────────────

    async def _poll_metrics(self, interval: float = 3.0) -> None:
        """Poll metrics incrementally and update UI."""
        while True:
            try:
                if self._connection and self._connection.current_run_id:
                    run_id = self._connection.current_run_id
                    new_metrics = self._connection.get_new_metrics(run_id)
                    if new_metrics:
                        self.query_one('#metrics', MetricsPanel).append_metrics(new_metrics)
                        self._update_status_bar(latest_metrics=new_metrics[-1])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f'Metrics poll error: {e}')
            await asyncio.sleep(interval)

    async def _poll_logs(self, interval: float = 2.0) -> None:
        """Poll logs incrementally and push to LogPanel."""
        while True:
            try:
                if self._connection and self._connection.current_run_id:
                    new_logs = self._connection.get_new_logs(self._connection.current_run_id)
                    if new_logs:
                        log_panel = self.query_one('#logs', LogPanel)
                        for entry in new_logs:
                            log_panel.append_log(entry.get('msg', ''))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f'Logs poll error: {e}')
            await asyncio.sleep(interval)

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _on_agent_message(self, message: str) -> None:
        """Push agent/monitor messages to chat panel."""
        self.query_one('#chat', ChatPanel).add_assistant_message(message)

    def _handle_metrics_zoom(self, action: str, **kwargs) -> None:
        metrics_panel = self.query_one('#metrics', MetricsPanel)
        if action == 'reset':
            metrics_panel.reset_zoom()
        else:
            metrics_panel.zoom(**kwargs)

    def _handle_select_metrics(self, keys: list[str]) -> dict:
        """Handle select_metrics tool call from agent."""
        metrics_panel = self.query_one('#metrics', MetricsPanel)
        return metrics_panel.select_keys(keys)

    def _on_run_selected(self, run_id: str) -> None:
        """Handle run switch: reset offsets, clear metrics, update status."""
        if self._connection:
            self._connection.reset_offsets(run_id)
        self.query_one('#metrics', MetricsPanel).update_metrics([])
        self._update_status_bar()

    def _update_status_bar(self, latest_metrics: dict | None = None) -> None:
        """Update status bar from current connection state (single meta read)."""
        if not self._connection or not self._connection.current_run_id:
            return
        run_id = self._connection.current_run_id
        status_bar = self.query_one('#status-bar', StatusBar)

        # Single meta read for all status queries
        meta = self._connection.get_meta(run_id)
        if meta:
            model_id = meta.get('model_id')
            if model_id:
                status_bar.update_status(model=model_id)

            status = meta.get('status', 'unknown')
            state_map = {
                'running': 'training',
                'paused': 'paused',
                'stopped': 'completed',
                'completed': 'completed',
                'error': 'error',
            }
            status_bar.update_status(state=state_map.get(status, 'idle'))

        status_bar.update_status(run_id=run_id)

        # Step from latest metrics
        if latest_metrics:
            status_bar.update_status(
                step=latest_metrics.get('step'),
                total_steps=latest_metrics.get('total_steps'),
            )

    # ──────────────────────────────────────────────────────────────────────
    # User interaction
    # ──────────────────────────────────────────────────────────────────────

    async def on_chat_panel_user_submitted(self, event: ChatPanel.UserSubmitted) -> None:
        """Handle user input from chat panel."""
        if not self._agent:
            return
        chat = self.query_one('#chat', ChatPanel)
        chat.start_streaming()
        try:
            await self._agent.send(
                event.text,
                on_token=chat.append_stream,
                on_stream_reset=chat.reset_stream,
            )
        except Exception as e:
            chat.finish_streaming()
            chat.add_assistant_message(f'[Error] {e}')
            return
        chat.finish_streaming()
        await asyncio.sleep(0)

    def action_toggle_metrics(self) -> None:
        self.query_one('#metrics', MetricsPanel).toggle_class('hidden')

    def action_clear_logs(self) -> None:
        self.query_one('#logs', LogPanel).clear()

    def action_quit(self) -> None:
        """Quit with proper cleanup of background tasks."""
        self._cancel_tasks()
        self.exit()

    def _cancel_tasks(self) -> None:
        for task in self._bg_tasks:
            task.cancel()
        self._bg_tasks.clear()
