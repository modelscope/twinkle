# Copyright (c) Twinkle Contributors. All rights reserved.
"""Chat panel widget - handles user/agent conversation display and input."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.message import Message as TextualMessage
from textual.widgets import Input, RichLog, Static
from textual.widget import Widget


class ChatPanel(Widget):
    """Interactive chat panel for user <-> agent conversation.

    Streaming text is written directly into the main chat-log (RichLog)
    in throttled chunks so the conversation flows naturally without
    a separate narrow preview widget.
    """

    DEFAULT_CSS = """
    ChatPanel {
        layout: vertical;
        border: solid $primary;
        padding: 0;
    }

    ChatPanel > #chat-title {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        text-align: center;
    }

    ChatPanel > #chat-log {
        height: 1fr;
        padding: 0 1;
    }

    ChatPanel > #chat-input {
        dock: bottom;
        height: 3;
        margin: 0 1;
    }
    """

    class UserSubmitted(TextualMessage):
        """Event emitted when user submits a message."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    # Minimum interval between flushing chunks to the RichLog
    _STREAM_THROTTLE = 0.08  # 80ms — balance between responsiveness and perf

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._streaming_buffer = ''  # un-flushed chars
        self._full_response = ''     # entire response accumulated
        self._is_streaming = False
        self._last_flush_time: float = 0.0
        self._header_written = False  # whether "Agent: " prefix is written

    def compose(self) -> ComposeResult:
        yield Static('Chat', id='chat-title')
        yield RichLog(id='chat-log', wrap=True, markup=True, max_lines=200)
        yield Input(placeholder='Ask the agent anything...', id='chat-input')

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        text = event.value.strip()
        if not text:
            return
        event.input.value = ''
        self.add_user_message(text)
        self.post_message(self.UserSubmitted(text))

    def add_user_message(self, text: str) -> None:
        """Add a user message to the chat log."""
        self.query_one('#chat-log', RichLog).write(f'[bold green]You:[/] {text}')

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to the chat log."""
        self.query_one('#chat-log', RichLog).write(f'[bold cyan]Agent:[/] {text}')

    # ── Streaming API ──

    def start_streaming(self) -> None:
        """Begin a streaming assistant response."""
        self._streaming_buffer = ''
        self._full_response = ''
        self._is_streaming = True
        self._last_flush_time = 0.0
        self._header_written = False

    def reset_stream(self) -> None:
        """Discard buffered streaming content (called when tool-calls detected).

        Resets state so the next LLM round starts fresh.
        """
        self._streaming_buffer = ''
        self._full_response = ''
        self._header_written = False
        log = self.query_one('#chat-log', RichLog)
        log.write('[dim]  ↳ calling tools...[/]')

    def append_stream(self, chunk: str) -> None:
        """Append a chunk from the LLM stream.

        Writes accumulated text to the chat-log in throttled batches
        so the conversation scrolls naturally.
        """
        self._streaming_buffer += chunk
        self._full_response += chunk
        now = time.monotonic()
        if now - self._last_flush_time >= self._STREAM_THROTTLE:
            self._flush_stream()

    def _flush_stream(self, force: bool = False) -> None:
        """Write buffered streaming text to the RichLog.

        Only flushes complete lines (up to the last newline) to avoid
        splitting multi-line structures like tables mid-row.
        If force=True, flushes everything (used at end of stream).
        """
        if not self._streaming_buffer:
            return

        if force:
            text_to_write = self._streaming_buffer
            self._streaming_buffer = ''
        else:
            # Only flush up to the last newline — keep incomplete line buffered
            last_nl = self._streaming_buffer.rfind('\n')
            if last_nl == -1:
                return  # No complete line yet, keep buffering
            text_to_write = self._streaming_buffer[:last_nl + 1]
            self._streaming_buffer = self._streaming_buffer[last_nl + 1:]

        if not text_to_write:
            return

        log = self.query_one('#chat-log', RichLog)
        if not self._header_written:
            log.write(f'[bold cyan]Agent:[/] {text_to_write}', shrink=False)
            self._header_written = True
        else:
            log.write(text_to_write, shrink=False)
        self._last_flush_time = time.monotonic()

    def finish_streaming(self) -> str:
        """End streaming and return the full accumulated response."""
        # Force-flush any remaining buffer (including incomplete lines)
        self._flush_stream(force=True)
        self._is_streaming = False
        full_text = self._full_response
        self._full_response = ''
        self._streaming_buffer = ''
        return full_text
