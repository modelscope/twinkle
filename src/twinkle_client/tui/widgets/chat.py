# Copyright (c) Twinkle Contributors. All rights reserved.
"""Chat panel widget - handles user/agent conversation display and input."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message as TextualMessage
from textual.widgets import Input, RichLog, Static
from textual.widget import Widget


class ChatPanel(Widget):
    """Interactive chat panel for user <-> agent conversation."""

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

    ChatPanel > #thinking-indicator {
        dock: bottom;
        height: 1;
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    """

    class UserSubmitted(TextualMessage):
        """Event emitted when user submits a message."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def compose(self) -> ComposeResult:
        yield Static('Chat', id='chat-title')
        yield RichLog(id='chat-log', wrap=True, markup=True, max_lines=200)
        yield Static('', id='thinking-indicator')
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

    def set_thinking(self, thinking: bool) -> None:
        """Show/hide thinking indicator."""
        indicator = self.query_one('#thinking-indicator', Static)
        indicator.update('Agent is thinking...' if thinking else '')
